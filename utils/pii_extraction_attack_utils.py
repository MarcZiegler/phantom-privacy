"""Helpers for the Enron cross-user PII extraction attack (Phase B)."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.enron_pii_utils import (
    build_distractor_pool,
    canonical_phone_digits,
    find_exposure_contexts,
    is_held_in,
    is_held_out,
)
from utils.synthetic_distractors import SyntheticDistractorGenerator


def strip_email_domain_from_name_values(
    all_ground_truth: dict[str, dict[str, dict[str, object]]],
) -> None:
    """Strip email domains from graph-derived name values.

    The PII graph's ``has_name`` fallback stores the user ID, which for Enron
    is an email address. This normalizes it to the local part so the ``name``
    target is comparable to Mallinckrodt names.
    """
    for user_gt in all_ground_truth.values():
        meta = user_gt.get("name")
        if meta is None:
            continue
        value = meta.get("value")
        if value is not None and "@" in str(value):
            meta["value"] = str(value).split("@")[0]
        raw_value = meta.get("raw_value", value)
        if raw_value is not None and "@" in str(raw_value):
            meta["raw_value"] = str(raw_value).split("@")[0]


def _canonical_value(value: str, pii_type: str) -> str:
    """Return a normalized string used to deduplicate distractor pools."""
    if pii_type == "email":
        return value.strip().lower()
    if pii_type in ("phone", "fax"):
        return canonical_phone_digits(value)
    return value.strip().lower()


def sample_distractors(
    all_ground_truth: dict[str, dict[str, dict[str, object]]],
    pii_type: str,
    target_value: str,
    num_distractors: int,
    rng: np.random.Generator,
    distractor_mode: str = "real",
    synthetic_generator: SyntheticDistractorGenerator | None = None,
) -> list[str]:
    """Sample same-type distractors for a target value.

    Args:
        all_ground_truth: Per-user ground-truth PII dict.
        pii_type: PII type being attacked.
        target_value: The true value to exclude from the distractor pool.
        num_distractors: Number of distractors to return.
        rng: NumPy random generator.
        distractor_mode: ``"real"`` uses other users' true values; ``"synthetic"``
            generates plausible synthetic values.
        synthetic_generator: Generator for synthetic distractors. Required when
            ``distractor_mode="synthetic"``.

    Returns:
        A list of distractor strings.
    """
    if distractor_mode == "synthetic":
        if synthetic_generator is None:
            raise ValueError(
                "distractor_mode='synthetic' requires a SyntheticDistractorGenerator"
            )
        return synthetic_generator.sample(pii_type, target_value, num_distractors)

    pool_values = [
        meta["value"]
        for gt in all_ground_truth.values()
        for pt, meta in gt.items()
        if pt == pii_type and meta.get("value") is not None
    ]
    pool = build_distractor_pool(pool_values, pii_type, target_value=target_value)
    if len(pool) <= num_distractors:
        return pool
    return rng.choice(pool, size=num_distractors, replace=False).tolist()


def score_candidates(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    candidates: list[str],
    device: torch.device,
    max_seq_length: int,
) -> list[tuple[float, int]]:
    """Return average per-token negative log-likelihood for each candidate.

    The score for candidate ``c`` is
    ``loss(prefix + c) - loss(prefix)``, normalized by the number of tokens
    in ``c`` as tokenized in isolation. Lower values mean higher probability
    under the model.

    Using the standalone candidate token count avoids the zero/negative-token
    problem caused by BPE boundary merging between ``prefix`` and ``c``.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer compatible with ``model``.
        prefix: Prompt prefix ending with the mask token.
        candidates: Candidate strings to score.
        device: Torch device.
        max_seq_length: Maximum sequence length for the full prefix+candidate.

    Returns:
        A list of ``(avg_per_token_loss, num_candidate_tokens)`` tuples.
    """
    model.eval()

    # Total loss of the shared prefix.
    prefix_ids = tokenizer.encode(
        prefix,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    ).to(device)
    prefix_len = prefix_ids.shape[1]
    if prefix_len == 0:
        return [(float("inf"), 0) for _ in candidates]

    with torch.no_grad():
        prefix_outputs = model(input_ids=prefix_ids, labels=prefix_ids)
    prefix_total_loss = float(prefix_outputs.loss.item()) * prefix_len

    full_texts = [prefix + c for c in candidates]
    enc = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        max_length=max_seq_length,
        truncation=True,
    ).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    loss_mask = attention_mask[:, 1:].contiguous()

    per_token_nll = functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
    ).view(labels.size())
    per_seq_total = (per_token_nll * loss_mask.float()).sum(dim=1)

    results: list[tuple[float, int]] = []
    for i, candidate in enumerate(candidates):
        candidate_total = per_seq_total[i].item() - prefix_total_loss
        # Use the standalone candidate token count as the denominator. This is
        # robust to BPE boundary merging and prevents zero/negative denominators.
        candidate_ids = tokenizer.encode(
            candidate, add_special_tokens=False, max_length=max_seq_length, truncation=True
        )
        candidate_tokens = len(candidate_ids)
        if candidate_tokens <= 0:
            results.append((float("inf"), 0))
        else:
            results.append((candidate_total / candidate_tokens, candidate_tokens))
    return results


def _compute_auroc(
    labels: list[int],
    scores: list[float],
) -> float | None:
    """Compute AUROC, returning None if there is only one class.

    Rows with non-finite scores are dropped so a single malformed row cannot
    crash the aggregate metric computation.
    """
    pairs = [
        (lab, score)
        for lab, score in zip(labels, scores, strict=False)
        if math.isfinite(score)
    ]
    if len(pairs) < 2:
        return None
    filtered_labels, filtered_scores = zip(*pairs, strict=False)
    if len(set(filtered_labels)) < 2:
        return None
    return float(roc_auc_score(filtered_labels, filtered_scores))


def _mask_values_in_prefix(prefix: str, values: Iterable[str], placeholder: str = "<MASK>") -> str:
    """Replace occurrences of sensitive values in a prompt prefix.

    Longer values are replaced first so that shorter values do not create
    partial substitutions inside longer ones. Matching is case-insensitive
    but preserves the surrounding text.
    """
    unique_values = sorted(
        {str(v).strip() for v in values if v is not None and str(v).strip()},
        key=len,
        reverse=True,
    )
    result = prefix
    for value in unique_values:
        pattern = re.compile(re.escape(value), re.IGNORECASE)
        result = pattern.sub(placeholder, result)
    return result


def compute_extraction_metrics(
    scored_rows: list[dict],
    score_key: str = "score",
) -> dict:
    """Compute aggregate AUROC metrics from scored candidate rows.

    Each row is expected to contain at least ``label`` (0 or 1) and a score
    field named by ``score_key`` (default ``"score"``), plus optional
    ``group``, ``pii_type``, and ``user`` keys.

    Args:
        scored_rows: Rows produced by Phase B scoring.
        score_key: Key in each row holding the numeric score.

    Returns:
        A nested dict with global, per-group, per-PII-type, and per-user AUROC.
    """
    filtered_rows = [
        r for r in scored_rows if math.isfinite(r.get(score_key, float("nan")))
    ]
    labels = [r["label"] for r in filtered_rows]
    scores = [r[score_key] for r in filtered_rows]
    metrics: dict = {
        "global": {
            "auroc": _compute_auroc(labels, scores),
            "num_true": sum(labels),
            "num_false": len(labels) - sum(labels),
        }
    }

    by_group: dict[str, list[tuple[int, float]]] = defaultdict(list)
    by_pii_type: dict[str, list[tuple[int, float]]] = defaultdict(list)
    by_user: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in filtered_rows:
        label_score = (row["label"], row[score_key])
        if "group" in row:
            by_group[row["group"]].append(label_score)
        if "pii_type" in row:
            by_pii_type[row["pii_type"]].append(label_score)
        if "user" in row:
            by_user[row["user"]].append(label_score)

    metrics["by_group"] = {
        group: {
            "auroc": _compute_auroc([ls[0] for ls in rows], [ls[1] for ls in rows]),
            "num_true": sum(ls[0] for ls in rows),
            "num_false": len(rows) - sum(ls[0] for ls in rows),
        }
        for group, rows in sorted(by_group.items())
    }

    metrics["by_pii_type"] = {
        pii_type: {
            "auroc": _compute_auroc([ls[0] for ls in rows], [ls[1] for ls in rows]),
            "num_true": sum(ls[0] for ls in rows),
            "num_false": len(rows) - sum(ls[0] for ls in rows),
        }
        for pii_type, rows in sorted(by_pii_type.items())
    }

    user_aurocs: list[float] = []
    for _user, rows in by_user.items():
        auroc = _compute_auroc([ls[0] for ls in rows], [ls[1] for ls in rows])
        if auroc is not None:
            user_aurocs.append(auroc)
    if user_aurocs:
        metrics["per_user_auroc"] = {
            "mean": float(np.mean(user_aurocs)),
            "median": float(np.median(user_aurocs)),
            "min": float(np.min(user_aurocs)),
            "max": float(np.max(user_aurocs)),
            "std": float(np.std(user_aurocs)),
            "num_users": len(user_aurocs),
        }
    else:
        metrics["per_user_auroc"] = None

    return metrics


def run_phase_b(
    *,
    data_path: Path,
    model_path: Path,
    base_model_name: str | None,
    all_ground_truth: dict[str, dict[str, dict[str, object]]],
    split_map: dict[str, set[str]],
    records: list[dict],
    pii_types: list[str],
    num_distractors: int,
    max_samples: int,
    max_context_tokens: int,
    mask_token: str,
    context_mode: str,
    max_seq_length: int | None,
    device_preference: str,
    seed: int,
    exposure_split: str = "train",
    exposure_from_held_in_only: bool = False,
    exposure_from_target_only: bool = False,
    distractor_mode: str = "real",
    email_local_only: bool = False,
    mask_pii_types_in_prefix: list[str] | None = None,
    mask_placeholder: str = "<MASK>",
    bert_model_path: str | Path | None = None,
) -> tuple[list[dict], dict]:
    """Run Phase B model-based extraction scoring and return rows + metrics.

    Args:
        data_path: Path to the preprocessed dataset (for result metadata).
        model_path: Path to the fine-tuned model or LoRA adapter.
        base_model_name: Base model identifier; inferred from adapter config if None.
        all_ground_truth: Per-user ground-truth PII dict (already filtered).
        split_map: Mapping from user to set of splits.
        records: Full list of records.
        pii_types: PII types to score.
        num_distractors: Number of distractors to sample per true value.
        max_samples: Maximum exposure contexts to score per user/type.
        max_context_tokens: Maximum prefix tokens before the masked value.
        mask_token: String used to mark the blank.
        max_seq_length: Tokenization/truncation limit; inferred from the model if None.
        device_preference: ``auto``, ``cuda``, or ``cpu``.
        seed: Random seed for distractor sampling.
        exposure_split: Dataset split used to find exposure contexts (default ``train``).
        exposure_from_held_in_only: If True, restrict exposure contexts to records
            authored by held-in users.
        exposure_from_target_only: If True, restrict exposure contexts to records
            authored by the target user herself. This tests inference from the
            user's own held-out records.
        distractor_mode: ``"real"`` uses other users' true values; ``"synthetic"``
            generates plausible synthetic values.
        email_local_only: If True, strip the domain from synthetic email distractors
            so they match the local-part target format.
        mask_pii_types_in_prefix: Optional list of PII types whose values for the
            target user should be redacted from the prefix before scoring. This
            prevents the model from copying the target from earlier mentions.
        mask_placeholder: String used to replace redacted values in the prefix.

    Returns:
        ``(scored_rows, metrics)`` where ``scored_rows`` is a list of dicts and
        ``metrics`` is the aggregate metric dict.
    """
    from utils.bert_reconstruction import load_bert_model, score_candidates_with_mlm
    from utils.model_utils import get_device, load_base_model, load_finetuned_model

    rng = np.random.default_rng(seed)
    device = get_device(device_preference)

    skipped_rows = 0

    print(f"\nLoading fine-tuned model from {model_path} ...")
    finetuned_model, tokenizer = load_finetuned_model(
        model_path,
        base_model_name=base_model_name,
        device=device,
    )
    print("Loading base model for reference ...")
    base_model, _ = load_base_model(
        base_model_name or "EleutherAI/gpt-neo-125M",
        device=device,
    )

    bert_model = None
    bert_tokenizer = None
    if bert_model_path is not None:
        bert_path = str(bert_model_path)
        print(f"Loading BERT model from {bert_path} ...")
        bert_model, bert_tokenizer = load_bert_model(bert_path, device=device)

    if max_seq_length is None:
        max_seq_length = getattr(
            finetuned_model.config,
            "max_position_embeddings",
            tokenizer.model_max_length,
        )
        print(f"Using model max_seq_length={max_seq_length}")

    exposure_records = [r for r in records if r.get("split") == exposure_split]
    if exposure_from_held_in_only:
        exposure_records = [
            r
            for r in exposure_records
            if is_held_in(split_map.get(r.get("user", ""), set()))
        ]
    users = sorted(all_ground_truth.keys())

    synthetic_generator: SyntheticDistractorGenerator | None = None
    if distractor_mode == "synthetic":
        synthetic_generator = SyntheticDistractorGenerator(
            all_ground_truth, pii_types, seed=seed
        )

    scored_rows: list[dict] = []
    for user in users:
        group = "held_out" if is_held_out(split_map[user]) else "held_in"
        gt = all_ground_truth[user]

        # Optionally restrict contexts to the target user's own records.
        context_records = exposure_records
        if exposure_from_target_only:
            context_records = [r for r in exposure_records if r.get("user") == user]

        for pii_type in pii_types:
            if pii_type not in gt:
                continue
            meta = gt[pii_type]
            value = meta.get("value")
            if value is None:
                continue
            raw_value = meta.get("raw_value", value)

            contexts = find_exposure_contexts(
                context_records,
                target_user=user,
                value=value,
                raw_value=raw_value,
                pii_type=pii_type,
                tokenizer=tokenizer,
                max_samples=max_samples,
                max_context_tokens=max_context_tokens,
                mask_token=mask_token,
                skip_own_records=not exposure_from_target_only,
                context_mode=context_mode,
            )
            if not contexts:
                print(f"  {user} {pii_type}: no exposure contexts; skipping")
                continue

            # Optionally redact other PII of the target user from the prefix so the
            # model cannot trivially copy it from earlier mentions.
            if mask_pii_types_in_prefix:
                prefix_mask_values: list[str] = []
                for mt in mask_pii_types_in_prefix:
                    if mt in gt:
                        meta = gt[mt]
                        for key in ("value", "raw_value"):
                            val = str(meta.get(key, "")).strip()
                            if val:
                                prefix_mask_values.append(val)
                if prefix_mask_values:
                    contexts = [
                        _mask_values_in_prefix(ctx, prefix_mask_values, mask_placeholder)
                        for ctx in contexts
                    ]

            distractors = sample_distractors(
                all_ground_truth,
                pii_type,
                target_value=value,
                num_distractors=num_distractors,
                rng=rng,
                distractor_mode=distractor_mode,
                synthetic_generator=synthetic_generator,
            )
            if email_local_only and pii_type == "email":
                distractors = [d.split("@")[0] for d in distractors]
            candidates = [value] + distractors

            print(
                f"  Scoring {user} {pii_type}: {len(contexts)} contexts, "
                f"{len(candidates)} candidates"
            )
            for ctx_idx, prefix in enumerate(contexts):
                if ctx_idx > 0 and ctx_idx % 25 == 0:
                    print(f"    ... context {ctx_idx}/{len(contexts)}")
                ft_losses = score_candidates(
                    finetuned_model,
                    tokenizer,
                    prefix,
                    candidates,
                    device,
                    max_seq_length,
                )
                base_losses = score_candidates(
                    base_model,
                    tokenizer,
                    prefix,
                    candidates,
                    device,
                    max_seq_length,
                )
                bert_losses: list[tuple[float, int] | None] = [None] * len(candidates)
                if bert_model is not None and bert_tokenizer is not None:
                    bert_losses = score_candidates_with_mlm(
                        prefix,
                        candidates,
                        bert_model,
                        bert_tokenizer,
                        device,
                        max_seq_length,
                    )
                for cand, (ft_loss, ft_toks), (base_loss, base_toks), bert_item in zip(
                    candidates, ft_losses, base_losses, bert_losses, strict=True
                ):
                    if (
                        not math.isfinite(ft_loss)
                        or not math.isfinite(base_loss)
                        or ft_toks <= 0
                        or base_toks <= 0
                    ):
                        skipped_rows += 1
                        continue

                    # Higher score means the fine-tuned model prefers this
                    # candidate relative to the base model.
                    score = base_loss - ft_loss
                    row: dict[str, Any] = {
                        "user": user,
                        "group": group,
                        "pii_type": pii_type,
                        "context_idx": ctx_idx,
                        "candidate": cand,
                        "label": 1 if _canonical_value(cand, pii_type) == _canonical_value(value, pii_type) else 0,
                        "score": score,
                        "finetuned_loss": ft_loss,
                        "base_loss": base_loss,
                        "finetuned_tokens": ft_toks,
                        "base_tokens": base_toks,
                    }
                    if bert_item is not None:
                        bert_pll, bert_toks = bert_item
                        if math.isfinite(bert_pll) and bert_toks > 0:
                            row["bert_pll"] = bert_pll
                            row["bert_score"] = -bert_pll
                    scored_rows.append(row)

    if skipped_rows:
        print(f"  Skipped {skipped_rows} non-finite candidate scores")

    metrics = compute_extraction_metrics(scored_rows)
    if bert_model is not None:
        metrics["bert"] = compute_extraction_metrics(scored_rows, score_key="bert_score")
    metrics["num_skipped_rows"] = skipped_rows
    metrics["num_scored_rows"] = len(scored_rows)
    return scored_rows, metrics


def save_phase_b_results(
    scored_rows: list[dict],
    metrics: dict,
    output_dir: Path,
    config: dict,
) -> None:
    """Save Phase B scored rows, metrics, and config."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with (output_dir / "extraction_scores.jsonl").open("w", encoding="utf-8") as f:
        for row in scored_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nPhase B metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nPhase B results saved to {output_dir}")
