"""Shared utilities for post-training utility evaluation.

Computes perplexity / cross-entropy loss on held-out splits such as ``test``
and ``holdout``. These metrics are independent of the attack pipeline and are
meant to quantify the privacy-utility trade-off.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_utils import group_by_user, load_jsonl


def load_split_texts(
    data_path: Path,
    split: str,
    max_users: int | None = None,
) -> list[str]:
    """Load records with the given split label and return their texts.

    Args:
        data_path: Path to the preprocessed JSONL file.
        split: Split label to filter on (e.g. ``"test"`` or ``"holdout"``).
        max_users: If given, cap evaluation to the first ``max_users`` users
            (sorted lexicographically) in the split.

    Returns:
        A list of non-empty text strings for the requested split.
    """
    print(f"Loading {split} data from {data_path} ...")
    records = list(load_jsonl(data_path))
    print(f"Total records: {len(records):,}")

    split_recs = [r for r in records if r.get("split") == split]
    print(f"{split.capitalize()} records: {len(split_recs):,}")

    if not split_recs:
        return []

    if max_users is not None:
        user_records = group_by_user(split_recs)
        selected = sorted(user_records.keys())[:max_users]
        split_recs = [r for r in split_recs if r.get("user") in selected]
        print(f"After max_users={max_users}: {len(split_recs):,} records")

    texts = [r.get("text", "") for r in split_recs if r.get("text")]
    print(f"Final {split} texts: {len(texts):,}")
    return texts


def evaluate_texts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int = 8,
) -> dict[str, float | int]:
    """Compute mean cross-entropy loss and perplexity on a list of texts.

    The loss is accumulated per-token across batches so that the final average
    is not biased by the last partial batch.

    Args:
        model: Fine-tuned causal language model.
        tokenizer: Tokenizer matching the model.
        texts: List of text strings to evaluate.
        device: Device the model lives on.
        max_length: Maximum sequence length for truncation/padding.
        batch_size: Number of texts per forward pass.

    Returns:
        Dictionary with ``loss``, ``perplexity``, ``n_records``, and
        ``n_tokens``.
    """
    if not texts:
        raise ValueError("No texts provided for evaluation.")

    model.eval()
    pad_token_id = tokenizer.pad_token_id
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = input_ids.clone()
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Compute log-probabilities in float32 for numerical stability.
            log_probs = functional.log_softmax(logits.float(), dim=-1)
            nll = functional.nll_loss(
                log_probs.view(-1, log_probs.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
                ignore_index=-100,
            )

            effective_tokens = (shift_labels != -100).sum().item()
            total_nll += nll.item()
            total_tokens += effective_tokens

    mean_loss = total_nll / max(1, total_tokens)
    perplexity = math.exp(mean_loss)
    return {
        "loss": float(mean_loss),
        "perplexity": float(perplexity),
        "n_records": len(texts),
        "n_tokens": int(total_tokens),
    }


def save_utility_metrics(output_path: Path, metrics: dict) -> None:
    """Write utility metrics to JSON.

    Args:
        output_path: Destination file path.
        metrics: Dictionary of metrics to serialize.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Utility metrics saved to {output_path}")


def format_utility_results(results: dict[str, dict]) -> str:
    """Return a human-readable summary of per-split utility metrics."""
    lines = ["Utility evaluation results:"]
    for split, metrics in results.items():
        lines.append(
            f"  {split}: loss={metrics['loss']:.4f}, "
            f"perplexity={metrics['perplexity']:.2f}, "
            f"records={metrics['n_records']}, "
            f"tokens={metrics['n_tokens']:,}"
        )
    return "\n".join(lines)
