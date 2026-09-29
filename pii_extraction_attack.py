#!/usr/bin/env python3
"""Cross-user PII extraction attack for Enron.

Phase A (``--exposure-only``): classify PII values per user and count how
often each value appears in the training split of *other* users. This is the
first step before running the model-based extraction in Phase B.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.data_utils import group_by_user
from utils.enron_pii_utils import (
    PII_TYPES,
    build_user_split_map,
    compute_exposure,
    extract_user_ground_truth,
    filter_shared_and_placeholder_pii,
    is_held_in,
    is_held_out,
    load_records,
)
from utils.model_utils import get_device
from utils.pii_extraction_attack_utils import (
    run_phase_b,
    save_phase_b_results,
    strip_email_domain_from_name_values,
)
from utils.pii_graph_utils import (
    GRAPH_PII_TYPES,
    build_ground_truth_from_graph,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _summarize_exposure(rows: list[dict]) -> dict:
    """Build an aggregate summary from the exposure table."""
    summary: dict = {
        "num_users": len({r["user"] for r in rows}),
        "num_held_in": len({r["user"] for r in rows if r["group"] == "held_in"}),
        "num_held_out": len({r["user"] for r in rows if r["group"] == "held_out"}),
        "num_records": len(rows),
    }

    per_type: dict[str, dict] = {}
    present_types = sorted({r["pii_type"] for r in rows})
    for pii_type in present_types:
        all_type_rows = [r for r in rows if r["pii_type"] == pii_type]
        type_rows = [r for r in all_type_rows if r["value"] is not None]
        if not all_type_rows:
            continue

        excluded = {
            "placeholder": sum(
                1 for r in all_type_rows if r.get("excluded_reason") == "placeholder"
            ),
            "shared": sum(
                1 for r in all_type_rows if r.get("excluded_reason") == "shared"
            ),
            "toll_free": sum(
                1 for r in all_type_rows if r.get("excluded_reason") == "toll_free"
            ),
        }

        per_type[pii_type] = {"excluded": excluded}
        for group in ("held_in", "held_out"):
            group_rows = [r for r in type_rows if r["group"] == group]
            exposures = [r["exposure_total"] for r in group_rows]
            per_type[pii_type][group] = {
                "num_values": len(group_rows),
                "mean_exposure": float(np.mean(exposures)) if exposures else 0.0,
                "median_exposure": float(np.median(exposures)) if exposures else 0.0,
                "max_exposure": int(max(exposures)) if exposures else 0,
                "zero_exposure": sum(1 for e in exposures if e == 0),
            }
    summary["per_pii_type"] = per_type
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-user PII extraction attack for Enron."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to preprocessed JSONL.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to fine-tuned model dir (required for Phase B).",
    )
    parser.add_argument("--base-model", type=str, default=None, help="Base model name.")
    parser.add_argument(
        "--name-map",
        type=Path,
        default=None,
        help="Optional JSONL/JSON mapping canonical email -> display name.",
    )
    parser.add_argument(
        "--pii-graph",
        type=Path,
        default=None,
        help="Optional PII graph JSON. If given, ground truth is read from the graph instead of regex/signature extraction.",
    )
    parser.add_argument(
        "--pii-types",
        type=str,
        nargs="+",
        default=["email", "phone"],
        help="PII types to extract (default: email phone).",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=2,
        help="Minimum occurrences of a phone/fax/address candidate in a user's records.",
    )
    parser.add_argument(
        "--prompt-set",
        type=str,
        default="D",
        choices=["A", "B", "D", "all"],
        help="Prompt set for Phase B (ignored in --exposure-only).",
    )
    parser.add_argument(
        "--num-distractors",
        type=int,
        default=20,
        help="Number of distractors per PII value (Phase B).",
    )
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override the tokenization/truncation length (Phase B).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=256,
        help="Maximum number of prefix tokens kept before the masked PII value "
             "in Phase B cloze prompts (default: 256).",
    )
    parser.add_argument(
        "--mask-token",
        type=str,
        default="",
        help="Token appended before the candidate in Phase B prompts. "
             "Empty means no separator (default: empty).",
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        default="any",
        choices=["any", "header", "structured"],
        help="Where to extract Phase B contexts: any occurrence (any), only "
             "main To/Cc/Bcc header fields (header), or headers + quoted/forwarded "
             "headers + signatures (structured) (default: any).",
    )
    parser.add_argument(
        "--exposure-split",
        type=str,
        default="train",
        choices=["train", "attacker", "holdout", "test", "val"],
        help="Dataset split used to find exposure contexts (default: train).",
    )
    parser.add_argument(
        "--exposure-from-held-in-only",
        action="store_true",
        help="Only use exposure contexts authored by held-in users.",
    )
    parser.add_argument(
        "--exposure-from-target-only",
        action="store_true",
        help="Only use exposure contexts authored by the target user herself. "
             "Useful for testing whether the model can infer a held-out user's PII "
             "from the user's own unseen records.",
    )
    parser.add_argument(
        "--distractor-mode",
        type=str,
        default="real",
        choices=["real", "synthetic"],
        help="Distractor source: real other-user values or synthetic values "
             "(default: real).",
    )
    parser.add_argument(
        "--exposure-only",
        action="store_true",
        help="Run only Phase A: classify PII and count train-split exposure.",
    )
    parser.add_argument(
        "--email-local-only",
        action="store_true",
        help="For email PII, use only the local part (before @) as the target value.",
    )
    parser.add_argument(
        "--mask-pii-types-in-prefix",
        type=str,
        default=None,
        help="Comma-separated list of PII types to redact from the scoring prefix "
             "for the target user, e.g. 'name,phone,job_title'. Prevents the model "
             "from copying the target from earlier mentions.",
    )
    parser.add_argument(
        "--mask-placeholder",
        type=str,
        default="<MASK>",
        help="Placeholder used when redacting values from the prefix "
             "(default: <MASK>).",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--bert-model",
        type=str,
        default=None,
        help="Optional masked LM path or HuggingFace ID for BERT-based candidate "
             "reranking. If given, extraction_scores.jsonl and metrics.json will "
             "also contain bert_pll and bert_score.",
    )
    args = parser.parse_args()

    if args.mask_pii_types_in_prefix is not None:
        args.mask_pii_types_in_prefix = [
            t.strip() for t in args.mask_pii_types_in_prefix.split(",") if t.strip()
        ]

    set_seed(args.seed)

    if not args.exposure_only and args.model_path is None:
        parser.error("--model-path is required unless --exposure-only is set.")

    # Validate PII types.
    allowed_types = set(PII_TYPES)
    if args.pii_graph is not None:
        allowed_types |= GRAPH_PII_TYPES
    invalid = [t for t in args.pii_types if t not in allowed_types]
    if invalid:
        sys.exit(
            f"ERROR: unsupported PII types: {invalid}. "
            f"Supported: {sorted(allowed_types)}"
        )

    print(f"Loading data from {args.data} ...")
    records = load_records(args.data)
    print(f"Total records: {len(records):,}")

    user_records = group_by_user(records)
    print(f"Unique users: {len(user_records):,}")

    split_map = build_user_split_map(records)
    held_in_users = sorted(u for u, s in split_map.items() if is_held_in(s))
    held_out_users = sorted(u for u, s in split_map.items() if is_held_out(s))
    print(f"Held-in users: {len(held_in_users):,}")
    print(f"Held-out users: {len(held_out_users):,}")

    users = held_in_users + held_out_users
    if args.max_users is not None:
        users = users[: args.max_users]
        print(f"Limited to {args.max_users} users")

    # Load optional name map (Phase B; harmless to load here for diagnostics).
    name_map: dict[str, str] = {}
    if args.name_map is not None:
        with args.name_map.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                name_map[rec["email"]] = rec["name"]
        print(f"Loaded name map with {len(name_map)} entries")

    # Build train record list once for exposure counting.
    train_records = [r for r in records if r.get("split") == "train"]
    print(f"Train records: {len(train_records):,}")

    all_ground_truth: dict[str, dict[str, dict[str, object]]] = {}
    if args.pii_graph is not None:
        print(f"\nLoading PII graph from {args.pii_graph} ...")
        all_ground_truth = build_ground_truth_from_graph(
            args.pii_graph,
            users,
            args.pii_types,
            user_records,
            min_support=args.min_support,
        )
        print(f"Ground truth: {len(all_ground_truth):,} users")
        # Shared/placeholder filtering still applies to email/phone values.
        filter_shared_and_placeholder_pii(all_ground_truth)
    else:
        # To catch phone numbers that are shared with fax lines, we always
        # extract fax internally even when the user only requested phone. The
        # fax values are dropped after the shared/placeholder filter has run.
        extraction_types = list(dict.fromkeys(list(args.pii_types) + ["fax"]))
        print(f"\nExtracting PII ground truth (requested: {args.pii_types}; "
              f"internal extraction: {extraction_types}) ...")
        for user in users:
            all_ground_truth[user] = extract_user_ground_truth(
                user,
                user_records[user],
                extraction_types,
                min_support=args.min_support,
                fallback_to_single=True,
            )
        filter_shared_and_placeholder_pii(all_ground_truth)

    # Graph-derived "name" values for Enron are email addresses; normalize them
    # to the local part so they act like names (uniform with Mallinckrodt).
    strip_email_domain_from_name_values(all_ground_truth)

    # Restrict reporting to the PII types the user asked for.
    for user in users:
        all_ground_truth[user] = {
            k: v for k, v in all_ground_truth[user].items() if k in args.pii_types
        }

    print("Computing exposure ...")
    exposure_rows: list[dict] = []
    exposure_iter = tqdm(
        ((user, pii_type, meta) for user in users for pii_type, meta in all_ground_truth[user].items()),
        total=sum(len(gt) for gt in all_ground_truth.values()),
        desc="Exposure",
        unit="user/type",
    )
    for user, pii_type, meta in exposure_iter:
        group = "held_out" if is_held_out(split_map[user]) else "held_in"
        value = meta.get("value")
        raw_value = meta.get("raw_value", value)
        exposure = compute_exposure(value, pii_type, train_records)
        exposure_rows.append(
                {
                    "user": user,
                    "group": group,
                    "pii_type": pii_type,
                    "value": value,
                    "raw_value": raw_value,
                    "support": meta.get("support"),
                    "low_confidence": meta.get("low_confidence", False),
                    "excluded_reason": meta.get("excluded_reason"),
                    "exposure_total": exposure["total"],
                    "exposure_header": exposure["header"],
                    "exposure_body": exposure["body"],
                    "exposure_signature": exposure["signature"],
                    "exposure_quoted": exposure["quoted"],
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"pii_extraction_{timestamp}"
    output_dir = args.results_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_out = {
        "data_path": str(args.data),
        "model_path": str(args.model_path) if args.model_path else None,
        "base_model": args.base_model,
        "name_map": str(args.name_map) if args.name_map else None,
        "pii_graph": str(args.pii_graph) if args.pii_graph else None,
        "pii_types": args.pii_types,
        "min_support": args.min_support,
        "prompt_set": args.prompt_set,
        "num_distractors": args.num_distractors,
        "max_users": args.max_users,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "device": str(get_device(args.device)),
        "max_seq_length": args.max_seq_length,
        "max_context_tokens": args.max_context_tokens,
        "mask_token": args.mask_token,
        "context_mode": args.context_mode,
        "exposure_split": args.exposure_split,
        "exposure_from_held_in_only": args.exposure_from_held_in_only,
        "exposure_from_target_only": args.exposure_from_target_only,
        "distractor_mode": args.distractor_mode,
        "email_local_only": args.email_local_only,
        "mask_pii_types_in_prefix": args.mask_pii_types_in_prefix,
        "mask_placeholder": args.mask_placeholder,
        "exposure_only": args.exposure_only,
        "bert_model": args.bert_model,
        "timestamp": timestamp,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2, ensure_ascii=False)

    with (output_dir / "exposure_table.jsonl").open("w", encoding="utf-8") as f:
        for row in exposure_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _summarize_exposure(exposure_rows)
    with (output_dir / "exposure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nExposure table: {len(exposure_rows):,} rows")
    print(json.dumps(summary, indent=2))
    print(f"\nPhase A results saved to {output_dir}")

    if not args.exposure_only:
        if args.model_path is None:
            parser.error("--model-path is required for Phase B.")
        scored_rows, metrics = run_phase_b(
            data_path=args.data,
            model_path=args.model_path,
            base_model_name=args.base_model,
            all_ground_truth=all_ground_truth,
            split_map=split_map,
            records=records,
            pii_types=args.pii_types,
            num_distractors=args.num_distractors,
            max_samples=args.max_samples,
            max_context_tokens=args.max_context_tokens,
            mask_token=args.mask_token,
            context_mode=args.context_mode,
            max_seq_length=args.max_seq_length,
            device_preference=args.device,
            seed=args.seed,
            exposure_split=args.exposure_split,
            exposure_from_held_in_only=args.exposure_from_held_in_only,
            exposure_from_target_only=args.exposure_from_target_only,
            distractor_mode=args.distractor_mode,
            email_local_only=args.email_local_only,
            mask_pii_types_in_prefix=args.mask_pii_types_in_prefix,
            mask_placeholder=args.mask_placeholder,
            bert_model_path=args.bert_model,
        )
        save_phase_b_results(scored_rows, metrics, output_dir, config_out)


if __name__ == "__main__":
    main()
