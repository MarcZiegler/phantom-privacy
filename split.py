from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def load_unsplit(input_path: Path) -> dict[str, list[dict]]:
    """Load unsplit JSONL rows and group by user."""
    user_records: dict[str, list[dict]] = {}
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user = row["user"]
            user_records.setdefault(user, []).append(row)
    return user_records


def split_kandpal(
    user_records: dict[str, list[dict]],
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    user_split: float = 0.5,
    attacker_frac: float = 0.10,
    seed: int = 42,
) -> list[dict]:
    """
    Paper split strategy.

    1. Split users into held-in and held-out sets.
    2. Reserve attacker_frac from every user first.
    3. From held-in users reserve val/test fractions.
    4. Remainder of held-in -> train; remainder of held-out -> holdout.
    """
    rng = random.Random(seed)
    users = list(user_records.keys())
    rng.shuffle(users)

    n_held_in = max(1, round(len(users) * user_split))
    held_in_users = set(users[:n_held_in])

    rows: list[dict] = []
    for user, recs in user_records.items():
        recs = list(recs)
        rng.shuffle(recs)

        n_attacker = max(0, round(len(recs) * attacker_frac))
        attacker = recs[:n_attacker]
        rest = recs[n_attacker:]

        for r in attacker:
            row = dict(r)
            row["split"] = "attacker"
            rows.append(row)

        if user in held_in_users:
            n_val = max(0, round(len(rest) * val_frac))
            n_test = max(0, round(len(rest) * test_frac))
            val = rest[:n_val]
            test = rest[n_val : n_val + n_test]
            train = rest[n_val + n_test :]

            for r in train:
                row = dict(r)
                row["split"] = "train"
                rows.append(row)
            for r in val:
                row = dict(r)
                row["split"] = "val"
                rows.append(row)
            for r in test:
                row = dict(r)
                row["split"] = "test"
                rows.append(row)
        else:
            for r in rest:
                row = dict(r)
                row["split"] = "holdout"
                rows.append(row)

    return rows


def split_custom(
    user_records: dict[str, list[dict]],
    train_frac: float = 0.8,
    eval_frac: float = 0.2,
    seed: int = 42,
) -> list[dict]:
    """Custom per-user train / eval / test split."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for recs in user_records.values():
        recs = list(recs)
        rng.shuffle(recs)
        n_eval = max(0, round(len(recs) * eval_frac))
        eval_set = recs[:n_eval]
        rest = recs[n_eval:]
        n_train = max(0, round(len(rest) * train_frac))
        train_set = rest[:n_train]
        test_set = rest[n_train:]
        for r in train_set:
            row = dict(r)
            row["split"] = "train"
            rows.append(row)
        for r in eval_set:
            row = dict(r)
            row["split"] = "eval"
            rows.append(row)
        for r in test_set:
            row = dict(r)
            row["split"] = "test"
            rows.append(row)
    return rows


def split_concat_kandpal(
    user_records: dict[str, list[dict]],
    user_split: float = 0.5,
    attacker_frac: float = 0.10,
    seed: int = 42,
) -> list[dict]:
    """
    Concatenated-document split strategy.

    Each user is expected to have exactly one record containing a long
    concatenated text. The last ``attacker_frac`` of the text is reserved as
    attacker knowledge; the remainder is train (held-in) or holdout (held-out).
    """
    rng = random.Random(seed)
    users = list(user_records.keys())
    rng.shuffle(users)

    n_held_in = max(1, round(len(users) * user_split))
    held_in_users = set(users[:n_held_in])

    rows: list[dict] = []
    for user, recs in user_records.items():
        if not recs:
            continue
        # Use the first (and expected only) record for this user.
        record = recs[0]
        text = record.get("text", "")
        total_len = len(text)
        n_attacker = max(1, round(total_len * attacker_frac))
        attacker_text = text[-n_attacker:]
        remainder_text = text[:-n_attacker]

        # Attacker record
        attacker_row = dict(record)
        attacker_row["text"] = attacker_text
        attacker_row["split"] = "attacker"
        rows.append(attacker_row)

        if user in held_in_users:
            # Remainder goes to train (no val/test split to maximize training data)
            train_row = dict(record)
            train_row["text"] = remainder_text
            train_row["split"] = "train"
            rows.append(train_row)
        else:
            holdout_row = dict(record)
            holdout_row["text"] = remainder_text
            holdout_row["split"] = "holdout"
            rows.append(holdout_row)

    return rows


def write_jsonl(rows: list[dict], output_path: Path) -> None:
    """Write rows as JSONL and print split statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    users = len({r["user"] for r in rows})
    splits = Counter(r["split"] for r in rows)
    print(f"\nOutput -> {output_path}")
    print(f"  Users : {users:,}")
    for split, count in sorted(splits.items()):
        print(f"  {split:<10}: {count:,}")
    print(f"  Total : {len(rows):,}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply paper-specific train/test/user splits to an unsplit JSONL."
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Unsplit JSONL from preprocess.py")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output split JSONL")
    p.add_argument(
        "--strategy",
        choices=["kandpal", "custom", "concat-kandpal"],
        default="kandpal",
        help="Split strategy to apply",
    )
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--user-split", type=float, default=0.5)
    p.add_argument("--attacker-frac", type=float, default=0.10)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"ERROR: input file not found at '{args.input}'.")

    print(f"Loading unsplit records from {args.input} ...")
    user_records = load_unsplit(args.input)
    print(f"Loaded {len(user_records):,} users")

    print(f"Applying {args.strategy} split ...")
    if args.strategy == "kandpal":
        rows = split_kandpal(
            user_records,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            user_split=args.user_split,
            attacker_frac=args.attacker_frac,
            seed=args.seed,
        )
    elif args.strategy == "concat-kandpal":
        rows = split_concat_kandpal(
            user_records,
            user_split=args.user_split,
            attacker_frac=args.attacker_frac,
            seed=args.seed,
        )
    else:
        rows = split_custom(
            user_records,
            train_frac=args.train_frac,
            eval_frac=args.eval_frac,
            seed=args.seed,
        )

    write_jsonl(rows, args.output)


if __name__ == "__main__":
    main()
