#!/usr/bin/env python3
"""Add canonical sender emails to the Enron PII graph.

For every user in the preprocessed Enron corpus, the canonical sender address
is already stored as the ``user`` field.  This script materialises those
addresses as ``has_email_address`` edges so that the Enron graph has the same
email coverage as the Mallinckrodt graph.

Outputs:
    preprocessed/enron_email_profiles.json
    preprocessed/pii_graph_enron_with_email.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from augment_pii_graph_with_email import augment_graph_with_emails  # noqa: E402


def build_enron_email_profiles(records_path: Path) -> dict:
    """Return a profiles dict mapping each Enron user to their canonical email."""
    users: set[str] = set()
    with records_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            user = record.get("user")
            if user:
                users.add(user)

    profiles: dict = {"users": {}}
    for user in sorted(users):
        email = user.strip().lower()
        profiles["users"][user] = {"ids": {"email": email}}

    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Add canonical emails to the Enron PII graph.")
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("preprocessed/enron.jsonl"),
        help="Preprocessed Enron JSONL.",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("preprocessed/pii_graph_enron.json"),
        help="Input PII graph JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preprocessed"),
        help="Directory for the profile file and augmented graph.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading records from {args.records} ...")
    profiles = build_enron_email_profiles(args.records)
    print(f"Collected {len(profiles['users'])} user emails.")

    profiles_path = args.output_dir / "enron_email_profiles.json"
    profiles_path.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote profiles to {profiles_path}")

    print(f"Loading graph from {args.graph} ...")
    graph = json.loads(args.graph.read_text(encoding="utf-8"))

    out_graph = augment_graph_with_emails(graph, profiles)

    out_path = args.output_dir / "pii_graph_enron_with_email.json"
    out_path.write_text(json.dumps(out_graph, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote augmented graph to {out_path}: "
        f"added {out_graph['stats']['email_nodes_added']} email nodes and "
        f"{out_graph['stats']['email_edges_added']} email edges."
    )


if __name__ == "__main__":
    main()
