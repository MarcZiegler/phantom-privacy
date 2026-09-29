from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path


def load_jsonl(path: Path) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def group_by_user(records: list[dict], split_filter: str | None = None) -> dict[str, list[dict]]:
    """Group records by user, optionally filtering by split."""
    user_records: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if split_filter is None or rec.get("split") == split_filter:
            user = rec.get("user", "")
            if user:
                user_records[user].append(rec)
    return dict(user_records)
