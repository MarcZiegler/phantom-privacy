"""Unit tests for split strategies."""

from __future__ import annotations

from collections import Counter

import split


def test_split_concat_kandpal_reserves_last_ten_percent_as_attacker() -> None:
    user_records = {
        "alice": [{"user": "alice", "text": "a" * 100}],
        "bob": [{"user": "bob", "text": "b" * 100}],
    }

    rows = split.split_concat_kandpal(user_records, user_split=0.5, attacker_frac=0.10, seed=42)

    by_user: dict[str, dict[str, str]] = {}
    for row in rows:
        by_user.setdefault(row["user"], {})[row["split"]] = row["text"]

    # Each user should have exactly one attacker and one train/holdout record.
    for user in ("alice", "bob"):
        assert "attacker" in by_user[user]
        assert len(by_user[user]["attacker"]) == 10  # 10% of 100
        assert len(by_user[user].get("train", "") + by_user[user].get("holdout", "")) == 90

    # One user held-in, one held-out.
    splits = Counter(r["split"] for r in rows)
    assert splits["attacker"] == 2
    assert splits["train"] == 1
    assert splits["holdout"] == 1
