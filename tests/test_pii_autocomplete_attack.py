"""Unit tests for pii_autocomplete_attack.py helpers."""

from __future__ import annotations

import pii_autocomplete_attack as ac


class _CharTokenizer:
    """Minimal tokenizer: one token per character."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: FBT001
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:  # noqa: FBT001
        return "".join(chr(i) for i in token_ids)


def test_normalize_text() -> None:
    assert ac.normalize_text("Hello, World!") == "hello world"
    assert ac.normalize_text("  Multiple   spaces  ") == "multiple spaces"


def test_canonical_phone() -> None:
    assert ac.canonical_phone("(713) 853-1234") == "7138531234"
    assert ac.canonical_phone("+1 713-853-1234") == "7138531234"
    assert ac.canonical_phone("17138531234") == "7138531234"


def test_match_value_phone() -> None:
    assert ac.match_value("713-853-1234", "713-853-1234", "phone")["exact"] is True
    assert ac.match_value("call 713-853-1234 today", "713-853-1234", "phone")["contains"] is True
    assert ac.match_value("713-853-9999", "713-853-1234", "phone")["contains"] is False


def test_match_value_email() -> None:
    assert ac.match_value("alice@example.com", "alice@example.com", "email")["exact"] is True
    assert ac.match_value("Reach alice@example.com please", "alice@example.com", "email")["contains"]
    assert not ac.match_value("bob@example.com", "alice@example.com", "email")["contains"]


def test_match_value_empty_true_value() -> None:
    assert not ac.match_value("anything", "", "email")["contains"]
    assert not ac.match_value("anything", "", "email")["exact"]


def test_compute_metrics() -> None:
    rows = [
        {
            "user": "u1", "group": "held_out", "pii_type": "phone", "prompt_set": "target",
            "ft_exact": True, "ft_contains": True, "base_exact": False, "base_contains": False,
        },
        {
            "user": "u1", "group": "held_out", "pii_type": "phone", "prompt_set": "target",
            "ft_exact": False, "ft_contains": True, "base_exact": False, "base_contains": True,
        },
    ]
    metrics = ac.compute_metrics(rows)
    assert metrics["global"]["ft_contains"]["rate"] == 1.0
    assert metrics["global"]["ft_exact"]["rate"] == 0.5
    assert metrics["global"]["base_contains"]["rate"] == 0.5
    assert metrics["per_user_summary"]["num_users"] == 1


def test_compute_metrics_per_user() -> None:
    rows = [
        {
            "user": "u1", "group": "held_out", "pii_type": "phone", "prompt_set": "target",
            "ft_exact": True, "ft_contains": True,
            "base_exact": False, "base_contains": False,
        },
        {
            "user": "u2", "group": "held_in", "pii_type": "phone", "prompt_set": "cross",
            "ft_exact": False, "ft_contains": False,
            "base_exact": False, "base_contains": False,
        },
    ]
    metrics = ac.compute_metrics(rows)
    assert metrics["per_user_summary"]["num_users"] == 2
    assert metrics["per_user_summary"]["mean_ft_contains_rate"] == 0.5


def test_build_prompts_includes_all_sets() -> None:
    tokenizer = _CharTokenizer()
    user_records = [
        {"split": "attacker", "text": "u1's number is 713-853-1234"},
    ]
    held_in_attacker_records = [
        {"user": "other", "text": "Call 713-853-1234"},
    ]
    prompts = ac.build_prompts(
        user="u1",
        pii_type="phone",
        value="713-853-1234",
        raw_value="713-853-1234",
        user_records=user_records,
        held_in_attacker_records=held_in_attacker_records,
        prompt_sets=("target", "cross", "synthetic"),
        max_samples=10,
        max_context_tokens=100,
        tokenizer=tokenizer,
    )
    assert "target" in prompts
    assert "cross" in prompts
    assert "synthetic" in prompts
    assert prompts["target"]
    assert prompts["cross"]
    assert prompts["synthetic"]


def test_build_prompts_masks_values() -> None:
    tokenizer = _CharTokenizer()
    # Prefix contains an email that should be masked, but the target phone is
    # the generated value and therefore does not appear in the prompt.
    user_records = [
        {"split": "attacker", "text": "Email u1@example.com or call 713-853-1234"},
    ]
    held_in_attacker_records = []
    user_ground_truth = {
        "phone": {"value": "713-853-1234", "raw_value": "713-853-1234"},
        "email": {"value": "u1@example.com", "raw_value": "u1@example.com"},
    }
    prompts = ac.build_prompts(
        user="u1",
        pii_type="phone",
        value="713-853-1234",
        raw_value="713-853-1234",
        user_records=user_records,
        held_in_attacker_records=held_in_attacker_records,
        prompt_sets=("target",),
        max_samples=10,
        max_context_tokens=100,
        tokenizer=tokenizer,
        mask_pii_types_in_prefix=("email",),
        mask_placeholder="<MASK>",
        user_ground_truth=user_ground_truth,
    )
    assert all("u1@example.com" not in p for p in prompts["target"])
    assert any("<MASK>" in p for p in prompts["target"])
