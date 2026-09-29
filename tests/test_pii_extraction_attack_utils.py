"""Unit tests for Phase B extraction attack utilities."""

from __future__ import annotations

import numpy as np
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.pii_extraction_attack_utils import (
    _mask_values_in_prefix,
    compute_extraction_metrics,
    sample_distractors,
    score_candidates,
    strip_email_domain_from_name_values,
)


def test_strip_email_domain_from_name_values() -> None:
    gt = {
        "u1@enron.com": {
            "name": {"value": "u1@enron.com", "raw_value": "U1@enron.com"},
            "phone": {"value": "713-853-0001"},
        },
        "plain_name": {"name": {"value": "Alice Smith"}},
        "no_name": {"phone": {"value": "713-853-0002"}},
    }
    strip_email_domain_from_name_values(gt)
    assert gt["u1@enron.com"]["name"]["value"] == "u1"
    assert gt["u1@enron.com"]["name"]["raw_value"] == "U1"
    assert gt["plain_name"]["name"]["value"] == "Alice Smith"
    assert "no_name" not in gt["no_name"].get("name", {})


def test_mask_values_in_prefix() -> None:
    prefix = "Call Tom at 555-1234 or email tom.smith@example.com. Tom said..."
    masked = _mask_values_in_prefix(
        prefix, ["Tom", "tom.smith@example.com", "555-1234"], "<MASK>"
    )
    assert "Tom" not in masked
    assert "tom.smith@example.com" not in masked
    assert "555-1234" not in masked
    assert "<MASK>" in masked
    # Longer value should be masked before shorter sub-value.
    assert "<MASK>@example.com" not in masked


def test_mask_values_in_prefix_ignores_empty_values() -> None:
    prefix = "Call Tom at 555-1234."
    masked = _mask_values_in_prefix(prefix, ["Tom", "", "   ", None, "555-1234"], "<MASK>")
    assert "Tom" not in masked
    assert "555-1234" not in masked
    # Empty values must not corrupt the text by inserting placeholders between chars.
    assert "<MASK>C<MASK>" not in masked
    assert masked.count("<MASK>") == 2


def test_sample_distractors_email() -> None:
    gt = {
        "a@enron.com": {"email": {"value": "a@enron.com"}},
        "b@enron.com": {"email": {"value": "b@enron.com"}},
        "c@enron.com": {"email": {"value": "c@enron.com"}},
        "d@other.com": {"email": {"value": "d@other.com"}},
    }
    rng = np.random.default_rng(0)
    pool = sample_distractors(gt, "email", "a@enron.com", 2, rng)
    assert "a@enron.com" not in pool
    assert len(pool) == 2
    assert all(v.endswith("@enron.com") for v in pool)


def test_sample_distractors_phone() -> None:
    gt = {
        "a@enron.com": {"phone": {"value": "713-853-0001"}},
        "b@enron.com": {"phone": {"value": "713-853-0002"}},
        "c@enron.com": {"phone": {"value": "713-853-0003"}},
    }
    rng = np.random.default_rng(0)
    pool = sample_distractors(gt, "phone", "713-853-0001", 2, rng)
    assert "713-853-0001" not in pool
    assert len(pool) == 2


def test_compute_extraction_metrics_global_auroc() -> None:
    rows = [
        {"label": 1, "score": 0.9, "group": "held_in", "pii_type": "email", "user": "a"},
        {"label": 0, "score": 0.1, "group": "held_in", "pii_type": "email", "user": "a"},
        {"label": 1, "score": 0.8, "group": "held_out", "pii_type": "email", "user": "b"},
        {"label": 0, "score": 0.2, "group": "held_out", "pii_type": "email", "user": "b"},
        {"label": 1, "score": 0.85, "group": "held_in", "pii_type": "phone", "user": "a"},
        {"label": 0, "score": 0.15, "group": "held_out", "pii_type": "phone", "user": "b"},
    ]
    metrics = compute_extraction_metrics(rows)
    assert metrics["global"]["auroc"] == 1.0
    assert metrics["global"]["num_true"] == 3
    assert metrics["global"]["num_false"] == 3
    assert metrics["by_group"]["held_in"]["auroc"] == 1.0
    assert metrics["by_group"]["held_out"]["auroc"] == 1.0
    assert metrics["by_pii_type"]["email"]["auroc"] == 1.0
    assert metrics["by_pii_type"]["phone"]["auroc"] == 1.0
    assert metrics["per_user_auroc"]["num_users"] == 2


def test_compute_extraction_metrics_undefined_auroc() -> None:
    # Only true labels -> undefined AUROC.
    rows = [{"label": 1, "score": 0.5, "user": "a"}]
    metrics = compute_extraction_metrics(rows)
    assert metrics["global"]["auroc"] is None
    assert metrics["per_user_auroc"] is None


def test_compute_extraction_metrics_ignores_non_finite_scores() -> None:
    rows = [
        {"label": 1, "score": 0.9, "group": "held_in", "pii_type": "email", "user": "a"},
        {"label": 0, "score": float("nan"), "group": "held_in", "pii_type": "email", "user": "a"},
        {"label": 0, "score": float("inf"), "group": "held_in", "pii_type": "email", "user": "a"},
        {"label": 0, "score": 0.1, "group": "held_in", "pii_type": "email", "user": "a"},
    ]
    metrics = compute_extraction_metrics(rows)
    assert metrics["global"]["auroc"] == 1.0
    assert metrics["global"]["num_true"] == 1
    assert metrics["global"]["num_false"] == 1


@pytest.mark.slow
@pytest.mark.skip("Requires a small causal LM; run manually for smoke testing.")
def test_score_candidates_smoke() -> None:
    """Real-model smoke test: a candidate that matches the prompt should score
    lower than a random distractor.
    """
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    device = model.device
    prefix = "Hello my name is"
    candidates = [" Alice", " xyz123abc"]
    losses, _ = zip(
        *score_candidates(model, tokenizer, prefix, candidates, device, max_seq_length=128),
        strict=False,
    )
    assert losses[0] < losses[1]
