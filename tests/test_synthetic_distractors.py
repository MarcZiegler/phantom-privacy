"""Tests for synthetic distractor generation."""

from __future__ import annotations

import pytest

from utils.synthetic_distractors import SyntheticDistractorGenerator


@pytest.fixture
def ground_truth() -> dict[str, dict[str, dict[str, object]]]:
    return {
        "u1": {
            "email": {"value": "john.doe@enron.com"},
            "phone": {"value": "713-555-1234"},
            "job_title": {"value": "Vice President"},
        },
        "u2": {
            "email": {"value": "jane.smith@enron.com"},
            "phone": {"value": "713-555-5678"},
            "job_title": {"value": "Manager"},
        },
    }


def test_sample_email(ground_truth: dict[str, dict[str, dict[str, object]]]) -> None:
    gen = SyntheticDistractorGenerator(
        ground_truth,
        pii_types=("email", "phone", "job_title"),
        seed=42,
    )
    true_email = "john.doe@enron.com"
    samples = gen.sample("email", true_email, 5)
    assert len(samples) == 5
    for s in samples:
        assert s.endswith("@enron.com")
        assert s != true_email


def test_sample_phone(ground_truth: dict[str, dict[str, dict[str, object]]]) -> None:
    gen = SyntheticDistractorGenerator(
        ground_truth,
        pii_types=("phone",),
        seed=0,
    )
    samples = gen.sample("phone", "713-555-1234", 10)
    assert len(samples) == 10
    for s in samples:
        assert s != "713-555-1234"
        # Should contain the sampled area code (713).
        assert "713" in s


def test_sample_unsupported_type(
    ground_truth: dict[str, dict[str, dict[str, object]]],
) -> None:
    gen = SyntheticDistractorGenerator(ground_truth, pii_types=("email",), seed=0)
    with pytest.raises(ValueError):
        gen.sample("address", "1600 Pennsylvania Ave", 1)
