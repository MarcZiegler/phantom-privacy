"""Unit tests for utility evaluation utilities."""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import torch

from utils.utility_utils import (
    evaluate_texts,
    format_utility_results,
    load_split_texts,
    save_utility_metrics,
)


def test_load_split_texts_filters_split_and_caps_users(tmp_path) -> None:
    data_file = tmp_path / "data.jsonl"
    records = [
        {"user": "alice", "split": "train", "text": "train text"},
        {"user": "bob", "split": "test", "text": "bob test"},
        {"user": "alice", "split": "test", "text": "alice test"},
        {"user": "carol", "split": "holdout", "text": "carol holdout"},
    ]
    data_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    test_texts = load_split_texts(data_file, "test")
    assert sorted(test_texts) == ["alice test", "bob test"]

    # max_users=1 keeps only the first user alphabetically.
    capped = load_split_texts(data_file, "test", max_users=1)
    assert capped == ["alice test"]


def test_load_split_texts_missing_split_returns_empty(tmp_path) -> None:
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        json.dumps({"user": "alice", "split": "train", "text": "train"}) + "\n",
        encoding="utf-8",
    )
    assert load_split_texts(data_file, "test") == []


def test_evaluate_texts_uniform_logits() -> None:
    """With uniform logits, perplexity equals the vocabulary size."""
    vocab_size = 3
    seq_len = 3
    model = MagicMock()
    # Causal LM logits have shape (batch=1, seq_len, vocab).
    model.return_value.logits = torch.zeros(1, seq_len, vocab_size)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = -1
    tokenizer.return_value = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    device = torch.device("cpu")
    result = evaluate_texts(model, tokenizer, ["hello"], device, max_length=seq_len)

    expected_loss = math.log(vocab_size)
    assert result["n_records"] == 1
    assert result["n_tokens"] == seq_len - 1
    assert math.isclose(result["loss"], expected_loss, rel_tol=1e-5)
    assert math.isclose(result["perplexity"], vocab_size, rel_tol=1e-5)


def test_save_and_format_utility_metrics(tmp_path) -> None:
    output_path = tmp_path / "utility_metrics.json"
    metrics = {
        "dataset_name": "test_data",
        "base_model": "test-model",
        "splits": {
            "test": {
                "loss": 1.2345,
                "perplexity": 3.45,
                "n_records": 10,
                "n_tokens": 100,
            }
        },
    }
    save_utility_metrics(output_path, metrics)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["splits"]["test"]["perplexity"] == 3.45

    formatted = format_utility_results(loaded["splits"])
    assert "test:" in formatted
    assert "perplexity=3.45" in formatted
