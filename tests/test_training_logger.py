"""Tests for utils/training_logger.py."""

from __future__ import annotations

import json

import pytest

from utils.training_logger import TrainingLogger, load_training_history


def test_training_logger_creates_directory_and_file(tmp_path):
    logger = TrainingLogger(tmp_path / "sub")
    assert logger.log_path.exists()
    assert logger.log_path.name == "training_history.jsonl"


def test_training_logger_appends_records(tmp_path):
    logger = TrainingLogger(tmp_path)
    logger.log(0, {"train_loss": 1.0, "learning_rate": 5e-5})
    logger.log(1, {"train_loss": 0.5, "learning_rate": 5e-5})

    history = load_training_history(logger.log_path)
    assert len(history) == 2
    assert history[0]["step"] == 0
    assert history[0]["train_loss"] == 1.0
    assert history[1]["step"] == 1
    assert history[1]["train_loss"] == 0.5
    assert "timestamp" in history[0]


def test_training_logger_converts_non_scalars(tmp_path):
    logger = TrainingLogger(tmp_path)
    logger.log(0, {"train_loss": 1.0})

    raw = logger.log_path.read_text(encoding="utf-8").strip()
    record = json.loads(raw)
    assert record["step"] == 0
    assert record["train_loss"] == 1.0


def test_load_training_history_missing_file(tmp_path):
    assert load_training_history(tmp_path / "missing.jsonl") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
