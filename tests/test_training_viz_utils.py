"""Tests for utils/training_viz_utils.py."""

from __future__ import annotations

import pytest

from utils.training_logger import TrainingLogger
from utils.training_viz_utils import (
    format_training_summary,
    plot_training_curves,
)


def _make_history(tmp_path, records):
    logger = TrainingLogger(tmp_path)
    for step, metrics in records:
        logger.log(step, metrics)
    return logger.log_path


def test_plot_training_curves_generates_expected_pngs(tmp_path):
    history = _make_history(
        tmp_path,
        [
            (0, {"train_loss": 2.0, "learning_rate": 5e-5, "grad_norm": 1.0}),
            (1, {"train_loss": 1.5, "learning_rate": 4e-5, "grad_norm": 0.8}),
            (2, {"train_loss": 1.0, "learning_rate": 3e-5, "grad_norm": 0.6}),
        ],
    )
    output_dir = tmp_path / "plots"
    generated = plot_training_curves(history, output_dir)

    assert generated
    assert (output_dir / "train_loss.png").exists()
    assert (output_dir / "learning_rate.png").exists()
    assert (output_dir / "grad_norm.png").exists()


def test_plot_training_curves_skips_empty_history(tmp_path):
    history = tmp_path / "training_history.jsonl"
    history.write_text("", encoding="utf-8")
    assert plot_training_curves(history, tmp_path / "plots") == []


def test_plot_training_curves_eval_metrics(tmp_path):
    history = _make_history(
        tmp_path,
        [
            (0, {"train_loss": 2.0}),
            (1, {"train_loss": 1.5, "eval_loss": 1.2}),
            (2, {"train_loss": 1.0, "eval_loss": 0.9}),
        ],
    )
    output_dir = tmp_path / "plots"
    plot_training_curves(history, output_dir)
    assert (output_dir / "eval_loss.png").exists()


def test_format_training_summary(tmp_path):
    history = _make_history(
        tmp_path,
        [
            (0, {"train_loss": 2.0}),
            (1, {"train_loss": 1.0}),
        ],
    )
    summary = format_training_summary(history)
    assert "2 logged points" in summary
    assert "train_loss" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
