"""Plotting utilities for training-history JSONL files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from utils.training_logger import load_training_history
from utils.viz_config import VizConfig, save_figure

os.environ.setdefault("MPLBACKEND", "Agg")


_METRIC_PLOTS: tuple[tuple[str, str, str], ...] = (
    ("train_loss", "Training Loss", "Loss"),
    ("eval_loss", "Evaluation Loss", "Loss"),
    ("eval_perplexity", "Evaluation Perplexity", "Perplexity"),
    ("learning_rate", "Learning Rate Schedule", "Learning Rate"),
    ("grad_norm", "Gradient Norm", "Gradient Norm"),
    ("privacy_epsilon", "Cumulative Privacy Epsilon", "ε"),
    ("privacy_epsilon_record", "Record-Level Cumulative Epsilon", "ε"),
    ("privacy_epsilon_user", "User-Level Cumulative Epsilon", "ε"),
)


def _plot_metric(
    records: list[dict[str, Any]],
    metric_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
    cfg: VizConfig | None = None,
) -> bool:
    """Plot a single metric versus step and save to a figure file.

    Args:
        records: Training history records.
        metric_key: Metric to plot.
        output_path: Destination figure path.
        title: Plot title.
        ylabel: Y-axis label.
        cfg: Shared visualization config.

    Returns:
        True if the metric existed and a plot was written, False otherwise.
    """
    cfg = cfg or VizConfig()
    import matplotlib.pyplot as plt

    points = [(r["step"], r[metric_key]) for r in records if metric_key in r]
    if not points:
        return False

    steps, values = zip(*points, strict=False)

    fig, ax = plt.subplots(figsize=cfg.training_curve)
    ax.plot(steps, values, linewidth=1.5, color=cfg.blue)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    plt.close(fig)
    return True


def plot_training_curves(
    history_path: Path,
    output_dir: Path,
    cfg: VizConfig | None = None,
) -> list[Path]:
    """Generate figure files for all metrics found in a training history JSONL.

    Args:
        history_path: Path to `training_history.jsonl`.
        output_dir: Directory to write figure files.
        cfg: Shared visualization config.

    Returns:
        List of paths to the generated figure files.
    """
    cfg = cfg or VizConfig()
    records = load_training_history(history_path)
    if not records:
        return []

    generated: list[Path] = []
    for metric_key, title, ylabel in _METRIC_PLOTS:
        output_path = output_dir / f"{metric_key}.{cfg.format}"
        if _plot_metric(records, metric_key, output_path, title, ylabel, cfg):
            generated.append(output_path)

    return generated


def format_training_summary(history_path: Path) -> str:
    """Return a short text summary of the logged metrics."""
    records = load_training_history(history_path)
    if not records:
        return "No training history found."

    lines = [f"Training history: {len(records)} logged points"]
    keys = sorted({k for r in records for k in r if k not in ("step", "timestamp")})
    for key in keys:
        values = [r[key] for r in records if key in r]
        lines.append(f"  {key}: min={min(values):.4f}, max={max(values):.4f}")
    return "\n".join(lines)
