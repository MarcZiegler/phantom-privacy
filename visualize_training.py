#!/usr/bin/env python
"""Interactively regenerate training-curve plots from logged JSONL history."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.training_logger import load_training_history
from utils.training_viz_utils import format_training_summary, plot_training_curves
from utils.viz_config import VizConfig, apply_viz_style


def _find_model_dirs(root: Path) -> list[Path]:
    """Return final model directories that contain a training history file."""
    candidates: list[Path] = []
    for pattern in ("models/finetuned*/final", "models/finetuned*/latest"):
        candidates.extend(root.glob(pattern))

    # De-duplicate and keep only dirs with history.
    seen = set()
    result: list[Path] = []
    for candidate in candidates:
        key = candidate.resolve()
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "training_history.jsonl").exists():
            result.append(candidate)
    return sorted(result)


def _select_model_dir(dirs: list[Path]) -> Path:
    """Ask the user to pick a model directory."""
    if not dirs:
        print("No model directories with training_history.jsonl found.", file=sys.stderr)
        sys.exit(1)

    print("\nAvailable training runs:\n")
    for i, d in enumerate(dirs, 1):
        with (d / "training_history.jsonl").open("r", encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
        print(f"  {i:2}. {d}  ({n} log points)")

    while True:
        choice = input("\nSelect a run (number) or 'q' to quit: ").strip()
        if choice.lower() == "q":
            sys.exit(0)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(dirs):
                return dirs[idx]
        except ValueError:
            pass
        print("Invalid choice. Try again.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate training-curve plots from training_history.jsonl."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to a model directory containing training_history.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write plots (default: MODEL_DIR/training_plots).",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output figure format (default: png).",
    )
    args = parser.parse_args()

    cfg = apply_viz_style(VizConfig(format=args.format))

    if args.model_dir is not None:
        model_dir = args.model_dir
    else:
        if not sys.stdin.isatty():
            print(
                "ERROR: No --model-dir given and stdin is not interactive.",
                file=sys.stderr,
            )
            sys.exit(1)
        model_dir = _select_model_dir(_find_model_dirs(Path.cwd()))

    history_path = model_dir / "training_history.jsonl"
    if not history_path.exists():
        print(f"No training history found at {history_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or (model_dir / "training_plots")
    records = load_training_history(history_path)
    if not records:
        print(f"Training history at {history_path} is empty.", file=sys.stderr)
        sys.exit(1)

    generated = plot_training_curves(history_path, output_dir, cfg=cfg)
    print(f"Generated {len(generated)} plots in {output_dir}")
    print(format_training_summary(history_path))


if __name__ == "__main__":
    main()
