"""Simple JSONL training-history logger.

Writes one line per logging event so that training curves can be plotted
after a run without depending on TensorBoard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TrainingLogger:
    """Append-only JSONL logger for training metrics."""

    output_dir: Path
    filename: str = field(default="training_history.jsonl")

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / self.filename
        # Create an empty file so the log path exists even before the first log.
        self.log_path.touch(exist_ok=True)

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        """Append a metrics record for the given step.

        Args:
            step: Training step number.
            metrics: Dictionary of scalar metrics to log.
        """
        record: dict[str, Any] = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        }
        # Only serialize JSON-safe scalars.
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool, str)):
                record[key] = value
            else:
                record[key] = float(value)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_training_history(log_path: Path) -> list[dict[str, Any]]:
    """Load a JSONL training history file.

    Args:
        log_path: Path to the JSONL file.

    Returns:
        List of metric records ordered as written.
    """
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
