from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from utils.viz_config import VizConfig
from utils.viz_config import save_figure as _save_figure


class ResultType(Enum):
    """Category of a visualization result directory."""

    UNKNOWN = "unknown"
    PII_EXTRACTION = "pii_extraction"
    PII_AUTOCOMPLETE = "pii_autocomplete"


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_figure(fig: Figure, path: Path, cfg: VizConfig | None = None) -> None:
    """Save a matplotlib figure, creating parent directories if needed.

    Uses the shared ``VizConfig`` for DPI, bbox, and padding defaults.
    """
    _save_figure(fig, path, cfg)


def list_result_dirs(results_root: Path, max_depth: int = 4) -> list[Path]:
    """Return a sorted list of result directories under ``results_root``.

    A directory is considered a result directory if it contains a
    ``metrics.json`` file. The search is recursive up to ``max_depth`` levels
    so that nested cluster outputs (e.g.
    ``results/<spec>/<jobid>/results/pii_extraction_...``) are also
    discovered.
    """
    if not results_root.exists():
        return []

    result_dirs: list[Path] = []
    _collect_result_dirs(results_root, current_depth=0, max_depth=max_depth, out=result_dirs)
    return sorted(result_dirs, reverse=True)


def _collect_result_dirs(
    root: Path,
    current_depth: int,
    max_depth: int,
    out: list[Path],
) -> None:
    """Recursively collect directories that contain ``metrics.json``."""
    if current_depth > max_depth:
        return

    for path in root.iterdir():
        if not path.is_dir():
            continue
        if (path / "metrics.json").exists():
            out.append(path)
        else:
            _collect_result_dirs(path, current_depth + 1, max_depth, out)


def detect_result_type(result_dir: Path) -> ResultType:
    """Detect the type of result stored in ``result_dir``.

    Detection is based on the presence of characteristic output files:

    - ``extraction_scores.jsonl`` + ``exposure_table.jsonl`` → PII extraction
    - ``generations.jsonl`` + ``by_prompt_set`` metrics → autocomplete
    """
    if not result_dir.is_dir():
        return ResultType.UNKNOWN

    metrics_path = result_dir / "metrics.json"
    if not metrics_path.exists():
        return ResultType.UNKNOWN

    if (result_dir / "extraction_scores.jsonl").exists() and (
        result_dir / "exposure_table.jsonl"
    ).exists():
        return ResultType.PII_EXTRACTION

    if (result_dir / "generations.jsonl").exists():
        return ResultType.PII_AUTOCOMPLETE

    return ResultType.UNKNOWN
