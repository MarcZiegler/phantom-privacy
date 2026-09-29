"""Tests for utils/viz_utils.py."""

from __future__ import annotations

import json
from pathlib import Path

from utils.viz_utils import ResultType, detect_result_type, list_result_dirs


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_list_result_dirs_finds_nested_directories(tmp_path: Path) -> None:
    """Result directories should be discovered recursively."""
    # Direct child with metrics.json
    direct = tmp_path / "attack_20250101_120000"
    direct.mkdir()
    _write_json(direct / "metrics.json", {"auroc": 0.8})

    # Nested cluster-style output
    nested = tmp_path / "attack-preset" / "12345" / "results" / "pii_extraction_20250101_120000"
    nested.mkdir(parents=True)
    _write_json(nested / "metrics.json", {"global": {"auroc": 0.7}})

    # Empty directory should be ignored
    (tmp_path / "empty_dir").mkdir()

    dirs = list_result_dirs(tmp_path)
    assert direct in dirs
    assert nested in dirs
    assert len(dirs) == 2


def test_detect_result_type_pii_extraction(tmp_path: Path) -> None:
    result_dir = tmp_path / "pii_extraction"
    result_dir.mkdir()
    _write_json(result_dir / "metrics.json", {"global": {"auroc": 0.6}})
    _write_jsonl(result_dir / "extraction_scores.jsonl", [])
    _write_jsonl(result_dir / "exposure_table.jsonl", [])
    assert detect_result_type(result_dir) == ResultType.PII_EXTRACTION


def test_detect_result_type_pii_autocomplete(tmp_path: Path) -> None:
    result_dir = tmp_path / "pii_autocomplete"
    result_dir.mkdir()
    _write_json(
        result_dir / "metrics.json",
        {"global": {}, "by_prompt_set": {"target": {}}},
    )
    _write_jsonl(result_dir / "generations.jsonl", [])
    assert detect_result_type(result_dir) == ResultType.PII_AUTOCOMPLETE


def test_detect_result_type_unknown(tmp_path: Path) -> None:
    result_dir = tmp_path / "unknown"
    result_dir.mkdir()
    _write_json(result_dir / "metrics.json", {"other": 1})
    assert detect_result_type(result_dir) == ResultType.UNKNOWN


def test_detect_result_type_missing_directory(tmp_path: Path) -> None:
    assert detect_result_type(tmp_path / "does_not_exist") == ResultType.UNKNOWN
