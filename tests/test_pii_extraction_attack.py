"""Smoke tests for pii_extraction_attack.py."""

from __future__ import annotations

import json
import sys

import pytest

import pii_extraction_attack


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_exposure_only_smoke(tmp_path, monkeypatch):
    """Phase A (exposure-only) runs end-to-end without a model."""
    data_path = tmp_path / "data.jsonl"
    results_dir = tmp_path / "results"

    records = [
        {
            "user": "alice@enron.com",
            "text": "Call Alice at 713-853-1111.",
            "body": "Call Alice at 713-853-1111.",
            "split": "train",
        },
        {
            "user": "alice@enron.com",
            "text": "Meeting at 10.",
            "body": "Meeting at 10.",
            "split": "attacker",
        },
        {
            "user": "bob@enron.com",
            "text": "Bob's number is 713-853-2222.",
            "body": "Bob's number is 713-853-2222.",
            "split": "holdout",
        },
        {
            "user": "bob@enron.com",
            "text": "Reach me at 713-853-2222.",
            "body": "Reach me at 713-853-2222.",
            "split": "attacker",
        },
    ]
    _write_jsonl(data_path, records)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii_extraction_attack.py",
            "--data", str(data_path),
            "--results-dir", str(results_dir),
            "--pii-types", "email", "phone",
            "--exposure-only",
            "--max-users", "2",
            "--seed", "42",
        ],
    )

    pii_extraction_attack.main()

    output_dirs = list(results_dir.iterdir())
    assert len(output_dirs) == 1
    out = output_dirs[0]
    assert (out / "config.json").exists()
    assert (out / "exposure_table.jsonl").exists()
    assert (out / "exposure_summary.json").exists()

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["exposure_only"] is True
    assert config["pii_types"] == ["email", "phone"]

    rows = [
        json.loads(line)
        for line in (out / "exposure_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4  # email + phone for each of 2 users
    groups = {r["group"] for r in rows}
    assert groups == {"held_in", "held_out"}

    summary = json.loads((out / "exposure_summary.json").read_text(encoding="utf-8"))
    assert summary["num_users"] == 2
    assert summary["num_records"] == 4


def test_exposure_only_with_pii_graph(tmp_path, monkeypatch):
    """Phase A can load ground truth from a PII graph."""
    data_path = tmp_path / "data.jsonl"
    graph_path = tmp_path / "graph.json"
    results_dir = tmp_path / "results"

    records = [
        {"user": "u1", "text": "u1 text", "body": "u1 text", "split": "train"},
        {"user": "u2", "text": "u2 713-853-1234", "body": "u2 713-853-1234", "split": "holdout"},
    ]
    _write_jsonl(data_path, records)

    graph = {
        "nodes": [
            {"id": "USER:u1", "type": "USER", "label": "u1"},
            {"id": "USER:u2", "type": "USER", "label": "u2"},
            {"id": "PHONE_NUMBER:713-853-1234", "type": "PHONE_NUMBER", "label": "713-853-1234"},
        ],
        "edges": [
            {
                "source": "USER:u2",
                "target": "PHONE_NUMBER:713-853-1234",
                "relation": "has_phone_number",
                "score": 1.0,
            }
        ],
    }
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii_extraction_attack.py",
            "--data", str(data_path),
            "--results-dir", str(results_dir),
            "--pii-graph", str(graph_path),
            "--pii-types", "phone",
            "--exposure-only",
            "--max-users", "2",
        ],
    )

    pii_extraction_attack.main()

    output_dirs = list(results_dir.iterdir())
    assert len(output_dirs) == 1
    rows = [
        json.loads(line)
        for line in (output_dirs[0] / "exposure_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["user"] == "u2"
    assert rows[0]["pii_type"] == "phone"


def test_invalid_pii_type_exits(tmp_path, monkeypatch):
    """An unsupported PII type causes a clean sys.exit."""
    data_path = tmp_path / "data.jsonl"
    _write_jsonl(data_path, [{"user": "u1", "text": "x", "body": "x", "split": "train"}])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pii_extraction_attack.py",
            "--data", str(data_path),
            "--pii-types", "ssn",
            "--exposure-only",
        ],
    )

    with pytest.raises(SystemExit):
        pii_extraction_attack.main()


def test_summarize_exposure_counts():
    """The exposure summary aggregates rows by group and PII type."""
    rows = [
        {
            "user": "a", "group": "held_in", "pii_type": "phone",
            "value": "123", "exposure_total": 5,
        },
        {
            "user": "b", "group": "held_out", "pii_type": "phone",
            "value": "456", "exposure_total": 0,
        },
    ]
    summary = pii_extraction_attack._summarize_exposure(rows)
    assert summary["num_users"] == 2
    assert summary["num_held_in"] == 1
    assert summary["num_held_out"] == 1
    assert summary["num_records"] == 2
    assert "phone" in summary["per_pii_type"]
