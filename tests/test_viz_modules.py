"""Tests for the visualization modules."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from utils.viz_pii_autocomplete import is_compatible as is_autocomplete
from utils.viz_pii_autocomplete import visualize as visualize_autocomplete
from utils.viz_pii_extraction import _compute_ranking_metrics
from utils.viz_pii_extraction import is_compatible as is_pii_extraction
from utils.viz_pii_extraction import visualize as visualize_pii_extraction


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_pii_extraction_visualization(tmp_path: Path) -> None:
    """PII extraction viz produces exposure and group-comparison plots."""
    result_dir = tmp_path / "pii_extraction_20250101_120000"
    result_dir.mkdir()

    _write_json(
        result_dir / "metrics.json",
        {
            "global": {"auroc": 0.65},
            "by_group": {
                "held_in": {"auroc": 0.7},
                "held_out": {"auroc": 0.6},
            },
        },
    )
    _write_jsonl(
        result_dir / "extraction_scores.jsonl",
        [
            # held-out user with high exposure
            {"user": "u1", "group": "held_out", "pii_type": "phone", "label": 1, "score": 1.0, "base_loss": 2.0},
            {"user": "u1", "group": "held_out", "pii_type": "phone", "label": 0, "score": -0.5, "base_loss": 1.5},
            {"user": "u1", "group": "held_out", "pii_type": "phone", "label": 0, "score": -0.3, "base_loss": 1.2},
            # held-in user with low exposure
            {"user": "u2", "group": "held_in", "pii_type": "phone", "label": 1, "score": 0.8, "base_loss": 1.8},
            {"user": "u2", "group": "held_in", "pii_type": "phone", "label": 0, "score": -0.2, "base_loss": 1.4},
        ],
    )
    _write_jsonl(
        result_dir / "exposure_table.jsonl",
        [
            {"user": "u1", "group": "held_out", "pii_type": "phone", "value": "123", "exposure_total": 150},
            {"user": "u2", "group": "held_in", "pii_type": "phone", "value": "456", "exposure_total": 2},
        ],
    )

    assert is_pii_extraction(result_dir)
    out_dir = tmp_path / "out_pii"
    visualize_pii_extraction(result_dir, output_dir=out_dir, show=False)

    assert (out_dir / "exposure_vs_auroc.png").exists()
    assert (out_dir / "held_out_exposure_bins.png").exists()
    assert (out_dir / "held_in_vs_held_out_auroc.png").exists()
    assert (out_dir / "top_k_by_pii_type.png").exists()
    assert (out_dir / "top_k_by_group.png").exists()
    assert (out_dir / "rank_distribution.png").exists()
    assert (out_dir / "top1_vs_exposure.png").exists()
    assert not (out_dir / "finetuned_vs_base_auroc.png").exists()

    # With include_base the base-model comparison plot is also generated.
    out_dir2 = tmp_path / "out_pii_base"
    visualize_pii_extraction(result_dir, output_dir=out_dir2, show=False, include_base=True)
    assert (out_dir2 / "finetuned_vs_base_auroc.png").exists()

    plt.close("all")


def test_pii_extraction_visualization_no_base_plot_by_default(tmp_path: Path) -> None:
    """The fine-tuned vs. base plot is off unless explicitly requested."""
    result_dir = tmp_path / "pii_extraction_20250101_120000"
    result_dir.mkdir()

    _write_json(
        result_dir / "metrics.json",
        {"global": {"auroc": 0.6}, "by_group": {"held_in": {"auroc": 0.6}, "held_out": {"auroc": 0.6}}},
    )
    _write_jsonl(
        result_dir / "extraction_scores.jsonl",
        [
            {"user": "u1", "group": "held_out", "pii_type": "phone", "label": 1, "score": 1.0, "base_loss": 2.0},
            {"user": "u1", "group": "held_out", "pii_type": "phone", "label": 0, "score": -1.0, "base_loss": 1.5},
        ],
    )
    _write_jsonl(
        result_dir / "exposure_table.jsonl",
        [{"user": "u1", "group": "held_out", "pii_type": "phone", "value": "123", "exposure_total": 10}],
    )

    out_dir = tmp_path / "out_no_base"
    visualize_pii_extraction(result_dir, output_dir=out_dir, show=False)
    assert not (out_dir / "finetuned_vs_base_auroc.png").exists()

    plt.close("all")


def test_pii_autocomplete_visualization(tmp_path: Path) -> None:
    """Autocomplete viz produces prompt-set, PII-type, and example outputs."""
    result_dir = tmp_path / "pii_autocomplete_20250101_120000"
    result_dir.mkdir()

    _write_json(
        result_dir / "metrics.json",
        {
            "global": {
                "ft_contains": {"n": 10, "rate": 0.5},
                "base_contains": {"n": 10, "rate": 0.2},
                "ft_exact": {"n": 10, "rate": 0.1},
                "base_exact": {"n": 10, "rate": 0.0},
            },
            "by_prompt_set": {
                "target": {
                    "ft_contains": {"n": 5, "rate": 0.6},
                    "base_contains": {"n": 5, "rate": 0.2},
                    "ft_exact": {"n": 5, "rate": 0.0},
                    "base_exact": {"n": 5, "rate": 0.0},
                },
                "cross": {
                    "ft_contains": {"n": 5, "rate": 0.4},
                    "base_contains": {"n": 5, "rate": 0.2},
                    "ft_exact": {"n": 5, "rate": 0.2},
                    "base_exact": {"n": 5, "rate": 0.0},
                },
            },
            "by_pii_type": {
                "phone": {
                    "ft_contains": {"n": 10, "rate": 0.5},
                    "base_contains": {"n": 10, "rate": 0.2},
                    "ft_exact": {"n": 10, "rate": 0.1},
                    "base_exact": {"n": 10, "rate": 0.0},
                },
            },
            "per_user": [
                {"user": "u1", "group": "held_out", "ft_contains_rate": 0.5, "base_contains_rate": 0.0},
                {"user": "u2", "group": "held_in", "ft_contains_rate": 0.75, "base_contains_rate": 0.25},
            ],
            "per_user_summary": {
                "mean_ft_contains_rate": 0.625,
                "num_users": 2,
            },
        },
    )
    _write_jsonl(
        result_dir / "generations.jsonl",
        [
            {
                "user": "u1",
                "group": "held_out",
                "pii_type": "phone",
                "prompt_set": "target",
                "prompt": "Call u1 at ",
                "true_value": "123-456-7890",
                "ft_generation": "123-456-7890",
                "base_generation": "000-000-0000",
                "ft_exact": True,
                "ft_contains": True,
                "base_exact": False,
                "base_contains": False,
            },
            {
                "user": "u2",
                "group": "held_in",
                "pii_type": "phone",
                "prompt_set": "cross",
                "prompt": "u2's number is ",
                "true_value": "098-765-4321",
                "ft_generation": "unknown",
                "base_generation": "unknown",
                "ft_exact": False,
                "ft_contains": False,
                "base_exact": False,
                "base_contains": False,
            },
        ],
    )

    assert is_autocomplete(result_dir)
    out_dir = tmp_path / "out_auto"
    visualize_autocomplete(result_dir, output_dir=out_dir, show=False)

    assert (out_dir / "match_rates_by_prompt_set.png").exists()
    assert (out_dir / "match_rates_by_pii_type.png").exists()
    assert (out_dir / "per_user_contains_rate.png").exists()
    assert (out_dir / "example_generations.jsonl").exists()

    plt.close("all")


def test_compute_ranking_metrics() -> None:
    """Ranking metrics should reflect the position of the true candidate."""
    scores = [
        # Context 1: true ranked first
        {"user": "u1", "pii_type": "phone", "context_idx": 0, "label": 1, "score": 1.0, "base_loss": 2.0, "group": "held_out"},
        {"user": "u1", "pii_type": "phone", "context_idx": 0, "label": 0, "score": 0.5, "base_loss": 1.5, "group": "held_out"},
        {"user": "u1", "pii_type": "phone", "context_idx": 0, "label": 0, "score": 0.2, "base_loss": 1.2, "group": "held_out"},
        # Context 2: true ranked second
        {"user": "u1", "pii_type": "phone", "context_idx": 1, "label": 1, "score": 0.4, "base_loss": 1.8, "group": "held_out"},
        {"user": "u1", "pii_type": "phone", "context_idx": 1, "label": 0, "score": 0.9, "base_loss": 1.4, "group": "held_out"},
        {"user": "u1", "pii_type": "phone", "context_idx": 1, "label": 0, "score": 0.1, "base_loss": 1.1, "group": "held_out"},
    ]
    metrics, ranks = _compute_ranking_metrics(scores)
    assert metrics["global"]["n"] == 2
    assert metrics["global"]["top1"] == 0.5
    assert metrics["global"]["top5"] == 1.0
    assert metrics["global"]["mrr"] == (1.0 + 1.0 / 2.0) / 2
    assert ranks == [1, 2]
