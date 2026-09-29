from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.viz_config import VizConfig
from utils.viz_utils import load_json, load_jsonl, save_figure


def is_compatible(result_dir: Path) -> bool:
    """Return True if ``result_dir`` contains a PII autocomplete attack result."""
    if not result_dir.is_dir() or not (result_dir / "generations.jsonl").exists():
        return False
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.exists():
        return False
    metrics = load_json(metrics_path)
    return "by_prompt_set" in metrics


def _plot_rates_by_prompt_set(
    metrics: dict,
    output_path: Path,
    title: str = "Match Rates by Prompt Set",
    cfg: VizConfig | None = None,
) -> plt.Figure:
    """Grouped bar chart of exact/contains rates per prompt set."""
    cfg = cfg or VizConfig()
    by_prompt_set = metrics.get("by_prompt_set", {})
    prompt_sets = list(by_prompt_set.keys())

    ft_contains = [by_prompt_set[ps]["ft_contains"]["rate"] for ps in prompt_sets]
    base_contains = [by_prompt_set[ps]["base_contains"]["rate"] for ps in prompt_sets]
    ft_exact = [by_prompt_set[ps]["ft_exact"]["rate"] for ps in prompt_sets]
    base_exact = [by_prompt_set[ps]["base_exact"]["rate"] for ps in prompt_sets]

    x = np.arange(len(prompt_sets))
    width = 0.2

    fig, ax = plt.subplots(figsize=cfg.full_width)
    ax.bar(x - 1.5 * width, ft_contains, width, label="FT contains", color=cfg.vermillion)
    ax.bar(x - 0.5 * width, base_contains, width, label="Base contains", color=cfg.reddish_purple)
    ax.bar(x + 0.5 * width, ft_exact, width, label="FT exact", color=cfg.blue)
    ax.bar(x + 1.5 * width, base_exact, width, label="Base exact", color=cfg.sky_blue)

    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_sets)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, axis=cfg.grid_axis, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    return fig


def _plot_rates_by_pii_type(
    metrics: dict,
    output_path: Path,
    title: str = "Match Rates by PII Type",
    cfg: VizConfig | None = None,
) -> plt.Figure:
    """Grouped bar chart of exact/contains rates per PII type."""
    cfg = cfg or VizConfig()
    by_pii_type = metrics.get("by_pii_type", {})
    pii_types = list(by_pii_type.keys())

    ft_contains = [by_pii_type[pt]["ft_contains"]["rate"] for pt in pii_types]
    base_contains = [by_pii_type[pt]["base_contains"]["rate"] for pt in pii_types]
    ft_exact = [by_pii_type[pt]["ft_exact"]["rate"] for pt in pii_types]
    base_exact = [by_pii_type[pt]["base_exact"]["rate"] for pt in pii_types]

    x = np.arange(len(pii_types))
    width = 0.2

    fig, ax = plt.subplots(figsize=cfg.full_width)
    ax.bar(x - 1.5 * width, ft_contains, width, label="FT contains", color=cfg.vermillion)
    ax.bar(x - 0.5 * width, base_contains, width, label="Base contains", color=cfg.reddish_purple)
    ax.bar(x + 0.5 * width, ft_exact, width, label="FT exact", color=cfg.blue)
    ax.bar(x + 1.5 * width, base_exact, width, label="Base exact", color=cfg.sky_blue)

    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(pii_types)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, axis=cfg.grid_axis, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    return fig


def _plot_per_user_rate_distribution(
    metrics: dict,
    output_path: Path,
    title: str = "Per-User Contains-Rate Distribution",
    cfg: VizConfig | None = None,
) -> plt.Figure:
    """Box plot of per-user fine-tuned contains rate by group."""
    cfg = cfg or VizConfig()
    per_user = metrics.get("per_user", [])
    held_in = [u["ft_contains_rate"] for u in per_user if u.get("group") == "held_in"]
    held_out = [u["ft_contains_rate"] for u in per_user if u.get("group") == "held_out"]

    fig, ax = plt.subplots(figsize=cfg.single_col)
    bp = ax.boxplot(
        [held_in, held_out],
        patch_artist=True,
        showmeans=True,
    )
    ax.set_xticklabels(["Held-in", "Held-out"])
    colors = [cfg.vermillion, cfg.blue]
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    ax.set_ylabel("Per-user contains rate")
    ax.set_title(title)
    ax.set_ylim([0, 1])
    ax.grid(True, axis=cfg.grid_axis, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    return fig


def _save_example_generations(
    generations: list[dict],
    output_path: Path,
    n_success: int = 10,
    n_failure: int = 5,
) -> None:
    """Save example generations where the model leaked or failed."""
    successes = [g for g in generations if g.get("ft_contains") and not g.get("base_contains")]
    failures = [g for g in generations if not g.get("ft_contains")]

    examples: list[dict] = []
    for g in successes[:n_success]:
        examples.append(
            {
                "category": "ft_leaked_base_did_not",
                "user": g["user"],
                "group": g["group"],
                "pii_type": g["pii_type"],
                "prompt_set": g["prompt_set"],
                "prompt": g["prompt"],
                "true_value": g["true_value"],
                "ft_generation": g["ft_generation"],
                "base_generation": g["base_generation"],
            }
        )
    for g in failures[:n_failure]:
        examples.append(
            {
                "category": "ft_failed",
                "user": g["user"],
                "group": g["group"],
                "pii_type": g["pii_type"],
                "prompt_set": g["prompt_set"],
                "prompt": g["prompt"],
                "true_value": g["true_value"],
                "ft_generation": g["ft_generation"],
                "base_generation": g["base_generation"],
            }
        )

    if examples:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Example generations saved to {output_path}")


def visualize(
    result_dir: Path,
    output_dir: Path | None = None,
    show: bool = False,
    cfg: VizConfig | None = None,
) -> None:
    """Generate PII-autocomplete visualizations for ``result_dir``."""
    cfg = cfg or VizConfig()
    if not is_compatible(result_dir):
        raise ValueError(f"{result_dir} is not a PII autocomplete result directory")

    out = output_dir or result_dir
    metrics = load_json(result_dir / "metrics.json")
    generations = load_jsonl(result_dir / "generations.jsonl")

    global_metrics = metrics.get("global", {})
    summary = metrics.get("per_user_summary", {}) or {}

    print("PII autocomplete attack summary")
    print(f"  FT contains rate:  {global_metrics.get('ft_contains', {}).get('rate', float('nan')):.4f}")
    print(f"  Base contains rate: {global_metrics.get('base_contains', {}).get('rate', float('nan')):.4f}")
    print(f"  FT exact rate:     {global_metrics.get('ft_exact', {}).get('rate', float('nan')):.4f}")
    print(f"  Base exact rate:   {global_metrics.get('base_exact', {}).get('rate', float('nan')):.4f}")
    print(f"  Users:             {summary.get('num_users', 'N/A')}")

    title_prefix = result_dir.name
    _plot_rates_by_prompt_set(
        metrics,
        output_path=out / f"match_rates_by_prompt_set.{cfg.format}",
        title=f"Match Rates by Prompt Set — {title_prefix}",
        cfg=cfg,
    )
    _plot_rates_by_pii_type(
        metrics,
        output_path=out / f"match_rates_by_pii_type.{cfg.format}",
        title=f"Match Rates by PII Type — {title_prefix}",
        cfg=cfg,
    )
    if metrics.get("per_user"):
        _plot_per_user_rate_distribution(
            metrics,
            output_path=out / f"per_user_contains_rate.{cfg.format}",
            title=f"Per-User Contains-Rate Distribution — {title_prefix}",
            cfg=cfg,
        )
    _save_example_generations(
        generations,
        output_path=out / "example_generations.jsonl",
    )

    if show:
        plt.show()
