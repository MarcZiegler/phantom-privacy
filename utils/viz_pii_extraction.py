from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from utils.viz_config import VizConfig
from utils.viz_utils import load_json, load_jsonl, save_figure


def is_compatible(result_dir: Path) -> bool:
    """Return True if ``result_dir`` contains a PII extraction attack result."""
    return (
        result_dir.is_dir()
        and (result_dir / "metrics.json").exists()
        and (result_dir / "extraction_scores.jsonl").exists()
        and (result_dir / "exposure_table.jsonl").exists()
    )


def _compute_per_user_metrics(
    scores: list[dict],
    exposure_rows: list[dict],
) -> tuple[list[dict], dict[tuple[str, str], int]]:
    """Compute per-user/PII-type AUROC and exposure.

    Returns a list of records and a mapping from (user, pii_type) to exposure.
    """
    exposure_map: dict[tuple[str, str], int] = {}
    for row in exposure_rows:
        if row.get("value") is not None:
            exposure_map[(row["user"], row["pii_type"])] = row.get("exposure_total", 0)

    by_key: dict[
        tuple[str, str],
        dict,
    ] = defaultdict(
        lambda: {
            "true_scores": [],
            "false_scores": [],
            "true_base": [],
            "false_base": [],
            "group": None,
            "pii_type": None,
        }
    )

    for row in scores:
        key = (row["user"], row["pii_type"])
        entry = by_key[key]
        entry["group"] = row.get("group")
        entry["pii_type"] = row["pii_type"]
        if row["label"] == 1:
            entry["true_scores"].append(float(row["score"]))
            entry["true_base"].append(-float(row["base_loss"]))
        else:
            entry["false_scores"].append(float(row["score"]))
            entry["false_base"].append(-float(row["base_loss"]))

    records: list[dict] = []
    for (user, _pii_type), entry in by_key.items():
        y_true = [1] * len(entry["true_scores"]) + [0] * len(entry["false_scores"])
        if len(set(y_true)) < 2:
            continue
        auroc = roc_auc_score(
            y_true,
            entry["true_scores"] + entry["false_scores"],
        )
        base_auroc = roc_auc_score(
            y_true,
            entry["true_base"] + entry["false_base"],
        )
        exposure = exposure_map.get((user, _pii_type), 0)
        records.append(
            {
                "user": user,
                "pii_type": _pii_type,
                "group": entry["group"],
                "exposure": exposure,
                "auroc": auroc,
                "base_auroc": base_auroc,
            }
        )

    return records, exposure_map


def _compute_ranking_metrics(
    scores: list[dict],
    pii_type: str | None = None,
) -> tuple[dict, list[int]]:
    """Compute top-k accuracy and MRR from extraction scores.

    Returns a nested metrics dictionary and the list of true ranks.
    """
    by_context: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in scores:
        if pii_type is not None and row["pii_type"] != pii_type:
            continue
        key = (row["user"], row["pii_type"], row.get("context_idx", 0))
        by_context[key].append(row)

    ranks: list[int] = []
    by_group: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"top1": [], "top5": [], "top10": [], "mrr": []}
    )
    by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"top1": [], "top5": [], "top10": [], "mrr": []}
    )
    by_user: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"top1": [], "top5": [], "top10": [], "mrr": []}
    )

    for ctx_rows in by_context.values():
        true_rows = [r for r in ctx_rows if r["label"] == 1]
        if not true_rows:
            continue
        true_score = true_rows[0]["score"]
        # Rank = 1 + number of distractors with strictly higher score.
        rank = 1 + sum(1 for r in ctx_rows if r["label"] == 0 and r["score"] > true_score)
        ranks.append(rank)

        top1 = float(rank == 1)
        top5 = float(rank <= 5)
        top10 = float(rank <= 10)
        rr = 1.0 / rank

        group = true_rows[0].get("group", "unknown")
        pii_type = true_rows[0]["pii_type"]
        user = true_rows[0]["user"]

        for bucket in (by_group[group], by_type[pii_type], by_user[(user, pii_type)]):
            bucket["top1"].append(top1)
            bucket["top5"].append(top5)
            bucket["top10"].append(top10)
            bucket["mrr"].append(rr)

    def _aggregate(values: dict[str, list[float]]) -> dict[str, float]:
        n = len(values["top1"])
        if n == 0:
            return {"n": 0, "top1": 0.0, "top5": 0.0, "top10": 0.0, "mrr": 0.0}
        return {
            "n": n,
            "top1": float(np.mean(values["top1"])),
            "top5": float(np.mean(values["top5"])),
            "top10": float(np.mean(values["top10"])),
            "mrr": float(np.mean(values["mrr"])),
        }

    metrics = {
        "global": _aggregate_ranks(ranks),
        "by_group": {k: _aggregate(v) for k, v in by_group.items()},
        "by_pii_type": {k: _aggregate(v) for k, v in by_type.items()},
        "by_user": {k: _aggregate(v) for k, v in by_user.items()},
    }
    return metrics, ranks


def _aggregate_ranks(ranks: list[int]) -> dict[str, float]:
    """Return top-k accuracy and MRR for a list of ranks."""
    n = len(ranks)
    if n == 0:
        return {"n": 0, "top1": 0.0, "top5": 0.0, "top10": 0.0, "mrr": 0.0}
    return {
        "n": n,
        "top1": sum(1 for r in ranks if r == 1) / n,
        "top5": sum(1 for r in ranks if r <= 5) / n,
        "top10": sum(1 for r in ranks if r <= 10) / n,
        "mrr": sum(1.0 / r for r in ranks) / n,
    }


def _plot_exposure_vs_auroc(
    records: list[dict],
    output_path: Path,
    title: str = "Exposure vs. Per-User AUROC",
    pii_type: str | None = None,
    cfg: VizConfig | None = None,
) -> plt.Figure:
    """Scatter plot of exposure count vs. per-user AUROC."""
    cfg = cfg or VizConfig()
    if pii_type is not None:
        records = [r for r in records if r["pii_type"] == pii_type]
    exposures = np.array([r["exposure"] for r in records], dtype=float)
    aurocs = np.array([r["auroc"] for r in records], dtype=float)
    groups = [r["group"] for r in records]

    fig, ax = plt.subplots(figsize=cfg.full_width)
    colors = {"held_in": cfg.vermillion, "held_out": cfg.blue}
    labels = {"held_in": "Held-in", "held_out": "Held-out"}
    for group in sorted(set(groups)):
        mask = np.array([g == group for g in groups])
        ax.scatter(
            exposures[mask] + 1,
            aurocs[mask],
            alpha=0.6,
            s=40,
            color=colors.get(group, cfg.black),
            label=labels.get(group, group),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Exposure count in held-in training records (log scale, x+1)")
    ax.set_ylabel("Per-user AUROC")
    ax.set_title(title)
    ax.set_ylim([0, 1])
    ax.axhline(
        0.5,
        color=cfg.baseline_color,
        linestyle=cfg.baseline_linestyle,
        linewidth=cfg.baseline_linewidth,
        label="Random",
    )
    ax.legend()
    ax.grid(True, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    plt.close(fig)
    return fig


def _plot_exposure_bins(
    records: list[dict],
    output_path: Path,
    title: str = "Held-Out AUROC by Exposure Bin",
    pii_type: str | None = None,
    cfg: VizConfig | None = None,
) -> plt.Figure:
    """Bar chart of mean held-out AUROC across exposure bins."""
    cfg = cfg or VizConfig()
    if pii_type is not None:
        records = [r for r in records if r["pii_type"] == pii_type]
    held_out = [r for r in records if r["group"] == "held_out"]

    bins = [
        (0, 0, "0"),
        (1, 5, "1–5"),
        (6, 20, "6–20"),
        (21, 100, "21–100"),
        (101, 10**9, "101+"),
    ]
    labels: list[str] = []
    means: list[float] = []
    counts: list[int] = []
    for lo, hi, label in bins:
        subset = [r for r in held_out if lo <= r["exposure"] <= hi]
        if subset:
            labels.append(label)
            means.append(float(np.mean([r["auroc"] for r in subset])))
            counts.append(len(subset))

    fig, ax = plt.subplots(figsize=cfg.full_width)
    bars = ax.bar(labels, means, color=cfg.blue)
    ax.set_xlabel("Exposure count in held-in training records")
    ax.set_ylabel("Mean held-out AUROC")
    ax.set_title(title)
    ax.set_ylim([0, 1])
    ax.axhline(
        0.5,
        color=cfg.baseline_color,
        linestyle=cfg.baseline_linestyle,
        linewidth=cfg.baseline_linewidth,
    )
    ax.grid(True, axis=cfg.grid_axis, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)

    for bar, count in zip(bars, counts, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=cfg.annotation_font_size,
        )

    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    plt.close(fig)
    return fig


def _plot_group_comparison(
    records: list[dict],
    output_path: Path,
    title: str = "Per-User AUROC: Held-In vs. Held-Out",
    pii_type: str | None = None,
    cfg: VizConfig | None = None,
) -> plt.Figure:
    """Box plot comparing per-user AUROC distributions by group."""
    cfg = cfg or VizConfig()
    if pii_type is not None:
        records = [r for r in records if r["pii_type"] == pii_type]
    held_in = [r["auroc"] for r in records if r["group"] == "held_in"]
    held_out = [r["auroc"] for r in records if r["group"] == "held_out"]

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

    ax.set_ylabel("Per-user AUROC")
    ax.set_title(title)
    ax.set_ylim([0, 1])
    ax.axhline(
        0.5,
        color=cfg.baseline_color,
        linestyle=cfg.baseline_linestyle,
        linewidth=cfg.baseline_linewidth,
    )
    ax.grid(True, axis=cfg.grid_axis, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    plt.close(fig)
    return fig


def _plot_finetuned_vs_base(
    records: list[dict],
    output_path: Path,
    title: str = "Fine-Tuned vs. Base Model Per-User AUROC",
    pii_type: str | None = None,
    cfg: VizConfig | None = None,
) -> plt.Figure:
    """Scatter plot comparing fine-tuned and base-model AUROC per user."""
    cfg = cfg or VizConfig()
    if pii_type is not None:
        records = [r for r in records if r["pii_type"] == pii_type]
    aurocs = np.array([r["auroc"] for r in records])
    base_aurocs = np.array([r["base_auroc"] for r in records])
    groups = [r["group"] for r in records]

    fig, ax = plt.subplots(figsize=cfg.single_col)
    colors = {"held_in": cfg.vermillion, "held_out": cfg.blue}
    labels = {"held_in": "Held-in", "held_out": "Held-out"}
    for group in sorted(set(groups)):
        mask = np.array([g == group for g in groups])
        ax.scatter(
            base_aurocs[mask],
            aurocs[mask],
            alpha=0.6,
            s=40,
            color=colors.get(group, cfg.black),
            label=labels.get(group, group),
        )

    lims = [min(0, base_aurocs.min(), aurocs.min()), max(1, base_aurocs.max(), aurocs.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="No change")
    ax.set_xlabel("Base model per-user AUROC")
    ax.set_ylabel("Fine-tuned per-user AUROC")
    ax.set_title(title)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.grid(True, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    plt.close(fig)
    return fig


def _plot_top_k_by_pii_type(
    by_type: dict[str, dict],
    output_path: Path,
    title: str = "Top-k Accuracy by PII Type",
    cfg: VizConfig | None = None,
) -> plt.Figure:
    """Grouped bar chart of top-1/top-5/top-10 accuracy per PII type."""
    cfg = cfg or VizConfig()
    pii_types = list(by_type.keys())
    top1 = [by_type[t]["top1"] for t in pii_types]
    top5 = [by_type[t]["top5"] for t in pii_types]
    top10 = [by_type[t]["top10"] for t in pii_types]

    x = np.arange(len(pii_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=cfg.full_width)
    ax.bar(x - width, top1, width, label="Top-1", color=cfg.blue)
    ax.bar(x, top5, width, label="Top-5", color=cfg.vermillion)
    ax.bar(x + width, top10, width, label="Top-10", color=cfg.bluish_green)

    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(pii_types)
    ax.set_ylim([0, 1])
    ax.axhline(
        1.0 / 21.0,
        color=cfg.baseline_color,
        linestyle=cfg.baseline_linestyle,
        linewidth=cfg.baseline_linewidth,
        label="Random (1/21)",
    )
    ax.legend()
    ax.grid(True, axis=cfg.grid_axis, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    plt.close(fig)
    return fig


def _plot_top_k_by_group(
    by_group: dict[str, dict],
    output_path: Path,
    title: str = "Top-k Accuracy: Held-In vs. Held-Out",
    cfg: VizConfig | None = None,
) -> plt.Figure:
    """Grouped bar chart of top-k accuracy by group."""
    cfg = cfg or VizConfig()
    groups = [g for g in ("held_in", "held_out") if g in by_group]
    top1 = [by_group[g]["top1"] for g in groups]
    top5 = [by_group[g]["top5"] for g in groups]
    top10 = [by_group[g]["top10"] for g in groups]

    x = np.arange(len(groups))
    width = 0.25

    fig, ax = plt.subplots(figsize=cfg.single_col)
    ax.bar(x - width, top1, width, label="Top-1", color=cfg.blue)
    ax.bar(x, top5, width, label="Top-5", color=cfg.vermillion)
    ax.bar(x + width, top10, width, label="Top-10", color=cfg.bluish_green)

    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", "-").title() for g in groups])
    ax.set_ylim([0, 1])
    ax.axhline(
        1.0 / 21.0,
        color=cfg.baseline_color,
        linestyle=cfg.baseline_linestyle,
        linewidth=cfg.baseline_linewidth,
        label="Random (1/21)",
    )
    ax.legend()
    ax.grid(True, axis=cfg.grid_axis, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    plt.close(fig)
    return fig


def _plot_rank_distribution(
    ranks: list[int],
    output_path: Path,
    title: str = "Distribution of True Ranks",
    cfg: VizConfig | None = None,
) -> plt.Figure:
    """Histogram of the rank of the true candidate across all contexts."""
    cfg = cfg or VizConfig()
    fig, ax = plt.subplots(figsize=cfg.full_width)
    max_rank = min(max(ranks, default=1), 50)
    bins = np.arange(1, max_rank + 3) - 0.5
    ax.hist(ranks, bins=bins, color=cfg.blue, edgecolor=cfg.black)
    ax.set_xlabel("Rank of true candidate")
    ax.set_ylabel("Number of contexts")
    ax.set_title(title)
    ax.axvline(1.5, color=cfg.vermillion, linestyle="--", linewidth=1, label="Top-1 cutoff")
    ax.axvline(5.5, color=cfg.orange, linestyle="--", linewidth=1, label="Top-5 cutoff")
    ax.legend()
    ax.grid(True, axis=cfg.grid_axis, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    plt.close(fig)
    return fig


def _plot_top1_vs_exposure(
    records: list[dict],
    output_path: Path,
    title: str = "Top-1 Accuracy vs. Exposure",
    pii_type: str | None = None,
    cfg: VizConfig | None = None,
) -> plt.Figure:
    """Scatter plot of per-user top-1 accuracy vs. exposure count."""
    cfg = cfg or VizConfig()
    if pii_type is not None:
        records = [r for r in records if r["pii_type"] == pii_type]
    exposures = np.array([r["exposure"] for r in records], dtype=float)
    top1s = np.array([r.get("top1", 0.0) for r in records], dtype=float)
    groups = [r["group"] for r in records]

    fig, ax = plt.subplots(figsize=cfg.full_width)
    colors = {"held_in": cfg.vermillion, "held_out": cfg.blue}
    labels = {"held_in": "Held-in", "held_out": "Held-out"}
    for group in sorted(set(groups)):
        mask = np.array([g == group for g in groups])
        ax.scatter(
            exposures[mask] + 1,
            top1s[mask],
            alpha=0.6,
            s=40,
            color=colors.get(group, cfg.black),
            label=labels.get(group, group),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Exposure count (log scale, x+1)")
    ax.set_ylabel("Per-user top-1 accuracy")
    ax.set_title(title)
    ax.set_ylim([0, 1])
    ax.axhline(
        1.0 / 21.0,
        color=cfg.baseline_color,
        linestyle=cfg.baseline_linestyle,
        linewidth=cfg.baseline_linewidth,
        label="Random",
    )
    ax.legend()
    ax.grid(True, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    plt.close(fig)
    return fig


def visualize(
    result_dir: Path,
    output_dir: Path | None = None,
    show: bool = False,
    include_base: bool = False,
    cfg: VizConfig | None = None,
) -> None:
    """Generate PII-extraction visualizations for ``result_dir``."""
    cfg = cfg or VizConfig()
    if not is_compatible(result_dir):
        raise ValueError(f"{result_dir} is not a PII extraction result directory")

    out = output_dir or result_dir
    metrics = load_json(result_dir / "metrics.json")
    scores = load_jsonl(result_dir / "extraction_scores.jsonl")
    exposure_rows = load_jsonl(result_dir / "exposure_table.jsonl")

    records, _exposure_map = _compute_per_user_metrics(scores, exposure_rows)
    if not records:
        print("No per-user metrics could be computed (all users have only one class).")
        return

    ranking_metrics, ranks = _compute_ranking_metrics(scores)
    for r in records:
        user_topk = ranking_metrics["by_user"].get((r["user"], r["pii_type"]), {})
        r["top1"] = user_topk.get("top1", 0.0)
        r["top5"] = user_topk.get("top5", 0.0)
        r["top10"] = user_topk.get("top10", 0.0)
        r["mrr"] = user_topk.get("mrr", 0.0)

    global_auroc = metrics.get("global", {}).get("auroc", float("nan"))
    by_group = metrics.get("by_group", {})
    held_in_auroc = by_group.get("held_in", {}).get("auroc", float("nan"))
    held_out_auroc = by_group.get("held_out", {}).get("auroc", float("nan"))

    held_in_records = [r for r in records if r["group"] == "held_in"]
    held_out_records = [r for r in records if r["group"] == "held_out"]
    hi_mean = float(np.mean([r["auroc"] for r in held_in_records])) if held_in_records else float("nan")
    ho_mean = float(np.mean([r["auroc"] for r in held_out_records])) if held_out_records else float("nan")

    exposures_all = [r["exposure"] for r in records]
    aurocs_all = [r["auroc"] for r in records]
    if len(set(exposures_all)) <= 1 or len(set(aurocs_all)) <= 1:
        rho, pvalue = float("nan"), float("nan")
    else:
        rho, pvalue = spearmanr(exposures_all, aurocs_all)

    high_exposure = [r["auroc"] for r in held_out_records if r["exposure"] >= 101]
    high_mean = float(np.mean(high_exposure)) if high_exposure else float("nan")

    rm_global = ranking_metrics["global"]
    rm_group = ranking_metrics["by_group"]
    rm_type = ranking_metrics["by_pii_type"]

    print("PII extraction attack summary")
    print(f"  Global AUROC:        {global_auroc:.4f}")
    print(f"  Held-in AUROC:       {held_in_auroc:.4f}")
    print(f"  Held-out AUROC:      {held_out_auroc:.4f}")
    print(f"  Per-user held-in:    {hi_mean:.4f}")
    print(f"  Per-user held-out:   {ho_mean:.4f}")
    print(f"  Gap:                 {hi_mean - ho_mean:.4f}")
    rho_str = f"{rho:.3f}" if not np.isnan(rho) else "nan"
    p_str = f"{pvalue:.4g}" if not np.isnan(pvalue) else "nan"
    print(f"  Exposure vs AUROC:   Spearman rho = {rho_str} (p = {p_str})")
    print(f"  Held-out >=101 exp:  {high_mean:.4f} ({len(high_exposure)} users)")
    print()
    print("Ranking metrics (top-k accuracy / MRR)")
    print(f"  Global:  top-1={rm_global['top1']:.3f}  top-5={rm_global['top5']:.3f}  top-10={rm_global['top10']:.3f}  MRR={rm_global['mrr']:.3f}  (n={rm_global['n']})")
    for group in sorted(rm_group.keys()):
        g = rm_group[group]
        print(f"  {group}: top-1={g['top1']:.3f}  top-5={g['top5']:.3f}  top-10={g['top10']:.3f}  MRR={g['mrr']:.3f}  (n={g['n']})")
    for pii_type in sorted(rm_type.keys()):
        t = rm_type[pii_type]
        print(f"  {pii_type}: top-1={t['top1']:.3f}  top-5={t['top5']:.3f}  top-10={t['top10']:.3f}  MRR={t['mrr']:.3f}  (n={t['n']})")

    title_prefix = result_dir.name
    _plot_exposure_vs_auroc(
        records,
        output_path=out / f"exposure_vs_auroc.{cfg.format}",
        title=f"Exposure vs. Per-User AUROC — {title_prefix}",
        cfg=cfg,
    )
    _plot_exposure_bins(
        records,
        output_path=out / f"held_out_exposure_bins.{cfg.format}",
        title=f"Held-Out AUROC by Exposure — {title_prefix}",
        cfg=cfg,
    )
    _plot_group_comparison(
        records,
        output_path=out / f"held_in_vs_held_out_auroc.{cfg.format}",
        title=f"Per-User AUROC by Group — {title_prefix}",
        cfg=cfg,
    )
    if include_base:
        _plot_finetuned_vs_base(
            records,
            output_path=out / f"finetuned_vs_base_auroc.{cfg.format}",
            title=f"Fine-Tuned vs. Base Model — {title_prefix}",
            cfg=cfg,
        )

    _plot_top_k_by_pii_type(
        rm_type,
        output_path=out / f"top_k_by_pii_type.{cfg.format}",
        title=f"Top-k Accuracy by PII Type — {title_prefix}",
        cfg=cfg,
    )
    _plot_top_k_by_group(
        rm_group,
        output_path=out / f"top_k_by_group.{cfg.format}",
        title=f"Top-k Accuracy by Group — {title_prefix}",
        cfg=cfg,
    )
    _plot_rank_distribution(
        ranks,
        output_path=out / f"rank_distribution.{cfg.format}",
        title=f"Distribution of True Ranks — {title_prefix}",
        cfg=cfg,
    )
    _plot_top1_vs_exposure(
        records,
        output_path=out / f"top1_vs_exposure.{cfg.format}",
        title=f"Top-1 Accuracy vs. Exposure — {title_prefix}",
        cfg=cfg,
    )

    # Per-PII-type exposure/effectiveness plots.
    pii_types = sorted({r["pii_type"] for r in records})
    if len(pii_types) > 1:
        print()
        print("Per-PII-type summaries")
        for pii_type in pii_types:
            type_records = [r for r in records if r["pii_type"] == pii_type]
            type_exposures = [r["exposure"] for r in type_records]
            type_aurocs = [r["auroc"] for r in type_records]
            if len(set(type_exposures)) > 1 and len(set(type_aurocs)) > 1:
                rho_t, p_t = spearmanr(type_exposures, type_aurocs)
            else:
                rho_t, p_t = float("nan"), float("nan")
            type_hi = [r["auroc"] for r in type_records if r["group"] == "held_in"]
            type_ho = [r["auroc"] for r in type_records if r["group"] == "held_out"]
            print(
                f"  {pii_type}: n={len(type_records)}, "
                f"held-in mean={np.mean(type_hi):.3f}, held-out mean={np.mean(type_ho):.3f}, "
                f"rho={rho_t:.3f} (p={p_t:.4g})"
            )

            _plot_exposure_vs_auroc(
                records,
                output_path=out / f"exposure_vs_auroc_{pii_type}.{cfg.format}",
                title=f"Exposure vs. AUROC — {pii_type} — {title_prefix}",
                pii_type=pii_type,
                cfg=cfg,
            )
            _plot_exposure_bins(
                records,
                output_path=out / f"held_out_exposure_bins_{pii_type}.{cfg.format}",
                title=f"Held-Out AUROC by Exposure — {pii_type} — {title_prefix}",
                pii_type=pii_type,
                cfg=cfg,
            )
            _plot_group_comparison(
                records,
                output_path=out / f"held_in_vs_held_out_auroc_{pii_type}.{cfg.format}",
                title=f"Per-User AUROC by Group — {pii_type} — {title_prefix}",
                pii_type=pii_type,
                cfg=cfg,
            )
            _plot_top1_vs_exposure(
                records,
                output_path=out / f"top1_vs_exposure_{pii_type}.{cfg.format}",
                title=f"Top-1 Accuracy vs. Exposure — {pii_type} — {title_prefix}",
                pii_type=pii_type,
                cfg=cfg,
            )
            if include_base:
                _plot_finetuned_vs_base(
                    records,
                    output_path=out / f"finetuned_vs_base_auroc_{pii_type}.{cfg.format}",
                    title=f"Fine-Tuned vs. Base — {pii_type} — {title_prefix}",
                    pii_type=pii_type,
                    cfg=cfg,
                )

            _, type_ranks = _compute_ranking_metrics(scores, pii_type=pii_type)
            if type_ranks:
                _plot_rank_distribution(
                    type_ranks,
                    output_path=out / f"rank_distribution_{pii_type}.{cfg.format}",
                    title=f"Distribution of True Ranks — {pii_type} — {title_prefix}",
                    cfg=cfg,
                )

    if show:
        plt.show()
