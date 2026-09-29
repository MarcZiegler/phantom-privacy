"""Compute and visualize general statistics for preprocessed datasets."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.viz_config import VizConfig, apply_viz_style, save_figure


def compute_stats(path: Path) -> dict[str, object]:
    """Return basic statistics for a preprocessed JSONL dataset."""
    total_records = 0
    user_counts: Counter[str] = Counter()
    user_splits: defaultdict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_records += 1
            record = json.loads(line)
            user = record.get("user") or record.get("user_id") or record.get("author")
            split = record.get("split", "unknown")
            if user:
                user_counts[user] += 1
                user_splits[user].add(str(split))
            split_counts[str(split)] += 1

    counts = list(user_counts.values())
    n_users = len(user_counts)

    training_users = sum(1 for splits in user_splits.values() if "train" in splits)
    shadow_users = sum(1 for splits in user_splits.values() if "holdout" in splits)

    return {
        "dataset": path.stem,
        "records": total_records,
        "users": n_users,
        "training_users": training_users,
        "shadow_users": shadow_users,
        "avg_records_per_user": round(sum(counts) / n_users, 1) if n_users else 0.0,
        "median_records_per_user": int(statistics.median(counts)) if counts else 0,
        "min_records_per_user": min(counts) if counts else 0,
        "max_records_per_user": max(counts) if counts else 0,
        "splits": dict(split_counts),
        "user_record_counts": sorted(counts, reverse=True),
    }


def format_number(n: int) -> str:
    """Format large integers with thousand separators."""
    return f"{n:,}"


def build_rows(stats_list: list[dict[str, object]]) -> list[list[str]]:
    """Build table rows for matplotlib/markdown."""
    header = [
        "Dataset",
        "Records",
        "Users",
        "Training / Phantom",
        "Avg / User",
        "Median / User",
        "Min / User",
        "Max / User",
        "Train",
        "Val",
        "Test",
        "Attacker",
        "Holdout",
    ]
    rows: list[list[str]] = [header]

    for stats in stats_list:
        splits = stats["splits"]
        display_name = str(stats.get("label", stats["dataset"]))
        user_split_str = f"{stats['training_users']} / {stats['shadow_users']}"  # Training / Phantom
        rows.append(
            [
                display_name,
                format_number(int(stats["records"])),
                format_number(int(stats["users"])),
                user_split_str,
                str(stats["avg_records_per_user"]),
                format_number(int(stats["median_records_per_user"])),
                format_number(int(stats["min_records_per_user"])),
                format_number(int(stats["max_records_per_user"])),
                format_number(int(splits.get("train", 0))),
                format_number(int(splits.get("val", 0))),
                format_number(int(splits.get("test", 0))),
                format_number(int(splits.get("attacker", 0))),
                format_number(int(splits.get("holdout", 0))),
            ]
        )

    return rows


def save_markdown(rows: list[list[str]], output_path: Path) -> None:
    """Write a markdown table."""
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    def fmt_row(row: list[str]) -> str:
        return "| " + " | ".join(
            cell.ljust(col_widths[i]) for i, cell in enumerate(row)
        ) + " |"

    lines = [fmt_row(rows[0]), "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"]
    lines.extend(fmt_row(row) for row in rows[1:])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_table_figure(rows: list[list[str]], output_path: Path, cfg: VizConfig) -> None:
    """Render a table as a figure."""
    fig, ax = plt.subplots(figsize=(16, 1.2 + 0.5 * len(rows)))
    ax.axis("off")
    ax.axis("tight")

    header_color = cfg.blue
    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="center",
        colColours=[header_color] * len(rows[0]),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(cfg.font_size)
    table.scale(1, 2)

    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f7f7f7" if row % 2 == 0 else "white")
        cell.set_edgecolor("#cccccc")

    ax.set_title("Preprocessed Dataset Statistics", fontweight="bold", pad=10)
    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    plt.close(fig)


def save_scatter_plot(stats_list: list[dict[str, object]], output_path: Path, cfg: VizConfig) -> None:
    """Render a scatter plot of users (x) vs records (y) for each dataset."""
    fig, ax = plt.subplots(figsize=cfg.full_width)

    colors = list(cfg.dataset_colors.values())[: len(stats_list)]
    for i, stats in enumerate(stats_list):
        users = int(stats["users"])
        records = int(stats["records"])
        avg_per_user = float(stats["avg_records_per_user"])
        label = str(stats.get("label", stats["dataset"]))
        color = colors[i % len(colors)]
        ax.scatter(
            users,
            records,
            s=200 + avg_per_user * 0.1,
            c=color,
            alpha=0.75,
            edgecolors="black",
            linewidths=1,
            label=label,
        )
        ax.annotate(
            label,
            (users, records),
            textcoords="offset points",
            xytext=(10, 10),
            fontweight="bold",
        )

    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Number of Records")
    ax.set_title("Users vs Records per Dataset", fontweight="bold", pad=10)
    ax.grid(True, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    ax.legend(title="Dataset", loc="upper left")

    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    plt.close(fig)


def save_distribution_plot(stats_list: list[dict[str, object]], output_path: Path, cfg: VizConfig) -> None:
    """Render a histogram of records per user using log-spaced bins."""
    fig, ax = plt.subplots(figsize=cfg.full_width)

    colors = list(cfg.dataset_colors.values())[: len(stats_list)]
    # Start the x-axis just below the per-dataset minimum threshold so the bars fill the plot.
    x_min_threshold = 100
    for i, stats in enumerate(stats_list):
        counts = list(stats["user_record_counts"])
        label = str(stats.get("label", stats["dataset"]))
        max_count = max(counts) if counts else 1
        # Log-spaced bins from the threshold up to slightly above the max count.
        log_min = math.log10(x_min_threshold)
        log_max = math.log10(max_count * 1.2)
        n_bins = 24
        bins = [10 ** (log_min + i * (log_max - log_min) / (n_bins - 1)) for i in range(n_bins)]
        ax.hist(
            counts,
            bins=bins,
            color=colors[i % len(colors)],
            alpha=0.6,
            edgecolor="black",
            linewidth=0.8,
            label=label,
        )

    ax.set_xlabel("Number of Records per User", labelpad=6)
    ax.set_ylabel("Number of Users", labelpad=6)
    ax.grid(True, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha)
    ax.legend(title="Dataset", loc="upper right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=x_min_threshold * 0.9)

    fig.tight_layout()
    save_figure(fig, output_path, cfg)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute general statistics for preprocessed JSONL datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Paths to preprocessed JSONL datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to save the generated table outputs.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="pdf",
        help="Output figure format (default: pdf).",
    )
    parser.add_argument(
        "--table-name",
        default="dataset_statistics",
        help="Base filename for the table figure (without extension).",
    )
    parser.add_argument(
        "--md-name",
        default="dataset_statistics.md",
        help="Filename for the markdown table.",
    )
    parser.add_argument(
        "--scatter-name",
        default="dataset_users_vs_records",
        help="Base filename for the users vs records scatter plot (without extension).",
    )
    parser.add_argument(
        "--distribution-name",
        default="dataset_records_per_user_distribution",
        help="Base filename for the records-per-user distribution plot (without extension).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional human-readable dataset labels used in legends and tables.",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.datasets):
        raise ValueError(
            f"--labels must provide exactly one label per dataset "
            f"(got {len(args.labels)} labels for {len(args.datasets)} datasets)."
        )

    cfg = apply_viz_style(VizConfig(format=args.format))
    stats_cfg = apply_viz_style(
        dataclasses.replace(
            cfg,
            font_size=12,
            label_font_size=13,
            tick_font_size=12,
            title_font_size=13,
            legend_font_size=12,
            annotation_font_size=10,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats_list = [compute_stats(Path(p)) for p in args.datasets]
    for i, stats in enumerate(stats_list):
        stats["label"] = args.labels[i] if args.labels else str(stats["dataset"])
    rows = build_rows(stats_list)

    md_path = args.output_dir / args.md_name
    table_path = args.output_dir / f"{args.table_name}.{stats_cfg.format}"
    scatter_path = args.output_dir / f"{args.scatter_name}.{stats_cfg.format}"
    distribution_path = args.output_dir / f"{args.distribution_name}.{stats_cfg.format}"

    save_markdown(rows, md_path)
    save_table_figure(rows, table_path, stats_cfg)
    save_scatter_plot(stats_list, scatter_path, stats_cfg)
    save_distribution_plot(stats_list, distribution_path, stats_cfg)

    print(f"Saved markdown table: {md_path}")
    print(f"Saved table figure: {table_path}")
    print(f"Saved scatter plot: {scatter_path}")
    print(f"Saved distribution plot: {distribution_path}")
    print()
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
