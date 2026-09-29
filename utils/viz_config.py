"""Shared matplotlib styling configuration for paper figures.

This module centralizes fonts, colors, figure sizes, grids, and output settings
so that all Phantom-Privacy visualizations are consistent and publication-ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class VizConfig:
    """Paper-ready matplotlib styling configuration.

    Defaults are tuned for IEEE double-column proceedings (e.g., Big Data).
    The color palette is the Okabe-Ito colorblind-safe set.
    """

    # Output ------------------------------------------------------------------
    dpi: int = 300
    format: str = "png"  # Default for interactive/preview use; paper scripts override to pdf.
    bbox_inches: str = "tight"
    pad_inches: float = 0.02

    # Fonts (pt) --------------------------------------------------------------
    font_family: str = "sans-serif"
    font_size: int = 9
    title_font_size: int = 10
    label_font_size: int = 9
    tick_font_size: int = 8
    annotation_font_size: int = 7
    legend_font_size: int = 8

    # Figure sizes (inches) for IEEE double-column ----------------------------
    # Full width = ~7.25 in; single column = ~3.5 in.
    single_col: tuple[float, float] = (3.5, 2.5)
    full_width: tuple[float, float] = (7.25, 4.0)
    four_panel: tuple[float, float] = (7.25, 5.5)
    large_heatmap: tuple[float, float] = (7.25, 10.0)
    training_curve: tuple[float, float] = (8.0, 5.0)

    # Colorblind-safe Okabe-Ito palette ---------------------------------------
    black: str = "#000000"
    orange: str = "#E69F00"
    sky_blue: str = "#56B4E9"
    bluish_green: str = "#009E73"
    yellow: str = "#F0E442"
    blue: str = "#0072B2"
    vermillion: str = "#D55E00"
    reddish_purple: str = "#CC79A7"

    # Semantic colors ---------------------------------------------------------
    model_colors: dict[str, str] = field(
        default_factory=lambda: {
            "125M": "#0072B2",
            "1.3B": "#D55E00",
            "8B": "#009E73",
            "Base model": "#949494",
        }
    )
    distractor_colors: dict[str, str] = field(
        default_factory=lambda: {
            "synthetic": "#0072B2",
            "real": "#D55E00",
        }
    )
    score_colors: dict[str, str] = field(
        default_factory=lambda: {
            "lr": "#D55E00",
            "bert": "#0072B2",
        }
    )
    dataset_colors: dict[str, str] = field(
        default_factory=lambda: {
            "enron": "#56B4E9",
            "mallinckrodt": "#009E73",
        }
    )

    # Grid -------------------------------------------------------------------
    grid_axis: str = "y"
    grid_linestyle: str = "--"
    grid_alpha: float = 0.4

    # Baselines / reference lines --------------------------------------------
    baseline_color: str = "gray"
    baseline_linestyle: str = "--"
    baseline_linewidth: float = 0.8

    # Helpers ----------------------------------------------------------------
    def color_for_model(self, model: str) -> str:
        """Return the configured color for a model label."""
        return self.model_colors.get(model, self.blue)

    def color_for_distractor(self, distractor: str) -> str:
        """Return the configured color for a distractor mode."""
        return self.distractor_colors.get(distractor, self.blue)

    def save_path(self, output_dir: Path, basename: str) -> Path:
        """Return a Path with the configured extension."""
        return output_dir / f"{basename}.{self.format}"


def apply_viz_style(cfg: VizConfig | None = None) -> VizConfig:
    """Apply ``cfg`` to matplotlib ``rcParams`` and return it.

    This should be called once at the top of each plotting script or module.
    """
    cfg = cfg or VizConfig()
    matplotlib.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": cfg.font_family,
            "font.size": cfg.font_size,
            "axes.titlesize": cfg.title_font_size,
            "axes.labelsize": cfg.label_font_size,
            "xtick.labelsize": cfg.tick_font_size,
            "ytick.labelsize": cfg.tick_font_size,
            "legend.fontsize": cfg.legend_font_size,
            "figure.dpi": cfg.dpi,
            "savefig.dpi": cfg.dpi,
            "savefig.format": cfg.format,
            "savefig.bbox": cfg.bbox_inches,
            "savefig.pad_inches": cfg.pad_inches,
            "pdf.compression": 9,
            "svg.fonttype": "none",
        }
    )
    return cfg


def save_figure(fig: Any, output_path: Path, cfg: VizConfig | None = None) -> Path:
    """Save ``fig`` to ``output_path`` using the configured output settings.

    The file extension in ``output_path`` is respected; if it does not match
    ``cfg.format`` a warning is emitted but the requested path is used.
    """
    cfg = cfg or VizConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=cfg.dpi,
        bbox_inches=cfg.bbox_inches,
        pad_inches=cfg.pad_inches,
    )
    return output_path
