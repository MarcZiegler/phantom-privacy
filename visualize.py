#!/usr/bin/env python
"""Interactive visualization dispatcher for all result types."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.viz_config import VizConfig, apply_viz_style
from utils.viz_pii_autocomplete import visualize as visualize_autocomplete
from utils.viz_pii_extraction import visualize as visualize_pii_extraction
from utils.viz_utils import ResultType, detect_result_type, list_result_dirs

_VIZUALIZERS: dict[ResultType, callable] = {
    ResultType.PII_EXTRACTION: visualize_pii_extraction,
    ResultType.PII_AUTOCOMPLETE: visualize_autocomplete,
}


def _format_type(result_type: ResultType) -> str:
    """Short human-readable label for a result type."""
    return {
        ResultType.PII_EXTRACTION: "PII extraction",
        ResultType.PII_AUTOCOMPLETE: "PII autocomplete",
        ResultType.UNKNOWN: "unknown",
    }[result_type]


def select_result(results_root: Path) -> Path:
    """List available result directories and let the user pick one."""
    dirs = list_result_dirs(results_root)
    if not dirs:
        print(f"No result directories found in {results_root}")
        sys.exit(1)

    print(f"\nFound {len(dirs)} result run(s) in {results_root}:\n")
    for i, d in enumerate(dirs, 1):
        rel = d.relative_to(results_root)
        rtype = detect_result_type(d)
        print(f"  {i}. [{_format_type(rtype)}] {rel}")

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
        description="Visualize attack results (standard, PII extraction, PII autocomplete)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory containing result folders.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run folder name or path. Skips interactive selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots are saved (default: the result folder).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively with plt.show().",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="For PII extraction results, also generate the fine-tuned vs. base-model plot.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output figure format (default: png).",
    )
    args = parser.parse_args()

    cfg = apply_viz_style(VizConfig(format=args.format))

    if args.run:
        candidate = Path(args.run)
        result_dir = candidate if candidate.is_absolute() else args.results_dir / candidate
        if not result_dir.exists():
            print(f"Run not found: {result_dir}")
            sys.exit(1)
    else:
        result_dir = select_result(args.results_dir)

    result_type = detect_result_type(result_dir)
    if result_type == ResultType.UNKNOWN:
        print(
            f"Could not detect result type for {result_dir}. "
            "Expected: PII extraction or PII autocomplete."
        )
        sys.exit(1)

    print(f"\nDetected result type: {_format_type(result_type)}")

    visualize_func = _VIZUALIZERS[result_type]
    call_args: dict = {
        "result_dir": result_dir,
        "output_dir": args.output_dir,
        "show": args.show,
        "cfg": cfg,
    }
    if result_type == ResultType.PII_EXTRACTION:
        call_args["include_base"] = args.include_base

    visualize_func(**call_args)

    print(f"\nPlots saved inside {args.output_dir or result_dir}")


if __name__ == "__main__":
    main()
