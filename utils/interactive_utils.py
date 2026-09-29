"""Interactive selection helpers for CLI launchers."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobs.presets import FinetunePreset


def _is_interactive() -> bool:
    """Return True if stdin is connected to an interactive terminal."""
    return sys.stdin.isatty()


def select_preset(presets: dict[str, FinetunePreset], title: str) -> FinetunePreset:
    """Display a numbered list of presets and return the selected one.

    Args:
        presets: Mapping from preset name to preset object.
        title: Human-readable label for the list (e.g. "fine-tuning").

    Returns:
        The selected preset object.

    Raises:
        SystemExit: If stdin is not interactive, no presets are available,
            or the user chooses to quit.
    """
    if not _is_interactive():
        print(
            "ERROR: No --preset given and stdin is not interactive.",
            file=sys.stderr,
        )
        print(
            "Run with --preset <name> or use an interactive terminal.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not presets:
        print(f"ERROR: No {title} presets available.", file=sys.stderr)
        sys.exit(1)

    options = sorted(presets.items(), key=lambda item: item[0])
    print(f"\nAvailable {title} presets:\n")
    for i, (name, preset) in enumerate(options, 1):
        print(
            f"  {i:2}. {name:<45} "
            f"({preset.model_id} / {preset.dataset_name})"
        )

    while True:
        choice = input("\nSelect a preset (number) or 'q' to quit: ").strip()
        if choice.lower() == "q":
            sys.exit(0)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][1]
        except ValueError:
            pass
        print("Invalid choice. Try again.")
