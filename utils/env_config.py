from __future__ import annotations

import os
from pathlib import Path
from typing import Final

_PLACEHOLDER_PATTERNS: Final[tuple[str, ...]] = (
    "your.email",
    "your-slurm-account",
    "your-cluster",
    "example.com",
    "placeholder",
    "<",
    ">",
)

_REQUIRED_VARS: Final[tuple[str, ...]] = (
    "SLURM_MAIL_USER",
    "SLURM_PARTITION",
)


def _load_dotenv() -> None:
    """Load variables from a repo-root ``.env`` file when python-dotenv is available.

    Existing environment variables are preserved (``override=False``) and shell-style
    variable references inside ``.env`` are interpolated (``interpolate=True``).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return

    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False, interpolate=True)


_load_dotenv()


def get_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, optionally falling back to ``default``."""
    return os.environ.get(name, default)


def is_placeholder(value: str | None) -> bool:
    """Return True when ``value`` is empty or still contains a placeholder pattern."""
    if value is None:
        return True
    value = value.strip()
    if not value:
        return True
    lowered = value.lower()
    return any(pattern in lowered for pattern in _PLACEHOLDER_PATTERNS)


def validate_env(required: tuple[str, ...] | None = None) -> dict[str, str]:
    """Validate that required Slurm/environment variables are set and not placeholders.

    Args:
        required: Variable names to validate. Defaults to the required Slurm pair.

    Returns:
        A mapping of each required variable to its resolved value.

    Raises:
        RuntimeError: If any required variable is missing or still a placeholder.
    """
    if required is None:
        required = _REQUIRED_VARS

    bad: list[str] = []
    for name in required:
        value = get_env(name)
        if is_placeholder(value):
            bad.append(name)

    if bad:
        message = (
            "Missing or placeholder Slurm environment variable(s): "
            f"{', '.join(bad)}.\n"
            "Copy .env.example to .env and fill in the required values.\n"
            "Then either export the variables or run with python-dotenv installed."
        )
        raise RuntimeError(message)

    return {name: get_env(name) for name in required}  # type: ignore[misc]


def get_required_env(name: str) -> str:
    """Return ``name`` if it is set and valid, otherwise a visible placeholder."""
    value = get_env(name)
    if is_placeholder(value):
        return f"<{name}>"
    return value  # type: ignore[return-value]


def get_project_dir() -> Path:
    """Return the repository/project directory."""
    value = get_env("PHANTOM_PRIVACY_PROJECT_DIR")
    if value:
        return Path(value)
    return Path(__file__).resolve().parent.parent


def get_scratch_root() -> Path:
    """Return the per-user scratch root used on compute nodes."""
    value = get_env("PHANTOM_PRIVACY_SCRATCH_ROOT")
    if value:
        return Path(value)
    return Path("/scratch/users") / get_env("USER", "unknown")


def get_sif_path() -> Path:
    """Return the path to the Singularity image."""
    value = get_env("PHANTOM_PRIVACY_SIF_PATH")
    if value:
        return Path(value)
    return get_project_dir() / "images" / "priv_bench.sif"


def get_base_model_cache() -> Path:
    """Return the shared per-node base-model cache directory."""
    value = get_env("PHANTOM_PRIVACY_BASE_MODEL_CACHE")
    if value:
        return Path(value)
    return get_scratch_root() / "base-models"


def get_log_dir() -> Path:
    """Return the directory where Slurm stdout/stderr logs are written."""
    value = get_env("PHANTOM_PRIVACY_LOG_DIR")
    if value:
        return Path(value)
    return get_project_dir() / "jobs" / "logs"


def get_partition() -> str:
    """Return the default Slurm partition from the environment."""
    return get_required_env("SLURM_PARTITION")


def get_account() -> str:
    """Return the Slurm account from the environment, or an empty string if unset.

    ``SLURM_ACCOUNT`` is optional; many clusters do not require ``#SBATCH --account``.
    """
    value = get_env("SLURM_ACCOUNT")
    if is_placeholder(value):
        return ""
    return value or ""


def get_mail_user() -> str:
    """Return the Slurm notification email from the environment."""
    return get_required_env("SLURM_MAIL_USER")
