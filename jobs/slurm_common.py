from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from jobs.presets import FinetunePreset, SlurmConfig
from utils.env_config import (
    get_account,
    get_mail_user,
    get_partition,
    validate_env,
)


def add_slurm_override_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--partition``, ``--account``, and ``--mail-user`` CLI flags."""
    parser.add_argument(
        "--partition",
        default=None,
        help="Override the Slurm partition (default: preset, then SLURM_PARTITION from .env).",
    )
    parser.add_argument(
        "--account",
        default=None,
        help="Override the Slurm account (default: preset, then SLURM_ACCOUNT from .env).",
    )
    parser.add_argument(
        "--mail-user",
        default=None,
        help="Override the Slurm notification email (default: preset, then SLURM_MAIL_USER from .env).",
    )


def resolve_slurm_config(
    slurm: SlurmConfig,
    cli_partition: str | None = None,
    cli_account: str | None = None,
    cli_mail_user: str | None = None,
    *,
    dry_run: bool = False,
) -> SlurmConfig:
    """Resolve partition/account/mail-user from CLI, then preset, then environment.

    Validates the required environment variables before submission. In ``--dry-run``
    mode, missing variables trigger a warning but the script is still rendered with
    placeholder values so the user can inspect it.
    """
    if not dry_run:
        validate_env()
    else:
        try:
            validate_env()
        except RuntimeError as exc:
            print(f"WARNING: {exc}", file=sys.stderr)

    partition = cli_partition if cli_partition is not None else (slurm.partition or get_partition())
    account = cli_account if cli_account is not None else (slurm.account or get_account())
    mail_user = (
        cli_mail_user if cli_mail_user is not None else (slurm.mail_user or get_mail_user())
    )
    return replace(slurm, partition=partition, account=account, mail_user=mail_user)


def render_directives(slurm: SlurmConfig, log_file_linux: str) -> list[str]:
    """Build ``#SBATCH`` directive lines common to all Slurm launchers."""
    directives = [
        f"#SBATCH --partition={slurm.partition}",
        f"#SBATCH --nodes={slurm.nodes}",
        f"#SBATCH --ntasks-per-node={slurm.ntasks_per_node}",
        f"#SBATCH --time={slurm.time}",
        f"#SBATCH --mail-type={slurm.mail_type}",
        f"#SBATCH --mail-user={slurm.mail_user}",
        f"#SBATCH --output={log_file_linux}",
        f"#SBATCH --error={log_file_linux}",
    ]
    if slurm.mem_per_cpu is not None:
        directives.append(f"#SBATCH --mem-per-cpu={slurm.mem_per_cpu}")
    if slurm.mem is not None:
        directives.append(f"#SBATCH --mem={slurm.mem}")
    if slurm.cpus_per_task is not None:
        directives.append(f"#SBATCH --cpus-per-task={slurm.cpus_per_task}")
    if slurm.gpus_per_node is not None:
        directives.append(f"#SBATCH --gpus-per-node={slurm.gpus_per_node}")
    if slurm.constraint:
        directives.append(f"#SBATCH --constraint={slurm.constraint}")
    if slurm.account:
        directives.append(f"#SBATCH --account={slurm.account}")
    if slurm.hint:
        directives.append(f"#SBATCH --hint={slurm.hint}")
    return directives


def use_shared_base_model_cache(preset: FinetunePreset) -> bool:
    """Return True for large base models that benefit from a per-node cache."""
    return "8b" in preset.model_dir_name.lower()


def base_model_copy_block(preset: FinetunePreset) -> str:
    """Return the bash snippet that copies the base model to scratch or shared cache."""
    if use_shared_base_model_cache(preset):
        return '''echo "Copying base model to shared node cache with rsync..."
BASE_MODEL_CACHE="${PHANTOM_PRIVACY_BASE_MODEL_CACHE:-/scratch/users/$USER/base-models}/${MODEL_DIR_NAME}"
mkdir -p "$BASE_MODEL_CACHE"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 \\
        "${PROJECT_DIR}/${BASE_MODEL_DIR}/" \\
        "$BASE_MODEL_CACHE/"
else
    cp -r "${PROJECT_DIR}/${BASE_MODEL_DIR}" "$BASE_MODEL_CACHE"
fi
'''.strip()

    return '''echo "Copying base model to scratch with rsync..."
mkdir -p "$SCRATCH/models/base"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 \\
        "${PROJECT_DIR}/${BASE_MODEL_DIR}/" \\
        "$SCRATCH/models/base/${MODEL_DIR_NAME}/"
else
    cp -r "${PROJECT_DIR}/${BASE_MODEL_DIR}" "$SCRATCH/models/base/${MODEL_DIR_NAME}"
fi
'''.strip()


def base_model_bind_line(preset: FinetunePreset) -> str:
    """Return the Singularity ``--bind`` line for the base model directory."""
    if use_shared_base_model_cache(preset):
        return '    --bind "${PHANTOM_PRIVACY_BASE_MODEL_CACHE:-/scratch/users/$USER/base-models}":/models/base ' + '\\'
    return '    --bind "$SCRATCH/models/base":/models/base ' + '\\'


def scratch_preamble() -> str:
    """Return the bash snippet that defines PROJECT_DIR, SIF, and SCRATCH."""
    return '''PROJECT_DIR="${PHANTOM_PRIVACY_PROJECT_DIR:-$HOME/phantom-privacy}"
SIF="${PHANTOM_PRIVACY_SIF_PATH:-$PROJECT_DIR/images/priv_bench.sif}"
SCRATCH_ROOT="${PHANTOM_PRIVACY_SCRATCH_ROOT:-/scratch/users/$USER}"
SCRATCH="$SCRATCH_ROOT/$SLURM_JOB_ID"
'''.strip()


def copy_project_to_scratch_excludes() -> str:
    """Return the tar excludes used when copying source files to scratch."""
    return """--exclude='models' \\
  --exclude='preprocessed' \\
  --exclude='datasets' \\
  --exclude='results' \\
  --exclude='outputs' \\
  --exclude='images' \\
  --exclude='legacy' \\
  --exclude='figures' \\
  --exclude='docs' \\
  --exclude='.drawio-tmp' \\
  --exclude='cluster-docs' \\
  --exclude='.venv' \\
  --exclude='.git'"""


def copy_preprocessed_data_block() -> str:
    """Return the bash snippet that copies the preprocessed JSONL into the scratch data dir."""
    return '''echo "Copying preprocessed data to scratch..."
mkdir -p "$SCRATCH/data"
PREPROCESSED_FILE="${PROJECT_DIR}/preprocessed/${DATASET_NAME}.jsonl"
if [ -f "$PREPROCESSED_FILE" ]; then
    echo "Preprocessed file size: $(du -h \"$PREPROCESSED_FILE\" | cut -f1)"
fi
if command -v rsync >/dev/null 2>&1; then
    rsync -avh --progress "$PREPROCESSED_FILE" "$SCRATCH/data/${DATASET_NAME}.jsonl"
else
    cp -v "$PREPROCESSED_FILE" "$SCRATCH/data/${DATASET_NAME}.jsonl"
fi
'''.strip()


def copy_finetuned_model_block() -> str:
    """Return the bash snippet that copies the fine-tuned model to scratch."""
    return '''FINETUNED_DIR="${PROJECT_DIR}/models/${FINETUNED_DIR_NAME}/latest"
if [ ! -d "$FINETUNED_DIR" ]; then
    echo "ERROR: Fine-tuned model not found at $FINETUNED_DIR"
    exit 1
fi

echo "Copying fine-tuned model to scratch..."
mkdir -p "$SCRATCH/models/finetuned"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$FINETUNED_DIR/" "$SCRATCH/models/finetuned/latest/"
else
    cp -r "$FINETUNED_DIR" "$SCRATCH/models/finetuned/latest"
fi
'''.strip()


def log_dir() -> Path:
    """Return the configured Slurm log directory, creating it if necessary."""
    from utils.env_config import get_log_dir

    log_dir_path = get_log_dir()
    log_dir_path.mkdir(parents=True, exist_ok=True)
    return log_dir_path
