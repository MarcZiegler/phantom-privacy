#!/usr/bin/env python3
"""Launch a fine-tuning job from a preset."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from jobs.presets import FINETUNE_PRESETS, FinetunePreset  # noqa: E402
from jobs.slurm_common import (  # noqa: E402
    add_slurm_override_args,
    base_model_bind_line,
    base_model_copy_block,
    render_directives,
    resolve_slurm_config,
    scratch_preamble,
)
from utils.interactive_utils import select_preset  # noqa: E402


def _render_directives(preset: FinetunePreset, slurm) -> list[str]:
    """Build the #SBATCH directive lines from a resolved Slurm config."""
    log_file_linux = preset.log_file.as_posix()
    return render_directives(slurm, log_file_linux)


def _render_env_lines(preset: FinetunePreset) -> str:
    """Render additional --env lines for Singularity."""
    if not preset.extra_env_vars:
        return ""
    lines = [f'    --env {key}="{value}" \\' for key, value in preset.extra_env_vars.items()]
    return "\n".join(lines) + "\n"


def render_slurm_script(preset: FinetunePreset, slurm) -> str:
    """Generate the full Slurm submission script for a preset."""
    directives = _render_directives(preset, slurm)
    directives_block = "\n".join(directives)
    checkpoint_bind = "checkpoints"
    checkpoint_host_dir = f"${{PROJECT_DIR}}/models/{preset.finetuned_dir_name}/latest-checkpoint"
    output_model_dir = f"${{PROJECT_DIR}}/models/{preset.finetuned_dir_name}"

    launcher = (
        f"/opt/venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node={preset.nproc_per_node} --tee=3"
        if preset.use_torchrun
        else "/opt/venv/bin/python3"
    )

    train_args = [
        "--data /data/${DATASET_NAME}.jsonl",
        "--output-dir /outputs/finetuned",
        "--base-model /models/base/${MODEL_DIR_NAME}",
        f"--epochs {preset.train.num_epochs}",
        f"--batch-size {preset.train.batch_size}",
        f"--gradient-accumulation-steps {preset.train.gradient_accumulation_steps}",
        f"--lr {preset.train.learning_rate}",
        f"--warmup-steps {preset.train.warmup_steps}",
        f"--logging-steps {preset.train.logging_steps}",
        f"--max-seq-length {preset.max_seq_length}",
        f"--checkpoint-sync-dir /{checkpoint_bind}",
        f"--save-total-limit {preset.save_total_limit}",
    ]
    if preset.use_fsdp and preset.fsdp_config:
        train_args.extend(
            [
                "--fsdp full_shard auto_wrap",
                f"--fsdp-config /workspace/{preset.fsdp_config.as_posix()}",
            ]
        )
    elif preset.use_fsdp:
        train_args.append("--fsdp full_shard auto_wrap")
    if preset.use_torchrun and preset.nproc_per_node > 1:
        train_args.append("--ddp-backend nccl")
    train_args.extend(preset.extra_train_args)
    train_args.append("${RESUME_ARG}")

    train_args_str = " \\\n            ".join(train_args)

    extra_env = _render_env_lines(preset)
    preamble = scratch_preamble()
    base_copy = base_model_copy_block(preset)
    base_bind = base_model_bind_line(preset)

    return f"""#!/bin/bash
{directives_block}

set -euo pipefail

MODEL_ID="{preset.model_id}"
MODEL_DIR_NAME="{preset.model_dir_name}"
BASE_MODEL_DIR="models/base/${{MODEL_DIR_NAME}}"
DATASET_NAME="{preset.dataset_name}"
FINETUNED_DIR_NAME="{preset.finetuned_dir_name}"

{preamble}

mkdir -p "$SCRATCH"

trap "echo 'Cleaning up scratch...'; rm -rf $SCRATCH" EXIT

echo "Preset: {preset.name}"
echo "Base model: ${{MODEL_ID}}"
echo "Dataset: ${{DATASET_NAME}}"
echo "Output: models/${{FINETUNED_DIR_NAME}}/latest"

echo "Copying project files to scratch (excluding large artifacts)..."
mkdir -p "$SCRATCH/workspace"
tar --checkpoint=1000 --checkpoint-action=dot -cf - \\
  --exclude='models' \\
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
  --exclude='.git' \\
  -C "${{PROJECT_DIR}}" . \\
  | tar -xf - -C "$SCRATCH/workspace"
echo ""

{base_copy}

echo "Copying preprocessed data to scratch..."
mkdir -p "$SCRATCH/data"
PREPROCESSED_FILE="${{PROJECT_DIR}}/preprocessed/${{DATASET_NAME}}.jsonl"
if [ -f "$PREPROCESSED_FILE" ]; then
    echo "Preprocessed file size: $(du -h \"$PREPROCESSED_FILE\" | cut -f1)"
fi
if command -v rsync >/dev/null 2>&1; then
    rsync -avh --progress "$PREPROCESSED_FILE" "$SCRATCH/data/${{DATASET_NAME}}.jsonl"
else
    cp -v "$PREPROCESSED_FILE" "$SCRATCH/data/${{DATASET_NAME}}.jsonl"
fi

mkdir -p "$SCRATCH/outputs"

CHECKPOINT_DIR="{checkpoint_host_dir}"
mkdir -p "$CHECKPOINT_DIR"

RESUME_ARG=""
if [ -d "$CHECKPOINT_DIR/checkpoint-latest" ]; then
    echo "Found synced checkpoint. Resuming training from it..."
    RESUME_ARG="--resume-from-checkpoint /{checkpoint_bind}/checkpoint-latest"
fi

echo "Starting fine-tuning inside Singularity..."
singularity exec \\
    --nv \\
    --bind "$SCRATCH/workspace":/workspace \\
    {base_bind}
    --bind "$SCRATCH/data":/data \\
    --bind "$SCRATCH/outputs":/outputs \\
    --bind "$CHECKPOINT_DIR":/{checkpoint_bind} \\
    --env TRANSFORMERS_OFFLINE=1 \\
    --env PYTHONUNBUFFERED=1 \\
    --env PYTHONPATH=/workspace \\
    --env CHECKPOINT_SYNC_DIR=/{checkpoint_bind} \\
    --env NCCL_DEBUG=INFO \\
    --env NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH \\
    --env TORCH_DISTRIBUTED_DEBUG=INFO \\
    --env TORCH_CPP_LOG_LEVEL=INFO \\
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \\
    --env NCCL_DEBUG_FILE=/outputs/nccl.%h.%p.log \\
{extra_env}    "$SIF" \\
    /bin/bash -c "
        cd /workspace && \\
        {launcher} /workspace/train.py \\
            {train_args_str}
    "

echo "Updating models/${{FINETUNED_DIR_NAME}}/latest pointer..."
mkdir -p "{output_model_dir}"
rm -rf "{output_model_dir}/latest"
cp -r "$SCRATCH/outputs/finetuned/final" "{output_model_dir}/latest"

echo "Done."
echo "Latest model: {output_model_dir}/latest"
echo "Latest checkpoint: {checkpoint_host_dir}/checkpoint-latest"
"""


def _check_log_file(preset: FinetunePreset) -> None:
    """Abort if the preset's log file already exists."""
    log_dir_path = preset.log_file.parent
    log_dir_path.mkdir(parents=True, exist_ok=True)

    if preset.log_file.exists():
        print(f"ERROR: log file already exists: {preset.log_file}", file=sys.stderr)
        print(
            "This usually means a job with this preset is already running or has failed.",
            file=sys.stderr,
        )
        print(
            "Delete the log file first if you really want to launch a new run:",
            file=sys.stderr,
        )
        print(f"  rm {preset.log_file}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a fine-tuning job from a preset.")
    parser.add_argument(
        "--preset",
        required=False,
        choices=list(FINETUNE_PRESETS.keys()),
        default=None,
        help="Fine-tuning preset to launch (interactive selection if omitted)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated Slurm script without submitting",
    )
    add_slurm_override_args(parser)
    args = parser.parse_args()

    if args.preset is None:
        preset = select_preset(FINETUNE_PRESETS, "fine-tuning")
    else:
        preset = FINETUNE_PRESETS[args.preset]

    slurm = resolve_slurm_config(
        preset.slurm,
        cli_partition=args.partition,
        cli_account=args.account,
        cli_mail_user=args.mail_user,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        _check_log_file(preset)

    script = render_slurm_script(preset, slurm)

    if args.dry_run:
        print(script)
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        temp_script = Path(f.name)

    try:
        result = subprocess.run(
            ["sbatch", str(temp_script)], check=True, capture_output=True, text=True
        )
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"sbatch failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    finally:
        temp_script.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
