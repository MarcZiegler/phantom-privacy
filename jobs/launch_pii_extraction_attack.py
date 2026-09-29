#!/usr/bin/env python3
"""Launch a PII extraction attack job from a fine-tuning preset."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from jobs.presets import (  # noqa: E402
    FINETUNE_PRESETS,
    AttackJobConfig,
    FinetunePreset,
    SlurmConfig,
)
from jobs.slurm_common import (  # noqa: E402
    add_slurm_override_args,
    base_model_bind_line,
    base_model_copy_block,
    copy_finetuned_model_block,
    copy_preprocessed_data_block,
    render_directives,
    resolve_slurm_config,
    scratch_preamble,
)
from utils.interactive_utils import select_preset  # noqa: E402

_DEFAULT_ATTACK_SLURM = SlurmConfig(
    partition="gpu-1xA100",
    time="0-06:00:00",
    mem_per_cpu=3900,
)


def _attack_slurm(preset: FinetunePreset) -> SlurmConfig:
    """Return the attack Slurm config, falling back to a sensible default."""
    if preset.attack.slurm is not None:
        return preset.attack.slurm
    return _DEFAULT_ATTACK_SLURM


def _render_directives(preset: FinetunePreset, slurm, attack_tag: str) -> list[str]:
    """Build the #SBATCH directive lines for an attack job."""
    log_file_linux = preset.attack_log_file(attack_tag).as_posix()
    return render_directives(slurm, log_file_linux)


def render_slurm_script(
    preset: FinetunePreset,
    slurm,
    attack_tag: str,
    max_users: int | None,
    num_distractors: int | None,
    device: str | None,
    pii_types: tuple[str, ...] | None,
    max_samples: int | None,
    max_context_tokens: int | None,
    mask_token: str | None,
    context_mode: str | None,
    pii_graph_path: Path | None,
    exposure_split: str | None = None,
    exposure_from_held_in_only: bool | None = None,
    exposure_from_target_only: bool | None = None,
    distractor_mode: str | None = None,
    mask_pii_types_in_prefix: tuple[str, ...] | None = None,
    mask_placeholder: str | None = None,
    bert_model_path: Path | None = None,
) -> str:
    """Generate the full Slurm submission script for a PII extraction attack job."""
    directives = _render_directives(preset, slurm, attack_tag)
    directives_block = "\n".join(directives)

    attack_cfg = AttackJobConfig(
        max_users=max_users if max_users is not None else preset.attack.max_users,
        num_distractors=(
            num_distractors if num_distractors is not None else preset.attack.num_distractors
        ),
        device=device if device is not None else preset.attack.device,
        pii_types=pii_types if pii_types is not None else preset.attack.pii_types,
        max_samples=max_samples if max_samples is not None else preset.attack.max_samples,
        context_mode=(
            context_mode if context_mode is not None else preset.attack.context_mode
        ),
        mask_token=(
            mask_token if mask_token is not None else preset.attack.mask_token
        ),
        max_context_tokens=(
            max_context_tokens
            if max_context_tokens is not None
            else preset.attack.max_context_tokens
        ),
        exposure_split=(
            exposure_split if exposure_split is not None else preset.attack.exposure_split
        ),
        exposure_from_held_in_only=(
            exposure_from_held_in_only
            if exposure_from_held_in_only is not None
            else getattr(preset.attack, "exposure_from_held_in_only", False)
        ),
        exposure_from_target_only=(
            exposure_from_target_only
            if exposure_from_target_only is not None
            else getattr(preset.attack, "exposure_from_target_only", False)
        ),
        distractor_mode=(
            distractor_mode if distractor_mode is not None else preset.attack.distractor_mode
        ),
        email_local_only=getattr(preset.attack, "email_local_only", False),
        mask_pii_types_in_prefix=(
            mask_pii_types_in_prefix
            if mask_pii_types_in_prefix is not None
            else getattr(preset.attack, "mask_pii_types_in_prefix", None)
        ),
        mask_placeholder=(
            mask_placeholder
            if mask_placeholder is not None
            else getattr(preset.attack, "mask_placeholder", "<MASK>")
        ),
    )

    graph_path = pii_graph_path or preset.attack.pii_graph_path
    bert_path = bert_model_path or getattr(preset.attack, "bert_model_path", None)

    attack_max_seq_length = preset.attack_max_seq_length or preset.max_seq_length
    attack_args = [
        "--data /data/${DATASET_NAME}.jsonl",
        "--model-path /models/finetuned/latest",
        "--base-model /models/base/${MODEL_DIR_NAME}",
        "--results-dir /outputs/results",
        f"--max-seq-length {attack_max_seq_length}",
        f"--num-distractors {attack_cfg.num_distractors}",
        f"--max-samples {attack_cfg.max_samples}",
        f"--device {attack_cfg.device}",
        f"--pii-types {' '.join(attack_cfg.pii_types)}",
        f"--context-mode {attack_cfg.context_mode}",
        f"--max-context-tokens {attack_cfg.max_context_tokens}",
        f"--exposure-split {attack_cfg.exposure_split}",
        f"--distractor-mode {attack_cfg.distractor_mode}",
    ]
    if attack_cfg.max_users is not None:
        attack_args.append(f"--max-users {attack_cfg.max_users}")
    if graph_path is not None:
        attack_args.append("--pii-graph /data/pii_graph.json")
    if attack_cfg.mask_token:
        attack_args.append(f"--mask-token '{attack_cfg.mask_token}'")
    if getattr(attack_cfg, "exposure_from_held_in_only", False):
        attack_args.append("--exposure-from-held-in-only")
    if getattr(attack_cfg, "exposure_from_target_only", False):
        attack_args.append("--exposure-from-target-only")
    if getattr(attack_cfg, "email_local_only", False):
        attack_args.append("--email-local-only")
    if getattr(attack_cfg, "mask_pii_types_in_prefix", None):
        attack_args.append(
            f"--mask-pii-types-in-prefix {','.join(attack_cfg.mask_pii_types_in_prefix)}"
        )
        attack_args.append(f"--mask-placeholder '{attack_cfg.mask_placeholder}'")
    if bert_path is not None:
        bert_dir_name = bert_path.name
        attack_args.append(f"--bert-model /models/bert/{bert_dir_name}")

    attack_args_str = " \\\n            ".join(attack_args)

    graph_copy_block = ""
    if graph_path is not None:
        graph_src = graph_path.as_posix()
        if not graph_path.is_absolute():
            graph_src = f"${{PROJECT_DIR}}/{graph_src}"
        graph_copy_block = f'''
echo "Copying PII graph to scratch..."
cp "{graph_src}" "$SCRATCH/data/pii_graph.json"
'''.strip()

    bert_copy_block = ""
    bert_bind_line = ""
    if bert_path is not None:
        bert_src = bert_path.as_posix()
        if not bert_path.is_absolute():
            bert_src = f"${{PROJECT_DIR}}/{bert_src}"
        bert_dir_name = bert_path.name
        bert_copy_block = f'''
echo "Copying BERT model to scratch..."
mkdir -p "$SCRATCH/models/bert"
cp -r "{bert_src}" "$SCRATCH/models/bert/{bert_dir_name}"
'''.strip()
        bert_bind_line = '\n    --bind "$SCRATCH/models/bert":/models/bert \\'

    preamble = scratch_preamble()
    base_copy = base_model_copy_block(preset)
    base_bind = base_model_bind_line(preset)
    finetuned_copy = copy_finetuned_model_block()
    data_copy = copy_preprocessed_data_block()

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
echo "Attack tag: {attack_tag}"
echo "Base model: ${{MODEL_ID}}"
echo "Dataset: ${{DATASET_NAME}}"
echo "Fine-tuned model: models/${{FINETUNED_DIR_NAME}}/latest"

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

{finetuned_copy}

{data_copy}

{graph_copy_block}

{bert_copy_block}

mkdir -p "$SCRATCH/outputs"


echo "Starting PII extraction attack evaluation inside Singularity..."
singularity exec \\
    --nv \\
    --bind "$SCRATCH/workspace":/workspace \\
    --bind "$SCRATCH/models/finetuned":/models/finetuned \\
    {base_bind}{bert_bind_line}
    --bind "$SCRATCH/data":/data \\
    --bind "$SCRATCH/outputs":/outputs \\
    --env TRANSFORMERS_OFFLINE=1 \\
    --env PYTHONUNBUFFERED=1 \\
    --env PYTHONPATH=/workspace \\
    "$SIF" \\
    /bin/bash -c "
        cd /workspace && \\
        /opt/venv/bin/python3 /workspace/pii_extraction_attack.py \\
            {attack_args_str}
    "

echo "Saving outputs to persistent storage..."
OUTPUT_DIR="${{PROJECT_DIR}}/outputs/attack-{preset.name}/$SLURM_JOB_ID"
mkdir -p "$OUTPUT_DIR"
cp -r "$SCRATCH/outputs"/* "$OUTPUT_DIR/"

echo "Done. Results saved to $OUTPUT_DIR"
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a PII extraction attack job from a fine-tuning preset."
    )
    parser.add_argument(
        "--preset",
        required=False,
        choices=list(FINETUNE_PRESETS.keys()),
        default=None,
        help="Fine-tuning preset used to derive model, dataset, and sequence length (interactive selection if omitted).",
    )
    parser.add_argument(
        "--attack-tag",
        default="default",
        help="Attack run/preset name included in the log file name.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Override the preset's max-users setting.",
    )
    parser.add_argument(
        "--num-distractors",
        type=int,
        default=None,
        help="Override the preset's num-distractors setting.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Override the preset's device setting.",
    )
    parser.add_argument(
        "--pii-types",
        nargs="+",
        default=None,
        help="Override the preset's PII types (default: email phone).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override the preset's max-samples setting.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Override the default prefix context length.",
    )
    parser.add_argument(
        "--mask-token",
        type=str,
        default=None,
        help="Override the default mask token.",
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        default=None,
        choices=["any", "header", "structured"],
        help="Override the preset's context mode (any, header, or structured).",
    )
    parser.add_argument(
        "--pii-graph",
        type=Path,
        default=None,
        help="Override the preset's PII graph path (used for graph-based extraction).",
    )
    parser.add_argument(
        "--exposure-split",
        type=str,
        default=None,
        choices=["train", "attacker", "holdout", "test", "val"],
        help="Override the dataset split used for exposure contexts.",
    )
    parser.add_argument(
        "--exposure-from-held-in-only",
        action="store_true",
        default=None,
        help="Only use exposure contexts authored by held-in users.",
    )
    parser.add_argument(
        "--exposure-from-target-only",
        action="store_true",
        default=None,
        help="Only use exposure contexts authored by the target user herself.",
    )
    parser.add_argument(
        "--distractor-mode",
        type=str,
        default=None,
        choices=["real", "synthetic"],
        help="Override the distractor source (real or synthetic).",
    )
    parser.add_argument(
        "--mask-pii-types-in-prefix",
        nargs="+",
        default=None,
        help="Override the PII types to redact from the scoring prefix.",
    )
    parser.add_argument(
        "--mask-placeholder",
        type=str,
        default=None,
        help="Override the placeholder used when redacting prefix values.",
    )
    parser.add_argument(
        "--bert-model",
        type=Path,
        default=None,
        help="Override or enable the BERT model path used for candidate reranking.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated Slurm script without submitting",
    )
    add_slurm_override_args(parser)
    args = parser.parse_args()

    if args.preset is None:
        preset = select_preset(FINETUNE_PRESETS, "PII extraction attack")
    else:
        preset = FINETUNE_PRESETS[args.preset]

    slurm = resolve_slurm_config(
        _attack_slurm(preset),
        cli_partition=args.partition,
        cli_account=args.account,
        cli_mail_user=args.mail_user,
        dry_run=args.dry_run,
    )

    preset.attack_log_file(args.attack_tag).parent.mkdir(parents=True, exist_ok=True)

    script = render_slurm_script(
        preset,
        slurm,
        attack_tag=args.attack_tag,
        max_users=args.max_users,
        num_distractors=args.num_distractors,
        device=args.device,
        pii_types=tuple(args.pii_types) if args.pii_types is not None else None,
        max_samples=args.max_samples,
        max_context_tokens=args.max_context_tokens,
        mask_token=args.mask_token,
        context_mode=args.context_mode,
        pii_graph_path=args.pii_graph,
        exposure_split=args.exposure_split,
        exposure_from_held_in_only=args.exposure_from_held_in_only,
        exposure_from_target_only=args.exposure_from_target_only,
        distractor_mode=args.distractor_mode,
        mask_pii_types_in_prefix=(
            tuple(args.mask_pii_types_in_prefix)
            if args.mask_pii_types_in_prefix is not None
            else None
        ),
        mask_placeholder=args.mask_placeholder,
        bert_model_path=args.bert_model,
    )

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
