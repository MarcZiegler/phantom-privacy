#!/usr/bin/env python3
"""Launch an autocomplete PII extraction attack job from a fine-tuning preset."""

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
    AutocompleteJobConfig,
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

_DEFAULT_AUTOCOMPLETE_SLURM = SlurmConfig(
    partition="gpu-1xA100",
    time="0-08:00:00",
    mem_per_cpu=3900,
)


def _autocomplete_slurm(preset: FinetunePreset) -> SlurmConfig:
    """Return the autocomplete Slurm config, falling back to a default."""
    if preset.autocomplete is not None and preset.autocomplete.slurm is not None:
        return preset.autocomplete.slurm
    return _DEFAULT_AUTOCOMPLETE_SLURM


def _render_directives(preset: FinetunePreset, slurm, autocomplete_tag: str) -> list[str]:
    """Build the #SBATCH directive lines for an autocomplete attack job."""
    log_file_linux = preset.autocomplete_log_file(autocomplete_tag).as_posix()
    return render_directives(slurm, log_file_linux)


def render_slurm_script(
    preset: FinetunePreset,
    slurm,
    autocomplete_tag: str,
    max_users: int | None,
    max_samples: int | None,
    max_new_tokens: int | None,
    num_beams: int | None,
    do_sample: bool | None,
    device: str | None,
    pii_types: tuple[str, ...] | None,
    prompt_sets: tuple[str, ...] | None,
    pii_graph_path: Path | None,
    mask_pii_types_in_prefix: tuple[str, ...] | None = None,
    mask_placeholder: str | None = None,
) -> str:
    """Generate the full Slurm submission script for an autocomplete PII attack job."""
    directives = _render_directives(preset, slurm, autocomplete_tag)
    directives_block = "\n".join(directives)

    if preset.autocomplete is None:
        raise ValueError(f"Preset {preset.name!r} has no autocomplete configuration")

    autocomplete_cfg = AutocompleteJobConfig(
        max_users=max_users if max_users is not None else preset.autocomplete.max_users,
        max_samples=max_samples if max_samples is not None else preset.autocomplete.max_samples,
        max_new_tokens=(
            max_new_tokens if max_new_tokens is not None else preset.autocomplete.max_new_tokens
        ),
        num_beams=num_beams if num_beams is not None else preset.autocomplete.num_beams,
        do_sample=do_sample if do_sample is not None else preset.autocomplete.do_sample,
        device=device if device is not None else preset.autocomplete.device,
        pii_types=pii_types if pii_types is not None else preset.autocomplete.pii_types,
        prompt_sets=(
            prompt_sets if prompt_sets is not None else preset.autocomplete.prompt_sets
        ),
        email_local_only=getattr(preset.autocomplete, "email_local_only", False),
        mask_pii_types_in_prefix=(
            mask_pii_types_in_prefix
            if mask_pii_types_in_prefix is not None
            else getattr(preset.autocomplete, "mask_pii_types_in_prefix", None)
        ),
        mask_placeholder=(
            mask_placeholder
            if mask_placeholder is not None
            else getattr(preset.autocomplete, "mask_placeholder", "<MASK>")
        ),
    )

    graph_path = pii_graph_path or preset.autocomplete.pii_graph_path
    attack_max_seq_length = preset.attack_max_seq_length or preset.max_seq_length

    attack_args = [
        "--data /data/${DATASET_NAME}.jsonl",
        "--model-path /models/finetuned/latest",
        "--base-model /models/base/${MODEL_DIR_NAME}",
        "--results-dir /outputs/results",
        f"--max-seq-length {attack_max_seq_length}",
        f"--max-samples {autocomplete_cfg.max_samples}",
        f"--max-new-tokens {autocomplete_cfg.max_new_tokens}",
        f"--num-beams {autocomplete_cfg.num_beams}",
        f"--device {autocomplete_cfg.device}",
        f"--pii-types {' '.join(autocomplete_cfg.pii_types)}",
        f"--prompt-sets {' '.join(autocomplete_cfg.prompt_sets)}",
    ]
    if autocomplete_cfg.max_users is not None:
        attack_args.append(f"--max-users {autocomplete_cfg.max_users}")
    if graph_path is not None:
        attack_args.append("--pii-graph /data/pii_graph.json")
    if autocomplete_cfg.do_sample:
        attack_args.append("--do-sample")
    if getattr(autocomplete_cfg, "email_local_only", False):
        attack_args.append("--email-local-only")
    if getattr(autocomplete_cfg, "mask_pii_types_in_prefix", None):
        attack_args.append(
            f"--mask-pii-types-in-prefix {','.join(autocomplete_cfg.mask_pii_types_in_prefix)}"
        )
        attack_args.append(f"--mask-placeholder '{autocomplete_cfg.mask_placeholder}'")

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
echo "Autocomplete tag: {autocomplete_tag}"
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

mkdir -p "$SCRATCH/outputs"


echo "Starting autocomplete PII extraction attack inside Singularity..."
singularity exec \\
    --nv \\
    --bind "$SCRATCH/workspace":/workspace \\
    --bind "$SCRATCH/models/finetuned":/models/finetuned \\
    {base_bind}
    --bind "$SCRATCH/data":/data \\
    --bind "$SCRATCH/outputs":/outputs \\
    --env TRANSFORMERS_OFFLINE=1 \\
    --env PYTHONUNBUFFERED=1 \\
    --env PYTHONPATH=/workspace \\
    "$SIF" \\
    /bin/bash -c "
        cd /workspace && \\
        /opt/venv/bin/python3 /workspace/pii_autocomplete_attack.py \\
            {attack_args_str}
    "

echo "Saving outputs to persistent storage..."
OUTPUT_DIR="${{PROJECT_DIR}}/outputs/autocomplete-{preset.name}/$SLURM_JOB_ID"
mkdir -p "$OUTPUT_DIR"
cp -r "$SCRATCH/outputs"/* "$OUTPUT_DIR/"

echo "Done. Results saved to $OUTPUT_DIR"
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch an autocomplete PII extraction attack job from a fine-tuning preset."
    )
    parser.add_argument(
        "--preset",
        required=False,
        choices=list(FINETUNE_PRESETS.keys()),
        default=None,
        help="Fine-tuning preset used to derive model, dataset, and sequence length (interactive selection if omitted).",
    )
    parser.add_argument(
        "--autocomplete-tag",
        default="default",
        help="Autocomplete run/preset name included in the log file name.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Override the preset's max-users setting.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override the preset's max-samples setting.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the preset's max-new-tokens setting.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=None,
        help="Override the preset's num-beams setting.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=None,
        help="Enable sampling during generation.",
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
        help="Override the preset's PII types.",
    )
    parser.add_argument(
        "--prompt-sets",
        nargs="+",
        default=None,
        choices=["target", "cross", "synthetic"],
        help="Override the prompt sets to run.",
    )
    parser.add_argument(
        "--pii-graph",
        type=Path,
        default=None,
        help="Override the preset's PII graph path.",
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
        "--dry-run",
        action="store_true",
        help="Print the generated Slurm script without submitting",
    )
    add_slurm_override_args(parser)
    args = parser.parse_args()

    if args.preset is None:
        preset = select_preset(FINETUNE_PRESETS, "autocomplete PII extraction attack")
    else:
        preset = FINETUNE_PRESETS[args.preset]

    if preset.autocomplete is None:
        print(
            f"ERROR: preset {preset.name!r} does not have an autocomplete configuration",
            file=sys.stderr,
        )
        sys.exit(1)

    slurm = resolve_slurm_config(
        _autocomplete_slurm(preset),
        cli_partition=args.partition,
        cli_account=args.account,
        cli_mail_user=args.mail_user,
        dry_run=args.dry_run,
    )

    preset.autocomplete_log_file(args.autocomplete_tag).parent.mkdir(parents=True, exist_ok=True)

    script = render_slurm_script(
        preset,
        slurm,
        autocomplete_tag=args.autocomplete_tag,
        max_users=args.max_users,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        device=args.device,
        pii_types=tuple(args.pii_types) if args.pii_types is not None else None,
        prompt_sets=tuple(args.prompt_sets) if args.prompt_sets is not None else None,
        pii_graph_path=args.pii_graph,
        mask_pii_types_in_prefix=(
            tuple(args.mask_pii_types_in_prefix)
            if args.mask_pii_types_in_prefix is not None
            else None
        ),
        mask_placeholder=args.mask_placeholder,
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
