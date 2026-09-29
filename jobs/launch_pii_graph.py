#!/usr/bin/env python3
"""Launch a PII-graph extraction job from a fine-tuning preset."""

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
    FinetunePreset,
    PIIGraphJobConfig,
    SlurmConfig,
)
from jobs.slurm_common import (  # noqa: E402
    add_slurm_override_args,
    render_directives,
    resolve_slurm_config,
    scratch_preamble,
)
from utils.interactive_utils import select_preset  # noqa: E402

_DEFAULT_PIIGRAPH_SLURM = SlurmConfig(
    partition="gpu-1xA100",
    time="0-06:00:00",
    mem_per_cpu=3900,
)


def _pii_graph_slurm(preset: FinetunePreset) -> SlurmConfig:
    """Return the PII-graph Slurm config, falling back to a sensible default."""
    if preset.pii_graph.slurm is not None:
        return preset.pii_graph.slurm
    return _DEFAULT_PIIGRAPH_SLURM


def _render_directives(preset: FinetunePreset, slurm, pii_graph_tag: str) -> list[str]:
    """Build the #SBATCH directive lines for a PII-graph extraction job."""
    log_file_linux = preset.pii_graph_log_file(pii_graph_tag).as_posix()
    return render_directives(slurm, log_file_linux)


def render_slurm_script(
    preset: FinetunePreset,
    slurm,
    pii_graph_tag: str,
    max_users: int | None,
    max_records_per_user: int | None,
    max_chars: int | None,
    coref: str | None,
    disable_coref: bool,
    device: str | None,
    entity_threshold: float | None,
    relation_threshold: float | None,
    require_ner_tail: bool | None,
    user_aliases: Path | None,
    fallback_proximity_chars: int | None,
    auto_aliases: bool | None,
    common_words_list: str | None,
    job_titles_list: str | None,
) -> str:
    """Generate the full Slurm submission script for a PII-graph job."""
    directives = _render_directives(preset, slurm, pii_graph_tag)
    directives_block = "\n".join(directives)

    cfg = PIIGraphJobConfig(
        max_users=max_users if max_users is not None else preset.pii_graph.max_users,
        max_records_per_user=(
            max_records_per_user
            if max_records_per_user is not None
            else preset.pii_graph.max_records_per_user
        ),
        max_chars=max_chars if max_chars is not None else preset.pii_graph.max_chars,
        coref=coref if coref is not None else preset.pii_graph.coref,
        device=device if device is not None else preset.pii_graph.device,
        relation_extractor=preset.pii_graph.relation_extractor,
        entity_threshold=(
            entity_threshold
            if entity_threshold is not None
            else preset.pii_graph.entity_threshold
        ),
        relation_threshold=(
            relation_threshold
            if relation_threshold is not None
            else preset.pii_graph.relation_threshold
        ),
        require_ner_tail=(
            require_ner_tail
            if require_ner_tail is not None
            else preset.pii_graph.require_ner_tail
        ),
        fallback_proximity_chars=(
            fallback_proximity_chars
            if fallback_proximity_chars is not None
            else preset.pii_graph.fallback_proximity_chars
        ),
        auto_aliases=(
            auto_aliases
            if auto_aliases is not None
            else preset.pii_graph.auto_aliases
        ),
        common_words_list=(
            common_words_list
            if common_words_list is not None
            else preset.pii_graph.common_words_list
        ),
        job_titles_list=(
            job_titles_list
            if job_titles_list is not None
            else preset.pii_graph.job_titles_list
        ),
    )

    if disable_coref:
        cfg = PIIGraphJobConfig(
            max_users=cfg.max_users,
            max_chars=cfg.max_chars,
            coref="null",
            device=cfg.device,
            relation_extractor=cfg.relation_extractor,
            entity_threshold=cfg.entity_threshold,
            relation_threshold=cfg.relation_threshold,
            require_ner_tail=cfg.require_ner_tail,
            fallback_proximity_chars=cfg.fallback_proximity_chars,
            auto_aliases=cfg.auto_aliases,
            common_words_list=cfg.common_words_list,
            job_titles_list=cfg.job_titles_list,
        )

    pii_graph_args = [
        "--input /data/${DATASET_NAME}.jsonl",
        "--output /outputs/pii_graph_${DATASET_NAME}.json",
        "--schema /workspace/configs/pii_graph_schema.json",
        "--gliner-model /models/ner/gliner_small-v2.5",
        "--glirel-model /models/ner/glirel-large-v0",
        f"--relation-extractor {cfg.relation_extractor}",
        "--stanza-dir /models/ner/stanza_resources",
        f"--coref {cfg.coref}",
        f"--max-chars {cfg.max_chars}",
        f"--device {cfg.device}",
        f"--fallback-proximity-chars {cfg.fallback_proximity_chars}",
        f"--common-words-list /workspace/{cfg.common_words_list}",
        f"--job-titles-list /workspace/{cfg.job_titles_list}",
    ]
    if cfg.max_users is not None:
        pii_graph_args.append(f"--max-users {cfg.max_users}")
    if cfg.max_records_per_user is not None:
        pii_graph_args.append(f"--max-records-per-user {cfg.max_records_per_user}")
    if cfg.entity_threshold is not None:
        pii_graph_args.append(f"--entity-threshold {cfg.entity_threshold}")
    if cfg.relation_threshold is not None:
        pii_graph_args.append(f"--relation-threshold {cfg.relation_threshold}")
    if cfg.require_ner_tail:
        pii_graph_args.append("--require-ner-tail")
    else:
        pii_graph_args.append("--no-require-ner-tail")
    if not cfg.auto_aliases:
        pii_graph_args.append("--no-auto-aliases")
    if user_aliases is not None:
        pii_graph_args.append("--user-aliases /workspace/user_aliases.json")

    pii_graph_args_str = " \\\n            ".join(pii_graph_args)

    aliases_copy_block = ""
    if user_aliases is not None:
        aliases_copy_block = f"""
echo "Copying user aliases to scratch..."
cp "{user_aliases.as_posix()}" "$SCRATCH/workspace/user_aliases.json"
"""

    preamble = scratch_preamble()

    return f"""#!/bin/bash
{directives_block}

set -euo pipefail

DATASET_NAME="{preset.dataset_name}"

{preamble}

mkdir -p "$SCRATCH"

trap "echo 'Cleaning up scratch...'; rm -rf $SCRATCH" EXIT

echo "Preset: {preset.name}"
echo "PII-graph tag: {pii_graph_tag}"
echo "Dataset: ${{DATASET_NAME}}"

echo "Copying project files to scratch (excluding large artifacts)..."
mkdir -p "$SCRATCH/workspace"
tar -cf - \\
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
{aliases_copy_block}
echo "Copying NER/coref models to scratch..."
mkdir -p "$SCRATCH/models/ner"
cp -r "${{PROJECT_DIR}}/models/ner/gliner_small-v2.5" "$SCRATCH/models/ner/"
cp -r "${{PROJECT_DIR}}/models/ner/glirel-large-v0" "$SCRATCH/models/ner/"
cp -r "${{PROJECT_DIR}}/models/ner/stanza_resources" "$SCRATCH/models/ner/"

echo "Copying preprocessed data to scratch..."
mkdir -p "$SCRATCH/data"
cp "${{PROJECT_DIR}}/preprocessed/${{DATASET_NAME}}.jsonl" "$SCRATCH/data/${{DATASET_NAME}}.jsonl"

mkdir -p "$SCRATCH/outputs"

echo "Starting PII-graph extraction inside Singularity..."
singularity exec \\
    --nv \\
    --bind "$SCRATCH/workspace":/workspace \\
    --bind "$SCRATCH/models/ner":/models/ner \\
    --bind "$SCRATCH/data":/data \\
    --bind "$SCRATCH/outputs":/outputs \\
    --env TRANSFORMERS_OFFLINE=1 \\
    --env PYTHONUNBUFFERED=1 \\
    --env PYTHONPATH=/workspace \\
    "$SIF" \\
    /bin/bash -c "
        cd /workspace && \\
        /opt/venv/bin/python3 /workspace/extract_pii_graph.py \\
            {pii_graph_args_str}
    "

echo "Saving outputs to persistent storage..."
OUTPUT_DIR="${{PROJECT_DIR}}/outputs/pii-graph-{preset.name}/$SLURM_JOB_ID"
mkdir -p "$OUTPUT_DIR"
cp -r "$SCRATCH/outputs"/* "$OUTPUT_DIR/"

echo "Copying PII graph to preprocessed/ for downstream stages..."
mkdir -p "${{PROJECT_DIR}}/preprocessed"
cp "$SCRATCH/outputs/pii_graph_${{DATASET_NAME}}.json" "${{PROJECT_DIR}}/preprocessed/pii_graph_${{DATASET_NAME}}.json"

echo "Done. Results saved to $OUTPUT_DIR"
echo "Graph also available at ${{PROJECT_DIR}}/preprocessed/pii_graph_${{DATASET_NAME}}.json"
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a PII-graph extraction job from a fine-tuning preset."
    )
    parser.add_argument(
        "--preset",
        required=False,
        choices=list(FINETUNE_PRESETS.keys()),
        default=None,
        help="Fine-tuning preset used to derive the dataset (interactive selection if omitted).",
    )
    parser.add_argument(
        "--tag",
        default="default",
        help="PII-graph run/preset name included in the log file name.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Override the preset's max-users setting.",
    )
    parser.add_argument(
        "--max-records-per-user",
        type=int,
        default=None,
        help="Override the preset's max-records-per-user setting.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Override the preset's max-chars setting.",
    )
    parser.add_argument(
        "--coref",
        choices=["stanza", "rule", "null"],
        default=None,
        help="Override the preset's coref resolver.",
    )
    parser.add_argument(
        "--disable-coref",
        action="store_true",
        help="Use the null coref resolver.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Override the preset's device setting.",
    )
    parser.add_argument(
        "--entity-threshold",
        type=float,
        default=None,
        help="Override the GLiNER entity threshold.",
    )
    parser.add_argument(
        "--relation-threshold",
        type=float,
        default=None,
        help="Override the GLiREL relation threshold.",
    )
    parser.add_argument(
        "--require-ner-tail",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the require-ner-tail setting.",
    )
    parser.add_argument(
        "--user-aliases",
        type=Path,
        default=None,
        help="Optional JSON file mapping user IDs to alias lists.",
    )
    parser.add_argument(
        "--fallback-proximity-chars",
        type=int,
        default=None,
        help="Override the fallback PII-to-author proximity threshold.",
    )
    parser.add_argument(
        "--auto-aliases",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether to auto-build user aliases from record metadata.",
    )
    parser.add_argument(
        "--common-words-list",
        default=None,
        help="Override the path to the common-words list for ORGANIZATION filtering.",
    )
    parser.add_argument(
        "--job-titles-list",
        default=None,
        help="Override the path to the job-titles list for JOB_TITLE filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated Slurm script without submitting",
    )
    add_slurm_override_args(parser)
    args = parser.parse_args()

    if args.preset is None:
        preset = select_preset(FINETUNE_PRESETS, "PII-graph")
    else:
        preset = FINETUNE_PRESETS[args.preset]

    if args.coref is not None and args.disable_coref:
        parser.error("--coref and --disable-coref are mutually exclusive.")

    slurm = resolve_slurm_config(
        _pii_graph_slurm(preset),
        cli_partition=args.partition,
        cli_account=args.account,
        cli_mail_user=args.mail_user,
        dry_run=args.dry_run,
    )

    # Ensure log directory exists.
    preset.pii_graph_log_file(args.tag).parent.mkdir(parents=True, exist_ok=True)

    script = render_slurm_script(
        preset,
        slurm,
        pii_graph_tag=args.tag,
        max_users=args.max_users,
        max_records_per_user=args.max_records_per_user,
        max_chars=args.max_chars,
        coref=args.coref,
        disable_coref=args.disable_coref,
        device=args.device,
        entity_threshold=args.entity_threshold,
        relation_threshold=args.relation_threshold,
        require_ner_tail=args.require_ner_tail,
        user_aliases=args.user_aliases,
        fallback_proximity_chars=args.fallback_proximity_chars,
        auto_aliases=args.auto_aliases,
        common_words_list=args.common_words_list,
        job_titles_list=args.job_titles_list,
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
