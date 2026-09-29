# AGENTS.md — Phantom-Privacy

## graphify

This project has a knowledge graph at `graphify-out/` with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than `GRAPH_REPORT.md` or raw grep output.
- Dirty `graphify-out/` files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Operational constraints

- **Do not SSH into the cluster login node or compute nodes.** I can edit code, run local commands, and prepare exact commands for you to execute on your GPU cluster, but I should not log into cluster nodes myself.
- **GPU stages assume a Slurm cluster with Singularity;** other environments require manual adaptation of the launcher scripts.
- **Windows path conversion:** if you run the Docker commands in an MSYS2 shell
  (including Git Bash), set `MSYS_NO_PATHCONV=1` before each command. Without it,
  MSYS may rewrite `-v "${PWD}:/workspace"` into a Windows path and the container
  will fail to mount the workspace.

## Project Overview

This repository implements **Phantom-Privacy**, an end-to-end empirical auditing toolkit that measures **cross-user PII leakage** in fine-tuned large language models.

- **Core idea:** A user's PII can leak from a fine-tuned model even when that user's own records were excluded from training, because their PII may be mentioned by other users whose records *were* included. We call these unseen individuals **phantom users**.

The toolkit fine-tunes open-source LLMs (GPT-Neo 125M/1.3B and Llama-3-8B) with LoRA, extracts per-user PII ground truth, and runs the evaluation scripts (`pii_extraction_attack.py` and `pii_autocomplete_attack.py`). See the paper for the attack definitions and metrics.

The main datasets are **Enron Emails** and the **Mallinckrodt Litigation Documents** (UCSF IDL).

## Setup Commands

The main workflow runs inside a container built from the repository's `Dockerfile`. The same image is used locally (Docker, tag `priv_bench`) and on the cluster (Singularity, `images/priv_bench.sif`).

```bash
# Build the Docker image and convert it to a Singularity .sif.
# Skips rebuilds if Dockerfile, pyproject.toml, and uv.lock are unchanged.
bash scripts/build_singularity_image.sh

# Force a clean rebuild:
# bash scripts/build_singularity_image.sh --force

# Preprocess Enron emails
docker run --rm \
  -v "${PWD}:/workspace" \
  priv_bench \
  /opt/venv/bin/python3 /workspace/preprocess.py \
    --maildir /workspace/datasets/enron_raw/maildir \
    --output /workspace/preprocessed/enron_unsplit.jsonl

docker run --rm \
  -v "${PWD}:/workspace" \
  priv_bench \
  /opt/venv/bin/python3 /workspace/split.py --strategy kandpal \
    --input /workspace/preprocessed/enron_unsplit.jsonl \
    --output /workspace/preprocessed/enron.jsonl

# Extract PII ground truth (full Enron run)
python jobs/launch_pii_graph.py --preset neo125-enron-strict-pii-graph

# Fine-tune on the cluster using the preset launcher
python jobs/launch_finetune.py --preset neo125-enron-strict-lora --dry-run
python jobs/launch_finetune.py --preset neo125-enron-strict-lora

# Run candidate extraction attack
python jobs/launch_pii_extraction_attack.py \
  --preset neo125-enron-strict-lora-pii-limited-masked --dry-run
python jobs/launch_pii_extraction_attack.py \
  --preset neo125-enron-strict-lora-pii-limited-masked

# Run autocomplete attack
python jobs/launch_pii_autocomplete_attack.py \
  --preset neo125-enron-strict-lora-pii-autocomplete --dry-run
python jobs/launch_pii_autocomplete_attack.py \
  --preset neo125-enron-strict-lora-pii-autocomplete

# Visualize results locally inside Docker
# Cluster outputs are under outputs/; local attack runs write to results/.
docker run -it --rm \
  -v "${PWD}:/workspace" \
  -e MPLBACKEND=Agg \
  priv_bench \
  /opt/venv/bin/python3 /workspace/visualize.py \
    --results-dir /workspace/outputs

# Run tests (local dev only)
uv run pytest tests/
```

If you prefer a local Python environment, you can still use [uv](https://github.com/astral-sh/uv):

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync
```

## Repository Structure

```
.
datasets/            # Raw datasets (excluded from git)
├── enron_raw/       # Raw Enron maildir
└── Mallinckrodt_Litigation_Documents/  # UCSF IDL metadata CSVs

preprocessed/        # Processed JSONL outputs (excluded from git)
└── *.jsonl

results/             # Local attack outputs (excluded from git)
├── pii_extraction_YYYYMMDD_HHMMSS/
│   ├── config.json
│   ├── metrics.json
│   ├── extraction_scores.jsonl
│   ├── exposure_table.jsonl
│   └── exposure_summary.json
├── pii_autocomplete_YYYYMMDD_HHMMSS/
│   ├── config.json
│   ├── metrics.json
│   └── generations.jsonl
└── ...

models/              # Fine-tuned model checkpoints (excluded from git)
└── finetuned-*/
    └── latest/

.env.example                     # Template for Slurm/local environment variables
preprocess.py                   # Unified preprocessing pipeline
split.py                        # Apply train/val/test/attacker/holdout splits
extract_pii_graph.py            # PII ground-truth extraction (GLiNER + Stanza)
extract_mallinckrodt_profiles.py # Extract canonical name/email/phone profiles (Mallinckrodt)
augment_pii_graph_with_email.py # Add canonical email edges to a PII graph
train.py                        # Fine-tune GPT-Neo/Llama with LoRA
pii_extraction_attack.py        # Candidate-style likelihood-ratio attack
pii_autocomplete_attack.py      # Generation-based autocomplete attack
visualize.py                    # Unified results visualization dispatcher
visualize_training.py           # Plot training-history curves
config.py                       # Centralized constants and hyperparameters
Dockerfile                      # Singularity-compatible image build

jobs/                # Slurm job scripts and preset launchers
├── launch_finetune.py        # Preset-based Slurm job launcher
├── launch_pii_graph.py       # PII-graph extraction launcher
├── launch_pii_extraction_attack.py
├── launch_pii_autocomplete_attack.py
├── presets.py                # Preset definitions
└── slurm_common.py           # Shared Slurm-script rendering helpers

scripts/             # Helper scripts
├── augment_enron_email.py
├── build_singularity_image.sh
├── download_base_model.sh
├── download_ner_models.sh
└── generate_dataset_stats_table.py

utils/               # Shared utilities
├── env_config.py    # .env loading/validation for Slurm launchers
├── data_utils.py    # JSONL loading and user grouping
├── model_utils.py   # Model/tokenizer loading, LoRA wrapping
├── pii_extraction_attack_utils.py
├── pii_graph_utils.py
├── synthetic_distractors.py
├── viz_pii_extraction.py
├── viz_pii_autocomplete.py
├── training_logger.py
├── training_viz_utils.py
└── viz_utils.py

README.md            # Human-facing toolkit landing page
AGENTS.md            # This file — agent-focused guidance
```

## Code Style & Conventions

- **Language:** Python 3.11+
- **Typing:** Use type hints everywhere (`from __future__ import annotations`).
- **Formatting:** Follow PEP 8. Max line length 100.
- **Imports:** Group as stdlib, third-party, local.
- **Scripts:** Keep CLI scripts self-contained with `argparse`. Avoid hard-coded paths; always accept `--input` / `--output`.
- **Experiments:** Log hyperparameters and random seeds. Save results as JSON/JSONL, not plain text.
- **Docstrings:** Use Google-style docstrings for public functions.

## Dataset Preprocessing

`preprocess.py` is a unified pipeline with swappable strategies. Use `--preset kandpal_strict` to reproduce the paper preprocessing for Enron and `--preset mallinckrodt` for the Mallinckrodt corpus. `split.py` then applies the paper-specific train/test/user splits. See the paper and source code for the strategy details.

### Enron

```bash
docker run --rm \
  -v "${PWD}:/workspace" \
  priv_bench \
  /opt/venv/bin/python3 /workspace/preprocess.py \
    --maildir /workspace/datasets/enron_raw/maildir \
    --output /workspace/preprocessed/enron_unsplit.jsonl

docker run --rm \
  -v "${PWD}:/workspace" \
  priv_bench \
  /opt/venv/bin/python3 /workspace/split.py --strategy kandpal \
    --input /workspace/preprocessed/enron_unsplit.jsonl \
    --output /workspace/preprocessed/enron.jsonl
```

### Mallinckrodt

```bash
docker run --rm \
  -v "${PWD}:/workspace" \
  priv_bench \
  /opt/venv/bin/python3 /workspace/preprocess.py --preset mallinckrodt \
    --output /workspace/preprocessed/mallinckrodt_unsplit.jsonl

docker run --rm \
  -v "${PWD}:/workspace" \
  priv_bench \
  /opt/venv/bin/python3 /workspace/split.py --strategy kandpal \
    --input /workspace/preprocessed/mallinckrodt_unsplit.jsonl \
    --output /workspace/preprocessed/mallinckrodt.jsonl
```

## PII Ground Truth Extraction

`extract_pii_graph.py` builds the per-user PII ground-truth graph from a preprocessed JSONL dataset using GLiNER and Stanza coreference. The output is used as ground truth for the downstream attacks.

### Setup

Complete README.md Setup steps 1–5 first (NER models, base model, and reference word lists).

### Local Docker run

```bash
# Quick CPU smoke test: 2 users, 5 records each, no coref.
docker run --rm \
  -v "${PWD}:/workspace" \
  priv_bench \
  /opt/venv/bin/python3 /workspace/extract_pii_graph.py \
    --input /workspace/preprocessed/enron.jsonl \
    --output /workspace/results/pii_graph_test.json \
    --disable-coref \
    --max-users 2 \
    --max-records-per-user 5 \
    --max-chars 1000 \
    --device cpu
```

### Cluster presets

| Preset | Dataset | Users × records | Max chars | Wall time |
|---|---|---|---|---|
| `neo125-enron-strict-pii-graph-test` | Enron | 50 × 50 | 2,000 | 6h |
| `neo125-enron-strict-pii-graph` | Enron | all | 1,350 | 12h |
| `neo125-mallinckrodt-pii-graph-test` | Mallinckrodt | 10 × 50 | 1,350 | 6h |
| `neo125-mallinckrodt-pii-graph` | Mallinckrodt | all | 1,350 | 24h |

The Enron full preset runs with GLiREL turned off (fallback heuristics only) and periodic GPU-cache cleanup every 1,000 records to slow Stanza-coref memory growth.

### Adding canonical emails

For Enron, canonical sender emails are already stored in the `user` field. Run `scripts/augment_enron_email.py` to materialize them as `has_email_address` edges:

```bash
docker run --rm \
  -v "${PWD}:/workspace" \
  priv_bench \
  /opt/venv/bin/python3 /workspace/scripts/augment_enron_email.py \
    --records /workspace/preprocessed/enron.jsonl \
    --graph /workspace/preprocessed/pii_graph_enron.json \
    --output-dir /workspace/preprocessed
```

For Mallinckrodt, first extract canonical email profiles with `extract_mallinckrodt_profiles.py`, then add the edges with `augment_pii_graph_with_email.py`:

```bash
docker run --rm \
  -v "${PWD}:/workspace" \
  priv_bench \
  /opt/venv/bin/python3 /workspace/extract_mallinckrodt_profiles.py \
    --input /workspace/preprocessed/mallinckrodt.jsonl \
    --output-dir /workspace/preprocessed \
    --device cpu

docker run --rm \
  -v "${PWD}:/workspace" \
  priv_bench \
  /opt/venv/bin/python3 /workspace/augment_pii_graph_with_email.py \
    --graph /workspace/preprocessed/pii_graph_mallinckrodt.json \
    --profiles /workspace/preprocessed/mallinckrodt_xassoc_profiles.json \
    --output /workspace/preprocessed/pii_graph_mallinckrodt_with_email.json
```

## Training

`train.py` fine-tunes a base causal LM (GPT-Neo or Llama) on the preprocessed JSONL. It expects a `text` field and a `split` field (keeps only `"train"` records). Full fine-tuning is the default; pass `--lora` to enable LoRA adapters.

### Cluster deployment

Full training runs on the Slurm cluster via the preset launcher in `jobs/launch_finetune.py`.

```bash
# Inspect the generated Slurm script before submitting
python jobs/launch_finetune.py --preset neo125-enron-strict-lora --dry-run

# Submit the job
python jobs/launch_finetune.py --preset neo125-enron-strict-lora
```

Run without `--preset` to see an interactive numbered list.

Current training presets most relevant to the two main datasets:

| Preset | Model | Dataset | Partition |
|---|---|---|---|
| `neo125-enron-strict-lora` | GPT-Neo-125M | `enron` | `gpu-1xA100` |
| `neo1300-enron-strict-lora` | GPT-Neo-1.3B | `enron` | `gpu-1xA100` |
| `llama8b-enron-strict-lora` | Llama-3-8B | `enron` | `gpu-8xA100` |
| `neo125-mallinckrodt` | GPT-Neo-125M | `mallinckrodt` | `gpu-1xA100` |
| `neo1300-mallinckrodt-lora` | GPT-Neo-1.3B | `mallinckrodt` | `gpu-1xA100` |
| `llama8b-mallinckrodt-lora` | Llama-3-8B | `mallinckrodt` | `gpu-8xA100` |

Each preset saves outputs under `models/finetuned-{preset}/latest` and `models/finetuned-{preset}/latest-checkpoint/`. If a job fails, resubmitting the same preset automatically resumes from the latest checkpoint.

## Attacks

Run the attack presets defined in `jobs/presets.py`. The paper defines the threat models, metrics, and distractor settings. Use the key flags below to override preset values for custom sweeps.

### Candidate-style PII extraction

```bash
python jobs/launch_pii_extraction_attack.py \
  --preset neo125-enron-strict-lora-pii-limited-masked
```

Use `--context-mode any|header|structured` and `--distractor-mode real|synthetic` to sweep variations without creating new presets.

### PII autocomplete

```bash
python jobs/launch_pii_autocomplete_attack.py \
  --preset neo125-enron-strict-lora-pii-autocomplete
```

### Key attack flags

- `--model-path`: fine-tuned model or LoRA adapter directory
- `--base-model`: base model name if not inferable from adapter config
- `--pii-types`: e.g. `name phone email job_title`
- `--pii-graph`: path to a PII graph JSON (required for Mallinckrodt; optional for Enron)
- `--num-distractors`: default 20
- `--max-samples`: exposure contexts scored per user/type (default 100)
- `--max-context-tokens`: prefix length before the masked value (default 256)
- `--context-mode`: `any`, `header`, `structured`
- `--distractor-mode`: `synthetic`, `real`
- `--mask-pii-types-in-prefix`: redact target PII types from the prefix, e.g. `name,phone,job_title,email`
- `--mask-placeholder`: default `<MASK>`

## Utility Evaluation

`train.py` computes post-training perplexity / cross-entropy on held-out splits automatically and writes `models/finetuned-{preset}/latest/utility_metrics.json`.

## Visualization

`visualize.py` detects the result type in a folder and calls the appropriate module. Per-run plots default to PNG; use `--format pdf` (or `svg`) for publication-ready output.

| Type | Detection | Module |
|---|---|---|
| PII extraction | `extraction_scores.jsonl` + `exposure_table.jsonl` | `utils/viz_pii_extraction.py` |
| PII autocomplete | `generations.jsonl` + `by_prompt_set` in `metrics.json` | `utils/viz_pii_autocomplete.py` |
| Training curves | `training_history.jsonl` | `utils/training_viz_utils.py` / `visualize_training.py` |

`visualize.py` scans the given directory recursively for result folders. Use
`/workspace/outputs` for cluster runs and `/workspace/results` for local attack
runs.

```bash
# Interactive mode
docker run -it --rm \
  -v "${PWD}:/workspace" \
  -e MPLBACKEND=Agg \
  priv_bench \
  /opt/venv/bin/python3 /workspace/visualize.py \
    --results-dir /workspace/outputs

# Direct mode (PNG preview)
docker run --rm \
  -v "${PWD}:/workspace" \
  -e MPLBACKEND=Agg \
  priv_bench \
  /opt/venv/bin/python3 /workspace/visualize.py \
    --results-dir /workspace/outputs \
    --run pii_extraction_YYYYMMDD_HHMMSS

# PDF output
docker run --rm \
  -v "${PWD}:/workspace" \
  -e MPLBACKEND=Agg \
  priv_bench \
  /opt/venv/bin/python3 /workspace/visualize.py \
    --results-dir /workspace/outputs \
    --run pii_extraction_YYYYMMDD_HHMMSS \
    --format pdf

# Training curves
docker run --rm \
  -v "${PWD}:/workspace" \
  -e MPLBACKEND=Agg \
  priv_bench \
  /opt/venv/bin/python3 /workspace/visualize_training.py \
    --model-dir /workspace/models/finetuned-neo125-enron-strict-lora/latest \
    --format pdf
```

All plotting modules use `utils.viz_config.VizConfig` for a shared, colorblind-safe style
(Okabe-Ito palette, IEEE-friendly sizes) and `utils.viz_config.save_figure` / `utils.viz_utils.save_figure`
for consistent DPI, bbox, and padding.

## Testing Instructions

- Add or update tests for any code you change.
- Run the full suite before committing: `uv run pytest tests/`
- Preprocessing changes must not break `--preset kandpal_strict` reproductions.
- If you add a new strategy (e.g., a new deduplicator), add a unit test and document it in this file.

## Security & Privacy Considerations

- This project handles **real email data** (Enron, Mallinckrodt). Never commit raw datasets or processed outputs containing PII to the repository.
- `.gitignore` must exclude `datasets/`, `preprocessed/`, and any `*.jsonl` files.
- When logging or printing examples, truncate or anonymize sender addresses.
- The attack code simulates an adversary; ensure it is clearly labeled as an auditing tool, not a malicious script.

## Reproducibility Notes

- **Compute target:** tested on a Slurm-managed cluster with NVIDIA A100 GPUs (1xA100 to 8xA100 nodes) and Singularity.
- **Framework:** PyTorch + HugFlow Transformers + PEFT (LoRA).
- **Base models:** GPT-Neo 125M/1.3B and Llama-3-8B (HuggingFace).
- **Target datasets:** Enron Emails and Mallinckrodt Litigation Documents.
- **Random seeds:** Always set `random`, `numpy`, and `torch` seeds. Default seed is 42.

