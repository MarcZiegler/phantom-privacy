# Phantom-Privacy

A toolkit for auditing **cross-user PII leakage** in local fine-tuned LLMs.

**Note:** This README is a practical setup and reproduction guide. The detailed
methodology, threat model, preprocessing choices, split strategy, and experiment
design are described in the paper (see [Citation](#citation)).

## What this toolkit does

Fine-tuned LLMs can leak PII about users whose own records were excluded from training. `phantom-privacy` audits this risk through six stages: preprocessing, splitting, PII extraction, fine-tuning, evaluation, and visualization. See the paper for the threat model and experimental design.

The toolkit is demonstrated on two real-world email corpora:

| Dataset | Users (training / phantom) | Records | Train | Attacker split | Holdout |
|---|---|---:|---:|---:|---:|
| Enron | 90 (45 / 45) | 83,069 | 39,392 | 8,301 | 30,998 |
| Mallinckrodt | 97 (48 / 49) | 176,047 | 82,592 | 17,602 | 66,673 |

## Data

This toolkit works on two public email corpora. Both contain real persons' communications; handle them responsibly and do not redistribute processed outputs that expose individual identities.

### Enron Emails

- **Source:** <https://www.cs.cmu.edu/~enron/>
- **Download:** the Enron `maildir` archive (e.g. `enron_mail_20150507.tar.gz`).
- **Size:** ~1.4 GB uncompressed; ~150 user folders; ~517,000 email files.
- **Expected layout:**
  ```
  datasets/enron_raw/maildir/
  ├── allen-p/
  │   ├── all_documents/
  │   │   └── 1.
  │   └── ...
  ├── arnold-j/
  └── ...
  ```
- The preprocessing script expects `--maildir /workspace/datasets/enron_raw/maildir`.

### Mallinckrodt Litigation Documents (UCSF IDL)

- **Source:** <https://ucsf.app.box.com/v/IDL-DataSets>
- **Download:** the Mallinckrodt Litigation Documents metadata CSVs from the UCSF Industry Documents Library.
- **Size:** ~182 GB across ~1,670 pipe-delimited CSV files (`Mallinckrodt_Litigation_Documents_*.csv`).
- **Expected layout:**
  ```
  datasets/Mallinckrodt_Litigation_Documents/
  ├── Mallinckrodt_Litigation_Documents_1.csv
  ├── Mallinckrodt_Litigation_Documents_2.csv
  └── ...
  ```
- Each row contains metadata fields (`type`, `author`, `recipient`, `copied`, `subject`, `document_date`, `custodian`) and an `ocr_text` column. Rows whose `type` contains `Email` are treated as emails.

> **Note:** downloading the full Mallinckrodt collection may require accepting the archive's terms of use and can take several hours.

## What we release

We release:
- this toolkit and its source code,
- the preprocessing, splitting, PII-extraction, and attack scripts,
- documentation and configuration files.

## Setup

This toolkit assumes a Slurm-managed GPU cluster with Singularity. GPU stages are submitted via the launchers and run inside the built Singularity image; other environments need adaptation.

All Slurm jobs read their cluster-specific settings from a `.env` file at the repository root. The file is gitignored and must never be committed.

### 1. Copy the template

```bash
cp .env.example .env
```

### 2. Fill in the required variables

Open `.env` and set at least:

```bash
SLURM_MAIL_USER=your.email@example.com
SLURM_PARTITION=gpu-1xA100

PHANTOM_PRIVACY_PROJECT_DIR=$HOME/phantom-privacy
PHANTOM_PRIVACY_SCRATCH_ROOT=/scratch/users/$USER
PHANTOM_PRIVACY_SIF_PATH=$PHANTOM_PRIVACY_PROJECT_DIR/images/priv_bench.sif
PHANTOM_PRIVACY_BASE_MODEL_CACHE=$PHANTOM_PRIVACY_SCRATCH_ROOT/base-models
PHANTOM_PRIVACY_LOG_DIR=$PHANTOM_PRIVACY_PROJECT_DIR/jobs/logs
```

Use `--dry-run` to inspect the generated Slurm script before submitting any job.

You can override these values for a single job:

```bash
python jobs/launch_finetune.py --preset neo125-enron-strict-lora \
  --partition gpu-2xA100 --mail-user me@example.com
```

### 3. Build the container

```bash
bash scripts/build_singularity_image.sh
```

This produces the Docker image `priv_bench` and the Singularity image `images/priv_bench.sif`.

### 4. Download offline models

```bash
bash scripts/download_ner_models.sh
bash scripts/download_base_model.sh EleutherAI/gpt-neo-125M
bash scripts/download_base_model.sh EleutherAI/gpt-neo-1.3B
bash scripts/download_base_model.sh NousResearch/Meta-Llama-3-8B
```

The NER setup downloads the offline GLiNER and Stanza models used by `extract_pii_graph.py`.

### 5. Download the reference word lists

Place the following files under `configs/`:

- `configs/pii_graph_common_words.txt` — common English words.  
  Source: <https://github.com/first20hours/google-10000-english>

- `configs/pii_graph_job_titles.txt` — newline-delimited job titles.  
  Example source: <https://github.com/jneidel/job-titles>


### How jobs use scratch space

Every Slurm job creates a unique directory under `$PHANTOM_PRIVACY_SCRATCH_ROOT/$SLURM_JOB_ID` on the compute node. The job:

1. Copies the repository source, base model, NER models, preprocessed data, and fine-tuned model (for attacks/utility) into scratch.
2. Runs inside the Singularity container with scratch directories bind-mounted to `/workspace`, `/models`, `/data`, and `/outputs`.
3. Copies results back to persistent storage under `$PHANTOM_PRIVACY_PROJECT_DIR`.
4. Removes the scratch directory on exit.

Using local scratch avoids heavy NFS traffic and is required by many cluster policies.

### Presets

All experiments are encoded as presets in `jobs/presets.py`. Run any launcher
without `--preset` to see the full list, or inspect `jobs/presets.py` for
parameter values. Preset names map to the experiments reported in the paper;
see the paper for the underlying threat-model definitions.

Main suffixes you will see:
- `-lora` — LoRA fine-tuning.
- `-pii-*` — attack / evaluation presets.
- `-limited-masked` / `-held-out-records-*` — candidate-extraction variants.
- `-real` — real distractors.
- `-structured` / `-header` — context ablations.
- `-autocomplete` / `-autocomplete-masked` — generation-based evaluation.

Use the commands in [Replicating the results](#replicating-the-results) to
reproduce the privacy audit analyzed in the paper.

## Replicating the results

Preprocessing, splitting, and visualization run inside the local Docker container.
PII-graph extraction, fine-tuning, and attacks are GPU stages and should be
submitted through the Slurm launchers.

### Enron

```bash
# 1. Preprocess and split (Docker, CPU)
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

# 2. Extract PII ground truth (Slurm cluster, GPU)
python jobs/launch_pii_graph.py --preset neo125-enron-strict-pii-graph

# 3. Add canonical email edges (Docker, CPU)
docker run --rm \
  -v "${PWD}:/workspace" \
  priv_bench \
  /opt/venv/bin/python3 /workspace/scripts/augment_enron_email.py \
    --records /workspace/preprocessed/enron.jsonl \
    --graph /workspace/preprocessed/pii_graph_enron.json \
    --output-dir /workspace/preprocessed

# 4. Fine-tune GPT-Neo-125M (Slurm cluster, GPU)
python jobs/launch_finetune.py --preset neo125-enron-strict-lora

# 5. Run attacks (Slurm cluster, GPU)
python jobs/launch_pii_extraction_attack.py \
  --preset neo125-enron-strict-lora-pii-limited-masked
python jobs/launch_pii_extraction_attack.py \
  --preset neo125-enron-strict-lora-pii-held-out-records-any
python jobs/launch_pii_autocomplete_attack.py \
  --preset neo125-enron-strict-lora-pii-autocomplete

# 6. Visualize (Docker, CPU)
docker run --rm \
  -v "${PWD}:/workspace" \
  -e MPLBACKEND=Agg \
  priv_bench \
  /opt/venv/bin/python3 /workspace/visualize.py \
    --results-dir /workspace/outputs
```

Use the `-test` PII-graph preset for a quick smoke test.

### Mallinckrodt

```bash
# 1. Preprocess and split (Docker, CPU)
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

# 2. Extract PII ground truth (Slurm cluster, GPU)
python jobs/launch_pii_graph.py --preset neo125-mallinckrodt-pii-graph

# 3. Extract canonical email profiles and add email edges (Docker, CPU)
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

# 4. Fine-tune GPT-Neo-125M (Slurm cluster, GPU)
python jobs/launch_finetune.py --preset neo125-mallinckrodt

# 5. Run attacks (Slurm cluster, GPU)
python jobs/launch_pii_extraction_attack.py \
  --preset neo125-mallinckrodt-pii-graph-attack-limited-masked
python jobs/launch_pii_extraction_attack.py \
  --preset neo125-mallinckrodt-pii-graph-attack-held-out-records
python jobs/launch_pii_autocomplete_attack.py \
  --preset neo125-mallinckrodt-pii-autocomplete

# 6. Visualize (Docker, CPU)
docker run --rm \
  -v "${PWD}:/workspace" \
  -e MPLBACKEND=Agg \
  priv_bench \
  /opt/venv/bin/python3 /workspace/visualize.py \
    --results-dir /workspace/outputs
```

## Project structure

```
.
├── .env.example                      # Template for Slurm/local environment variables
├── preprocess.py                    # Email preprocessing (Enron / Mallinckrodt)
├── split.py                         # Train/val/test/attacker/holdout splits
├── extract_pii_graph.py             # PII ground-truth extraction (GLiNER + Stanza)
├── extract_mallinckrodt_profiles.py # Extract canonical name/email/phone profiles (Mallinckrodt)
├── augment_pii_graph_with_email.py  # Add canonical email edges to PII graph
├── train.py                         # Fine-tune GPT-Neo / Llama with LoRA
├── pii_extraction_attack.py         # Candidate-style likelihood-ratio attack
├── pii_autocomplete_attack.py       # Generation-based autocomplete attack
├── visualize.py                     # Dispatch plots for attack results
├── visualize_training.py            # Plot training-history curves
├── scripts/                         # Container build, model download, dataset stats
├── jobs/                            # Slurm launchers and preset definitions
│   ├── presets.py                   # All preset configurations
│   ├── launch_finetune.py
│   ├── launch_pii_graph.py
│   ├── launch_pii_extraction_attack.py
│   └── launch_pii_autocomplete_attack.py
├── utils/                           # Shared utilities
├── configs/                         # PII schema, FSDP config, word-list paths (download per Setup step 5)
├── datasets/                        # Raw data (gitignored)
├── preprocessed/                    # JSONL outputs (gitignored)
├── models/                          # Fine-tuned checkpoints (gitignored)
└── results/                         # Local attack outputs (gitignored)
```

### Runtime file layout

After running the pipeline you will have:

```
.
├── datasets/                         # raw corpora (gitignored)
│   ├── enron_raw/
│   │   └── maildir/
│   └── Mallinckrodt_Litigation_Documents/
│       └── Mallinckrodt_Litigation_Documents_*.csv
├── preprocessed/                     # output of preprocess.py / split.py + PII graphs
│   ├── enron_unsplit.jsonl
│   ├── enron.jsonl
│   ├── pii_graph_enron.json
│   ├── pii_graph_enron_with_email.json
│   ├── mallinckrodt_unsplit.jsonl
│   ├── mallinckrodt.jsonl
│   ├── pii_graph_mallinckrodt.json
│   └── pii_graph_mallinckrodt_with_email.json
├── models/                           # base and fine-tuned models (gitignored)
│   ├── base/
│   │   ├── gpt-neo-125m/
│   │   ├── gpt-neo-1300m/
│   │   └── meta-llama-3-8b/
│   ├── ner/
│   │   ├── gliner_small-v2.5/
│   │   └── stanza_resources/
│   └── finetuned-<preset>/
├── outputs/                          # Slurm job outputs (gitignored)
│   ├── attack-<preset>/
│   │   └── <SLURM_JOB_ID>/
│   │       └── results/
│   │           └── pii_extraction_YYYYMMDD_HHMMSS/
│   ├── autocomplete-<preset>/
│   │   └── <SLURM_JOB_ID>/
│   │       └── results/
│   │           └── pii_autocomplete_YYYYMMDD_HHMMSS/
│   └── pii-graph-<preset>/
│       └── <SLURM_JOB_ID>/
│           └── pii_graph_<dataset>.json
├── images/
│   └── priv_bench.sif                # Singularity image produced by setup
└── jobs/logs/                        # Slurm stdout/stderr logs
```

Local attack runs (without Slurm) write timestamped folders under `results/`.

## Development

```bash
# Install local dev environment
uv sync --extra dev

# Run tests
uv run pytest tests/

# Lint
uv run ruff check .
```

## Citation

If you use this toolkit or the Phantom-Privacy datasets in your research, please cite our demo paper:

```bibtex
@inproceedings{ziegler2026phantom,
  title={Demo: Phantom-Privacy: A Toolkit for Auditing Cross-User PII Leakage in Fine-Tuned {LLMs}},
  author={Ziegler, Marc and Errounda, Fatima Zahra},
  booktitle={To be determined},
  year={2026},
  note={Paper submitted to IEEE Big Data 2026 and under review}
}
```

## Acknowledgments

This work was developed by Marc Ziegler during a visiting research stay at
Frostbyte Research Center, Reykjavik University, hosted by Fatima Zahra
Errounda, between June and September 2026.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

