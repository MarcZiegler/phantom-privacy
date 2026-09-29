#!/bin/bash
# Download a base causal LM to a local directory for offline cluster runs.
#
# Usage:
#   bash scripts/download_base_model.sh [model_id]
#
# Examples:
#   bash scripts/download_base_model.sh
#   bash scripts/download_base_model.sh EleutherAI/gpt-neo-125M
#   bash scripts/download_base_model.sh NousResearch/Meta-Llama-3-8B

set -euo pipefail

MODEL_ID="${1:-EleutherAI/gpt-neo-125M}"

# Derive a local folder name from the model ID, e.g.:
#   EleutherAI/gpt-neo-125M      -> gpt-neo-125m
#   NousResearch/Meta-Llama-3-8B -> meta-llama-3-8b
MODEL_DIR_NAME=$(echo "${MODEL_ID}" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/--*/-/g; s/^-//; s/-$//')
LOCAL_DIR="models/base/${MODEL_DIR_NAME}"

echo "Downloading ${MODEL_ID} to ${LOCAL_DIR} ..."
mkdir -p "${LOCAL_DIR}"

uv run hf download \
    "${MODEL_ID}" \
    --local-dir "${LOCAL_DIR}"

echo "Done. Model saved to ${LOCAL_DIR}"
