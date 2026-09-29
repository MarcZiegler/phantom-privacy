#!/bin/bash
# Download offline model weights for the PII-graph extraction pipeline.
# Run from the project root.
#
# This script is cluster-friendly: if the project .venv is not usable on the
# current host (e.g. it was created on Windows), it creates a temporary
# download venv and installs gliner/glirel/stanza there.

set -euo pipefail

MODELS_DIR="models/ner"
mkdir -p "$MODELS_DIR"

# ---------------------------------------------------------------------------
# Find or create a working Python interpreter with gliner/glirel/stanza.
# ---------------------------------------------------------------------------

DOWNLOAD_VENV="$MODELS_DIR/.download_venv"
PYTHON=""

if [ -x ".venv/bin/python3" ]; then
    if .venv/bin/python3 -c "import gliner, glirel, stanza" >/dev/null 2>&1; then
        PYTHON=".venv/bin/python3"
        echo "Using project virtual environment: $PYTHON"
    else
        echo "Project .venv exists but is missing gliner/glirel/stanza."
    fi
fi

if [ -z "$PYTHON" ]; then
    echo "Creating temporary download virtual environment..."
    rm -rf "$DOWNLOAD_VENV"
    uv venv "$DOWNLOAD_VENV" --python 3.11
    uv pip install --python "$DOWNLOAD_VENV/bin/python3" gliner glirel stanza loguru peft
    PYTHON="$DOWNLOAD_VENV/bin/python3"
    echo "Using temporary download virtual environment: $PYTHON"
fi

export PYTHON

# ---------------------------------------------------------------------------
# Download models.
# ---------------------------------------------------------------------------

echo "Downloading GLiNER NER model..."
hf download gliner-community/gliner_small-v2.5 \
  --local-dir "$MODELS_DIR/gliner_small-v2.5"

echo "Downloading GLiREL relation-extraction model..."
hf download jackboyla/glirel-large-v0 \
  --local-dir "$MODELS_DIR/glirel-large-v0"

echo "Downloading Stanza English coreference model..."
STANZA_RESOURCES_DIR="$MODELS_DIR/stanza_resources" "$PYTHON" - <<'PY'
import os
import stanza
os.environ["STANZA_RESOURCES_DIR"] = "models/ner/stanza_resources"
stanza.download("en", model_dir="models/ner/stanza_resources", processors="tokenize,coref")
PY

echo "Warming up GLiNER / GLiREL / Stanza caches..."
STANZA_RESOURCES_DIR="$MODELS_DIR/stanza_resources" "$PYTHON" - <<'PY'
import os
from gliner import GLiNER
from glirel import GLiREL
import stanza

GLiNER.from_pretrained("models/ner/gliner_small-v2.5")
GLiREL._from_pretrained(
    model_id='models/ner/glirel-large-v0',
    revision=None, cache_dir=None, force_download=False,
    proxies=None, resume_download=False, local_files_only=True,
    token=None, map_location='cpu', strict=False,
)
# Also load Stanza so the underlying XLM-R encoder is cached.
os.environ["STANZA_RESOURCES_DIR"] = "models/ner/stanza_resources"
stanza.Pipeline("en", dir="models/ner/stanza_resources", processors="tokenize,coref")
PY

echo "Done. Models are in $MODELS_DIR/ and HuggingFace cache has been populated."
