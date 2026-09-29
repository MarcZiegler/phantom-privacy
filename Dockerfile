# Singularity-compatible image for phantom-privacy.
# Build locally with Docker, export to tar, then convert to Singularity .sif.
#
#   docker build -t priv_bench .
#   docker save -o priv_bench.tar priv_bench
#   docker run --rm -v "${PWD}:/data" biohpc/sindocker \
#       singularity build --force /data/priv_bench.sif docker-archive:///data/priv_bench.tar

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1

# Install system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv to a system-wide location so it works for any user inside Singularity.
ADD https://astral.sh/uv/install.sh /tmp/uv-installer.sh
RUN sh /tmp/uv-installer.sh && rm /tmp/uv-installer.sh && \
    install -m 755 /root/.local/bin/uv /usr/local/bin/uv

# Install a managed Python 3.11 interpreter under /opt/uv-python (not under
# /root) so non-root Singularity users can traverse to and execute it.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.11 && \
    chmod -R 755 /opt/uv-python

# Create the project virtual environment at /opt/venv. This path is NOT under
# /workspace, so it is not hidden by the runtime bind mount that overlays
# source code onto /workspace. The venv symlinks to /opt/uv-python.
RUN uv venv /opt/venv --python $(uv python find 3.11)

# Put the venv first on PATH so that 'python3' resolves to the venv interpreter.
ENV PATH="/opt/venv/bin:/usr/local/bin:${PATH}"

WORKDIR /workspace

# Copy dependency metadata and README first for better layer caching.
# README.md is required because pyproject.toml references it.
COPY pyproject.toml uv.lock README.md ./

# Install locked runtime dependencies into /opt/venv. uv.lock now resolves
# torch to the CUDA 11.8 wheel index, so no special handling is needed.
RUN uv export --no-dev --no-emit-project --output-file requirements.txt && \
    uv pip install --python /opt/venv/bin/python3 -r requirements.txt --no-cache && \
    rm requirements.txt && \
    chmod -R a+rX /opt/venv

# Project source is bind-mounted from scratch at runtime, so it is not baked
# into the image. This keeps the image small and avoids rebuilding it for code
# changes.

# Default to a bash shell (Singularity exec will override this anyway)
CMD ["/bin/bash"]
