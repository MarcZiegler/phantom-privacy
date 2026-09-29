#!/bin/bash
# Build a Singularity image from the local Dockerfile.
#
# Usage:
#   bash scripts/build_singularity_image.sh [image_name]
#   bash scripts/build_singularity_image.sh --force [image_name]
#
# The final .sif is written to images/<image_name>.sif.
# Caching: if Dockerfile, pyproject.toml, and uv.lock have not changed since the
# last build, the existing Docker image and/or Singularity .sif are reused.

set -euo pipefail

FORCE=false
IMAGE_NAME=""

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --force)
            FORCE=true
            ;;
        -*)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--force] [image_name]"
            exit 1
            ;;
        *)
            if [ -z "$IMAGE_NAME" ]; then
                IMAGE_NAME="$arg"
            else
                echo "Usage: $0 [--force] [image_name]"
                exit 1
            fi
            ;;
    esac
done

IMAGE_NAME="${IMAGE_NAME:-priv_bench}"
DOCKER_TAG="${IMAGE_NAME}"
TAR_FILE="${IMAGE_NAME}.tar"
OUTPUT_DIR="images"
SIF_FILE="${OUTPUT_DIR}/${IMAGE_NAME}.sif"
HASH_FILE="${OUTPUT_DIR}/.${IMAGE_NAME}.hash"
DOCKER_HASH_FILE="${OUTPUT_DIR}/.${IMAGE_NAME}.docker_hash"

echo "Building Singularity image '${IMAGE_NAME}' ..."
echo "  Docker tag: ${DOCKER_TAG}"
echo "  Output:     ${SIF_FILE}"
echo "  Force:      ${FORCE}"

mkdir -p "${OUTPUT_DIR}"

# Files that affect the image environment
HASH_INPUTS=(Dockerfile pyproject.toml uv.lock)

# Compute a content hash of the image inputs
current_hash() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${HASH_INPUTS[@]}" | sha256sum | awk '{print $1}'
    elif command -v md5sum >/dev/null 2>&1; then
        md5sum "${HASH_INPUTS[@]}" | md5sum | awk '{print $1}'
    else
        echo "ERROR: sha256sum or md5sum is required for caching." >&2
        exit 1
    fi
}

CURRENT_HASH=$(current_hash)
echo "  Input hash: ${CURRENT_HASH}"

# Full cache skip: .sif exists and hash matches
if [ "$FORCE" != true ] && [ -f "${SIF_FILE}" ] && [ -f "${HASH_FILE}" ]; then
    if [ "$(cat "${HASH_FILE}")" = "${CURRENT_HASH}" ]; then
        echo ""
        echo "Singularity image is up to date. Skipping build."
        echo "Use --force to rebuild."
        exit 0
    fi
fi

# Decide whether we can reuse the existing Docker image
SKIP_DOCKER=false
if [ "$FORCE" != true ] && docker image inspect "${DOCKER_TAG}" >/dev/null 2>&1 && [ -f "${DOCKER_HASH_FILE}" ]; then
    if [ "$(cat "${DOCKER_HASH_FILE}")" = "${CURRENT_HASH}" ]; then
        SKIP_DOCKER=true
    fi
fi

# Git Bash needs Windows-style paths for Docker volume mounts.
if command -v cygpath >/dev/null 2>&1; then
    MOUNT_PATH=$(cygpath -w "$PWD")
elif pwd -W >/dev/null 2>&1; then
    MOUNT_PATH=$(pwd -W)
else
    MOUNT_PATH="$PWD"
fi

if [ "$SKIP_DOCKER" = true ]; then
    echo "[1/3] Docker image is up to date. Skipping docker build."
else
    echo "[1/3] Building Docker image..."
    docker build -t "${DOCKER_TAG}" .
    echo "${CURRENT_HASH}" > "${DOCKER_HASH_FILE}"
fi

echo "[2/3] Saving Docker image to tar..."
docker save -o "${TAR_FILE}" "${DOCKER_TAG}"

if [ ! -s "${TAR_FILE}" ]; then
    echo "ERROR: ${TAR_FILE} is missing or empty after docker save." >&2
    exit 1
fi

echo "[3/3] Converting tar to Singularity .sif..."
# MSYS_NO_PATHCONV prevents Git Bash from rewriting /data/... paths into Windows paths
# before passing them to the Windows docker executable.
MSYS_NO_PATHCONV=1 docker run --rm -v "${MOUNT_PATH}:/data" biohpc/sindocker \
    singularity build --force "/data/${SIF_FILE}" "docker-archive:///data/${TAR_FILE}"

echo "Cleaning up intermediate tar file..."
rm -f "${TAR_FILE}"

# Record the hash of inputs that produced this .sif
echo "${CURRENT_HASH}" > "${HASH_FILE}"

echo ""
echo "Done. Singularity image: ${SIF_FILE}"
ls -lh "${SIF_FILE}"
