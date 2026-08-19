#!/usr/bin/env bash
set -euo pipefail

# build-rocm-driver.sh
# Builds the minimal ROCm driver-only container image for podman/docker.
# The llama-server binary is mounted from host at runtime.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTAINERFILE="$PROJECT_DIR/tests/files/Containerfile.rocm"
IMAGE_NAME="localhost/ark-llama:rocm"
RUNTIME="${1:-podman}"

if ! command -v "$RUNTIME" &>/dev/null; then
  echo "Error: $RUNTIME not found. Install it or run with 'podman' or 'docker'."
  exit 1
fi

echo "==> Building $IMAGE_NAME from $CONTAINERFILE using $RUNTIME ..."

"$RUNTIME" build -t "$IMAGE_NAME" -f "$CONTAINERFILE" "$PROJECT_DIR" 2>&1

echo ""
echo "==> Done: $IMAGE_NAME ($("$RUNTIME" inspect "$IMAGE_NAME" --format '{{.Size}}' | numfmt --to=iec))"
