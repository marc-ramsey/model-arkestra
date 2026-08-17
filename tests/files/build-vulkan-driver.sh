#!/usr/bin/env bash
set -euo pipefail

# build-vulkan-driver.sh
# Builds the minimal Vulkan driver-only container image for podman/docker.
# The llama-server binary is mounted from host at runtime.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINERFILE="$SCRIPT_DIR/Containerfile.vulkan-radv"
IMAGE_NAME="ark-llama:vulkan-radv"

echo "==> Building $IMAGE_NAME from $CONTAINERFILE ..."

podman build -t "$IMAGE_NAME" -f "$CONTAINERFILE" . 2>&1

echo ""
echo "==> Done: $IMAGE_NAME ($(podman inspect "$IMAGE_NAME" --format '{{.Size}}' | numfmt --to=iec))"
