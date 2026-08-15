#!/usr/bin/env bash
# post_install.sh — create venv, install deps. No wrapping needed.
# pip installs entry points with correct shebang (#!/venv/bin/python)
# which works from any directory without activation.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT/venv"

# ── Create venv if needed ────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

# Activate so pip installs into this venv (correct shebang + .pth files)
. "$VENV/bin/activate"

# ── Install packages ────────────────────────────────────────────────────
echo "Installing package..."
[ -d "$PROJECT/vendor/llm-config-manager" ] && \
    pip install -e "$PROJECT/vendor/llm-config-manager" --quiet
pip install -e "$PROJECT/[proxy]" --quiet

echo "Done. Entry points use venv python shebang — works from any directory."
