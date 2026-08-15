#!/usr/bin/env bash
# post_install.sh — create venv, install deps, add venv to PATH in shell profiles.
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

# ── Add venv/bin to PATH in shell profiles ──────────────────────────────
VENV_LINE="export PATH=\"$VENV/bin:\$PATH\""
for profile_file in "$HOME/.bashrc" "$HOME/.profile"; do
    [ -f "$profile_file" ] || continue
    if grep -qF "$VENV_LINE" "$profile_file" 2>/dev/null; then
        echo "  PATH already added to $profile_file"
    else
        echo "" >> "$profile_file"
        echo "# model-arkestra venv ($(date +%Y-%m-%d))" >> "$profile_file"
        echo "$VENV_LINE" >> "$profile_file"
        echo "  Added PATH to $profile_file"
    fi
done

echo "Done. Source ~/.bashrc or restart your terminal."
