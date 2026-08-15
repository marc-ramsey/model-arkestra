#!/usr/bin/env bash
# post_install.sh — create venv, install deps, wrap entry points.
# Works from any directory or activation state.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT/venv"
PYTHON="python3"

# ── Step 1: Create venv if needed, then activate ────────────────────────
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV"
fi

# Activate — use absolute path so cd in activate is deterministic
. "$VENV/bin/activate"

# ── Step 2: Install packages into this venv ─────────────────────────────
echo "Installing package..."
[ -d "$PROJECT/vendor/llm-config-manager" ] && \
    pip install -e "$PROJECT/vendor/llm-config-manager" --quiet
pip install -e "$PROJECT/[proxy]" --quiet

# ── Step 3: Wrap entry-point scripts with absolute-path resolution ──────
for script in arkestra-server arkestra-cli; do
    dst="$VENV/bin/$script"
    [ -f "$dst" ] || continue

    if head -3 "$dst" 2>/dev/null | grep -q "Auto-generated.*activates venv"; then
        continue
    fi

    # Save original entry-point body (skip pip's shebang)
    tail -n +2 "$dst" > "$dst.real"
    chmod --reference="$dst" "$dst.real" 2>/dev/null || true

    # Write wrapper using absolute paths — no cd, no BASH_SOURCE tricks
    cat > "$dst" << EOF
#!/usr/bin/env bash
if [ -z "\${VIRTUAL_ENV:-}" ]; then
    . $(printf %q "$VENV/bin/activate")
fi
exec -a \$0 ${VENV}/bin/python "${dst}.real" "\$@"
EOF

    rm -f "$VENV/bin/$script.py" "$VENV/bin/__pycache__/"* 2>/dev/null || true
done

echo "Done. Entry points auto-activate venv on first call."
