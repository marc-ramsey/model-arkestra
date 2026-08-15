#!/usr/bin/env bash
# post_install.sh — create venv, install deps, wrap entry points.
# Wrapper resolves its own path absolutely; works from any shell context.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT/venv"

# ── Step 1: Create venv if needed ────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

# Activate inside this script so pip installs into the right place
. "$VENV/bin/activate"

# ── Step 2: Install packages into this venv ─────────────────────────────
echo "Installing package..."
[ -d "$PROJECT/vendor/llm-config-manager" ] && \
    pip install -e "$PROJECT/vendor/llm-config-manager" --quiet
pip install -e "$PROJECT/[proxy]" --quiet

# ── Step 3: Wrap entry-point scripts — no activation, path resolution only ─
for script in arkestra-server arkestra-cli; do
    dst="$VENV/bin/$script"
    [ -f "$dst" ] || continue

    # Already wrapped? Skip.
    if head -1 "$dst" 2>/dev/null | grep -q '#!/usr/bin/env bash'; then
        continue
    fi

    # Save original entry-point body (skip pip's shebang)
    tail -n +2 "$dst" > "$dst.real"
    chmod --reference="$dst" "$dst.real" 2>/dev/null || true

    # Write thin wrapper that resolves its own path in all shell contexts:
    #   ./venv/bin/arkestra-cli  → reads CWD-relative, readlink -f resolves
    #   /abs/path/arkestra-cli   → absolute, readlink -f works
    #   arkestra-cli (via PATH)  → BASH_SOURCE not set, falls back to command -v
    cat > "$dst" << 'WRAPPER'
#!/usr/bin/env bash
# Auto-generated — resolves own path, uses venv python directly.
SELF="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$(command -v "$0")}" 2>/dev/null || echo "$0")")" && pwd)"
exec "$SELF/python" "$SELF/${0##*/}.real" "$@"
WRAPPER

    rm -f "$VENV/bin/$script.py" "$VENV/bin/__pycache__/"* 2>/dev/null || true
done

echo "Done. Entry points work from any directory without activation."
