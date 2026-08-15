#!/usr/bin/env bash
# post_install.sh — idempotent entry-point wrapper + pip install with auto-activation.
# Works even when run from an unactivated shell.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="$(command -v python3)"
VENV="venv"
VENV_PYTHON="$PWD/$VENV/bin/python"

# ── Step 1: Create venv if needed, and always activate it here ────────
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV"
fi

# Activate — sets VIRTUAL_ENV, updates PATH for rest of script
. "$VENV/bin/activate"

# ── Step 2: Install the package into this venv ────────────────────────
echo "Installing package (editable mode with [proxy] extras)..."
# llm-config-manager is vendored locally — install it first
[ -d vendor/llm-config-manager ] && pip install -e vendor/llm-config-manager --quiet
pip install -e ".[proxy]" --quiet

# ── Step 3: Wrap entry-point scripts so they auto-activate too ────────
for script in arkestra-server arkestra-cli; do
    dst="$VENV/bin/$script"
    [ -f "$dst" ] || continue

    # Already wrapped? Skip.
    if head -3 "$dst" 2>/dev/null | grep -q "Auto-generated.*activates venv"; then
        continue
    fi

    # Save original body (everything after shebang)
    tail -n +2 "$dst" > "$dst.real"
    chmod --reference="$dst" "$dst.real" 2>/dev/null || true

    # Write auto-activating wrapper
    cat > "$dst" << 'WRAPPER'
#!/usr/bin/env bash
# Auto-generated: activates venv if $VIRTUAL_ENV is unset, then execs the real script.
if [ -z "${VIRTUAL_ENV:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/activate" ]; then
        . "$SCRIPT_DIR/activate"
    else
        echo "Error: could not locate venv activate at $SCRIPT_DIR" >&2
        exit 1
    fi
fi
PYTHON="${VIRTUAL_ENV}/bin/python"
[ -x "$PYTHON" ] || { echo "Error: venv python not found at $PYTHON" >&2; exit 1; }
exec -a "$0" "$PYTHON" "${BASH_SOURCE[0]}.real" "$@"
WRAPPER

    rm -f "$VENV/bin/$script.py" "$VENV/bin/__pycache__/"* 2>/dev/null || true
done

echo "Done. Entry points auto-activate venv on first call."
