#!/usr/bin/env bash
# post_install.sh — wrap pip-installed entry points so they auto-activate venv.
# Run this after: source venv/bin/activate && pip install -e ".[proxy]"
set -euo pipefail

venv_bin="$(cd "$(dirname "$0")/.." && pwd)/venv/bin"

for script in arkestra-server arkestra-cli; do
    dst="$venv_bin/$script"
    [ -f "$dst" ] || continue

    # Skip if this is already our auto-activating wrapper (idempotent)
    if head -3 "$dst" | grep -q "Auto-generated.*activates venv"; then
        continue
    fi

    # Preserve the original entry-point body as .real
    cp "$dst" "$dst.real"

    # Replace with auto-activating wrapper.
    # Strategy: source venv/bin/activate, then use $VIRTUAL_ENV directly
    # (avoids which/path resolution issues in some environments).
    cat > "$dst" << 'WRAPPER'
#!/usr/bin/env bash
# Auto-generated: activates venv if $VIRTUAL_ENV is unset, then execs the real script.
if [ -z "${VIRTUAL_ENV:-}" ]; then
    # Source activate from same bin directory as this script
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/activate" ]; then
        . "$SCRIPT_DIR/activate"
    else
        echo "Error: could not locate venv activate at $SCRIPT_DIR" >&2
        exit 1
    fi
fi

# Use VIRTUAL_ENV to find python (reliable, no PATH/which caching issues)
PYTHON="${VIRTUAL_ENV}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Error: venv python not found at $PYTHON (is VIRTUAL_ENV=$VIRTUAL_ENV valid?)" >&2
    exit 1
fi

exec -a "$0" "$PYTHON" "${BASH_SOURCE[0]}.real" "$@"
WRAPPER

    rm -f "$venv_bin/$script.py" "$venv_bin/__pycache__/"* 2>/dev/null || true
done

echo "Post-install hooks applied for: arkestra-server, arkestra-cli"
