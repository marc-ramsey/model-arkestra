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

    # Replace with auto-activating wrapper
    cat > "$dst" << 'WRAPPER'
#!/usr/bin/env bash
# Auto-generated: activates venv if $VIRTUAL_ENV is unset, then execs the real script.
if [ -z "${VIRTUAL_ENV:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # SCRIPT_DIR is <project>/venv/bin — look for activate in the same bin dir
    if [ -f "$SCRIPT_DIR/activate" ]; then
        . "$SCRIPT_DIR/activate"
    else
        echo "Warning: could not locate venv to auto-activate." >&2
    fi
fi
# shellcheck disable=SC1090
exec -a "$0" "$(which python)" "${BASH_SOURCE[0]}.real" "$@"
WRAPPER

    rm -f "$venv_bin/$script.py" "$venv_bin/__pycache__/"* 2>/dev/null || true
done

echo "Post-install hooks applied for: arkestra-server, arkestra-cli"
