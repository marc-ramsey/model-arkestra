#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# If venv doesn't exist yet, post_install.sh creates it + installs
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Post-install handles: pip install, vendor deps, entry-point wrapping
scripts/post_install.sh
