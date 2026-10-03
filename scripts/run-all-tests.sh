#!/usr/bin/env bash
# Run all tests (unit + e2e). Requires no other llama-server on this GPU.
set -uo pipefail

cd "$(dirname "$0")/.."

echo "=== Phase 1: unit + integration ==="
python -m pytest tests/ \
    --timeout=3600 \
    -v \
    -m "not e2e and not slow"
P1=$?

if [ $P1 -ne 0 ]; then
    echo ""
    echo "Phase 1 FAILED (exit $P1). Aborting."
    exit $P1
fi

echo ""
echo "=== Pausing 30s for GPU/driver resource release ==="
sleep 30

echo ""
echo "=== Phase 2: e2e (real model loads) ==="
python -m pytest tests/ \
    --timeout=3600 \
    -v \
    -m "e2e and not slow"
P2=$?

exit $P2
