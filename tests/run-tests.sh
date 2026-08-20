#!/usr/bin/env bash
# Wrapper around pytest that guarantees cleanup on exit (including SIGINT/SIGTERM/SIGKILL).
# Usage: ./tests/run-tests.sh [--args ...]
# Example: ./tests/run-tests.sh -v  (or pass any pytest args)

set -e

cd "$(dirname "$0")/.."

# ── Pre-flight cleanup ──────────────────────────────────────────────
echo "[pre] Killing stale llama-server on test ports…"
for port in $(seq 18000 18031); do
    fuser -k -9 "${port}/tcp" 2>/dev/null || true
done
for port in $(seq 20090 20110); do
    fuser -k -9 "${port}/tcp" 2>/dev/null || true
done

# Remove orphan buildah dirs (can be left by killed pytest sessions)
rm -rf /var/tmp/buildah* 2>/dev/null || true

# ── Run pytest with trap to clean up on ANY exit ────────────────────
trap '
    echo "[post] Killing stale llama-server on test ports…"
    for port in $(seq 18000 18031); do
        fuser -k -9 "${port}/tcp" 2>/dev/null || true
    done
    for port in $(seq 20090 20110); do
        fuser -k -9 "${port}/tcp" 2>/dev/null || true
    done
    rm -rf /var/tmp/buildah* 2>/dev/null || true
    echo "[post] Cleanup done."
' EXIT INT TERM

exec ./venv/bin/pytest "$@"
