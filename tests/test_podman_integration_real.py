"""Integration tests for PodmanModelRunner with real containers.

These tests spin up actual podman containers — they are marked
``@pytest.mark.slow`` and will skip if podman is unavailable.
"""

from __future__ import annotations
import asyncio
import subprocess

import pytest


def _podman_available() -> bool:
    try:
        result = subprocess.run(
            ["podman", "info"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _cleanup_zombies() -> None:
    """Kill any leftover llm-* podman containers from previous runs."""
    result = subprocess.run(
        [
            "podman", "ps", "-a", "--filter", "name=llm-",
            "--format", "{{.ID}}",
        ],
        capture_output=True, text=True,
    )
    for cid in result.stdout.strip().split():
        if cid:
            subprocess.run(["podman", "rm", "-f", cid], capture_output=True)
    # Wait so pasta listeners release their ports.
    import time as _time
    _time.sleep(0.3)


@pytest.mark.slow
@pytest.mark.skipif(not _podman_available(), reason="podman not available")
class TestPodmanRunnerIntegration:
    """Full end-to-end PodmanModelRunner tests with real container."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Clean up leftover containers before/after each test."""
        _cleanup_zombies()
        yield
        _cleanup_zombies()

    async def test_qwen3_4b_podman_invoke(self, mr):
        """Start qwen3.5-4b in podman container → invoke → stop."""
        await mr.start("qwen3.5-4b", runner="podman")

        response = await mr.ainvoke(
            "qwen3.5-4b",
            "What is the capital of France? Answer in 5 words.",
        )
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"[<] Response: {response[:120]}…")

    async def test_qwen3_4b_podman_stream(self, mr):
        """Start gemma-4-e2b in podman container → stream tokens → stop."""
        await mr.start("gemma-4-e2b", runner="podman")

        print("[*] Testing streaming (astream)...")
        full_content = ""
        async for chunk in mr.astream(
            "gemma-4-e2b", {"prompt": "Say hi!"}
        ):
            if "token" in chunk and chunk["token"]:
                print(chunk["token"], end="", flush=True)
                full_content += chunk["token"]
            elif "usage" in chunk:
                u = chunk["usage"]
                print(f"Done ({u.get('total_tokens', '?')} tokens, {u.get('tokens_per_second', '?')} tok/s)")
        print()
