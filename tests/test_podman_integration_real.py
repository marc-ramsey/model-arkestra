"""Integration tests for PodmanModelRunner with real containers.

These tests spin up actual podman containers — they are marked
``@pytest.mark.slow`` and will skip if podman is unavailable.
"""

from __future__ import annotations
import asyncio
import os
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
    """Kill any leftover llm-* podman containers from previous runs.

    Handles rootless podman pasta networking — lsof alone misses the pasta
    processes that hold onto host ports after a container is removed.
    """
    import re as _re

    # 1. Kill pasta listeners on any port (rootless networking)
    result = subprocess.run(
        ["ss", "-tlnp"], capture_output=True, text=True, timeout=5
    )
    for line in result.stdout.split("\n"):
        if "pasta" in line:
            m = _re.search(r"pid=(\d+)", line)
            if m:
                try:
                    os.kill(int(m.group(1)), 9)
                except OSError:
                    pass

    # 2. Remove all llm-* containers
    result = subprocess.run(
        [
            "podman", "ps", "-a", "--filter", "name=llm-",
            "--format", "{{.ID}}",
        ],
        capture_output=True, text=True,
    )
    for cid in result.stdout.strip().split():
        if cid:
            subprocess.run(["podman", "rm", "-f", cid], capture_output=True, timeout=10)

    # 3. Kill any remaining processes on the test port range
    for port in range(18000, 18032):
        result = subprocess.run(["lsof", f"-ti:{port}"], capture_output=True, text=True)
        for pid in result.stdout.strip().split():
            if pid:
                try:
                    os.kill(int(pid), 9)
                except OSError:
                    pass

    # Wait so killed listeners release their file descriptors.
    import time as _time
    _time.sleep(0.5)


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

    async def test_gemma_4_e2b_podman_stream(self, mr):
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

    async def test_podman_logs_captured(self, mr):
        """Start qwen3.5-4b in podman → verify container logs are captured."""
        await mr.start("qwen3.5-4b", runner="podman")

        # Find the podman runner that hosts qwen
        podman_runner = None
        for r in mr._runners.values():
            if "qwen3.5-4b" in r._models:
                podman_runner = r
                break
        assert podman_runner, "Podman runner should have qwen3.5-4b"

        logs = await podman_runner.get_logs("qwen3.5-4b", lines=100)
        assert len(logs) > 0, "No log lines captured from podman container"
        combined = "\n".join(logs).lower()
        # Should contain podman/docker container runtime output
        has_container_ref = any(
            kw in combined for kw in ["podman", "docker", "ark-llama",
                                       "image", "pull", "create"]
        )
        assert has_container_ref, (
            f"No container/runtime output in logs: {logs[:5]}"
        )
