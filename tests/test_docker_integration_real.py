"""Integration tests for DockerModelRunner with real containers.

These tests spin up actual docker containers — they are marked
``@pytest.mark.slow`` and will skip if docker is unavailable.
"""

from __future__ import annotations
import os
import subprocess

import pytest


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _cleanup_zombies() -> None:
    """Kill any leftover llm-* docker containers from previous runs."""
    # Remove all llm-* containers
    result = subprocess.run(
        [
            "docker", "ps", "-a", "--filter", "name=llm-",
            "--format", "{{.ID}}",
        ],
        capture_output=True, text=True,
    )
    for cid in result.stdout.strip().split():
        if cid:
            subprocess.run(
                ["docker", "rm", "-f", cid],
                capture_output=True, timeout=10,
            )

    # Kill any remaining processes on the test port range
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
@pytest.mark.skipif(not _docker_available(), reason="docker not available")
class TestDockerRunnerIntegration:
    """Full end-to-end DockerModelRunner tests with real container."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Clean up leftover containers before/after each test."""
        _cleanup_zombies()
        yield
        _cleanup_zombies()

    async def test_qwen3_4b_docker_invoke(self, mr):
        """Start qwen3.5-4b in docker container → invoke → stop."""
        await mr.start("qwen3.5-4b", runner="docker")

        response = await mr.ainvoke(
            "qwen3.5-4b",
            "What is the capital of France? Answer in 5 words.",
        )
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"[<] Response: {response[:120]}…")

    async def test_gemma_4_e2b_docker_stream(self, mr):
        """Start gemma-4-e2b in docker container → stream tokens → stop."""
        await mr.start("gemma-4-e2b", runner="docker")

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

    async def test_docker_logs_captured(self, mr):
        """Start qwen3.5-4b in docker → verify container logs are captured."""
        await mr.start("qwen3.5-4b", runner="docker")

        # Find the docker runner that hosts qwen
        docker_runner = None
        for r in mr._runners.values():
            if "qwen3.5-4b" in r._models:
                docker_runner = r
                break
        assert docker_runner, "Docker runner should have qwen3.5-4b"

        logs = await docker_runner.get_logs("qwen3.5-4b", lines=100)
        assert len(logs) > 0, "No log lines captured from docker container"
        combined = "\n".join(logs).lower()
        # Should contain docker/container runtime output
        has_container_ref = any(
            kw in combined for kw in ["docker", "ark-llama",
                                       "image", "pull", "create"]
        )
        assert has_container_ref, (
            f"No container/runtime output in logs: {logs[:5]}"
        )
