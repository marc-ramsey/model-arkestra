"""Integration tests - spin up real llama-server instances via runner classes."""

from __future__ import annotations
import subprocess

import pytest


# ------------------------------------------------------------------ #
# Helpers for runner availability                                    #
# ------------------------------------------------------------------ #

def _podman_available() -> bool:
    try:
        result = subprocess.run(
            ["podman", "info"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ------------------------------------------------------------------ #
# Tests: ProcessModelRunner - real HTTP round-trip                   #
# ------------------------------------------------------------------ #

class TestProcessModelRunnerIntegration:
    """Start model -> health check -> request -> stop."""

    @pytest.mark.slow
    async def test_ainvoke(self, mr):
        """Start qwen3.5-4b -> call ainvoke -> get response -> stop."""
        await mr.start("qwen3.5-4b")
        result = await mr.ainvoke("qwen3.5-4b", "say hi")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.slow
    async def test_astream(self, mr):
        """Start gemma-4-e2b -> stream tokens -> stop."""
        await mr.start("gemma-4-e2b")
        chunks = []
        async for chunk in mr.astream("gemma-4-e2b", {"prompt": "say hi"}):
            chunks.append(chunk)

        assert len(chunks) > 0
        last = chunks[-1]
        assert "usage" in last or any("token" in c for c in chunks)

    @pytest.mark.slow
    async def test_too_many_models(self, mr):
        """Start qwen + gemma (both fit), then voxtral should fail on port exhaustion."""
        await mr.start("qwen3.5-4b")
        await mr.start("gemma-4-e2b")

        with pytest.raises(RuntimeError):
            await mr.start("voxtral-mini")


# ------------------------------------------------------------------ #
# Tests: docker - conditionally run                                  #
# ------------------------------------------------------------------ #

class TestDockerIntegration:
    """Tests for DockerModelRunner. SKIPPED if docker unavailable."""

    @pytest.fixture(scope="function")
    def _skip_no_docker(self):
        if not _docker_available():
            pytest.skip("docker not available")

    @pytest.mark.slow
    async def test_docker_invoke(self, mr, _skip_no_docker):
        """Start qwen3.5-4b via docker -> call ainvoke -> stop."""
        await mr.start("qwen3.5-4b", runner="docker")
        result = await mr.ainvoke("qwen3.5-4b", "say hi")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.slow
    async def test_docker_stream(self, mr, _skip_no_docker):
        """Start gemma-4-e2b via docker -> stream tokens -> stop."""
        await mr.start("gemma-4-e2b", runner="docker")
        chunks = []
        async for chunk in mr.astream(
            "gemma-4-e2b", {"prompt": "say hi"}
        ):
            chunks.append(chunk)
        assert len(chunks) > 0
        last = chunks[-1]
        assert "usage" in last or any("token" in c for c in chunks)
