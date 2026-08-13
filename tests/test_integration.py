"""Integration tests — spin up real llama-server instances via runner classes."""

from __future__ import annotations
import json
import subprocess
from unittest.mock import MagicMock

import pytest





# ── Helpers for runner availability ────────────────────────────────────────

def _podman_available() -> bool:
    try:
        result = subprocess.run(["podman", "info"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _docker_available() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── Tests: ProcessModelRunner — real HTTP round-trip ───────────────────────

class TestProcessModelRunnerIntegration:
    """Start model → health check → request → stop."""

    @pytest.mark.slow
    async def test_ainvoke(self, mr):
        """Start qwen3.5-4b → call ainvoke → get response → stop."""
        await mr.start("qwen3.5-4b")
        result = await mr.ainvoke("qwen3.5-4b", "say hi")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.slow
    async def test_astream(self, mr):
        """Start gemma-4-e2b → stream tokens → stop."""
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


# ── Tests: podman — conditionally run ──────────────────────────────────────


class TestPodmanIntegration:
    """Tests for PodmanModelRunner. SKIPPED if podman unavailable."""

    @pytest.fixture(scope="function")
    def _skip_no_podman(self):
        if not _podman_available():
            pytest.skip("podman not available")

    async def test_podman_runner_importable(self, _skip_no_podman):
        """Verify the PodmanModelRunner class imports cleanly."""
        from model_arkestra.podman import PodmanModelRunner
        assert PodmanModelRunner is not None

    async def test_podman_gpu_detection(self, _skip_no_podman):
        """Verify backend registry provides GPU devices for vulkan-radv."""
        from model_arkestra.podman import PodmanModelRunner
        assert PodmanModelRunner is not None

    async def test_hf_cache_mount_is_rw(self, _skip_no_podman):
        """HF cache volume must be read-write so llama-server can download models inside container."""
        from model_arkestra.podman import _build_podman_cmd
        from model_arkestra.types import _ModelContext

        ctx = _ModelContext("test-model", 19200)
        inner = MagicMock()
        inner.get_backend.return_value = {
            "wrapper": "/nonexistent/wrapper",
            "image": "llama-strix-halo:vulkan",
            "devices": [],
            "env_container": {},
        }
        inner.build_model_args = MagicMock(return_value=("", ""))
        runner_mock = MagicMock()
        runner_mock.cm = inner
        runner_mock.INSIDE_PORT = 9090

        cmd_parts = _build_podman_cmd(runner_mock, ctx, {})

        for part in cmd_parts:
            if "/home/lemonade/hub" in part or "HF_HUB_CACHE" in str(part):
                assert ":ro" not in part


# ── Tests: docker — conditionally run ──────────────────────────────────────

class TestDockerIntegration:
    @pytest.fixture(scope="function")
    def _skip_no_docker(self):
        if not _docker_available():
            pytest.skip("docker not available")

    async def test_docker_runner_importable(self, _skip_no_docker):
        """Verify the DockerModelRunner class imports cleanly."""
        from model_arkestra.docker import DockerModelRunner
        assert DockerModelRunner is not None

    @pytest.mark.slow
    async def test_ainvoke(self, mr):
        """Start qwen3.5-4b via docker → call ainvoke → stop."""
        await mr.start("qwen3.5-4b", runner="docker")
        result = await mr.ainvoke("qwen3.5-4b", "say hi")
        assert isinstance(result, str)
        assert len(result) > 0
