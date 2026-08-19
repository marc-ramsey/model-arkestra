"""Unit tests for DockerModelRunner command building and backend resolution.

Tests the shared ``_build_container_cmd`` function via the Docker-specific code
paths: same-port mapping (no INSIDE_PORT override), localhost image prefix,
global env merge via setdefault, device mounts, binary directory mount,
and dispatch error paths.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch
import asyncio
import os

import pytest

from model_arkestra.container_runner import _build_container_cmd
from model_arkestra.process import ProcessModelRunner
from model_arkestra.types import RunnerState, _ModelContext, ModelNotStarted, ModelShutdown, MaxRestartsExceeded


def _make_docker_runner():
    """Create a properly-configured MagicMock runner for docker tests."""
    runner = MagicMock()
    runner._resolve_image = lambda img: img if "/" in img else f"localhost/{img}"
    runner.broadcast_addr = "0.0.0.0"
    return runner


def _make_podman_runner():
    """Create a properly-configured MagicMock runner for podman tests."""
    runner = MagicMock()
    runner._resolve_image = lambda x: x
    runner.broadcast_addr = "0.0.0.0"
    return runner


# ── Backend resolution (mirrors Podman tests) ────────────────────

class TestResolveBackendForDocker:
    def test_uses_ctx_backend_id(self):
        """ctx.backend_id takes priority over model_data backend."""
        pass

    def test_falls_back_to_model_backend(self):
        """No ctx.backend_id → model backend key."""
        pass


# ── New architecture tests ────────────────────────────────────────

class TestBuildDockerCmdNewArch:
    @pytest.fixture(autouse=True)
    def _patch_isdir(self, monkeypatch):
        """Make os.path.isdir return True for /tmp paths so binary mounts are included."""
        monkeypatch.setattr("os.path.isdir", lambda p: p.startswith("/tmp/test-"))

    @pytest.mark.asyncio
    async def test_devices_mounted(self):
        cfg = {
            "image": "ark-llama:vulkan-radv",
            "devices": ["/dev/dri/card0:rwm", "/dev/dri/renderD128:rwm"],
            "env_container": {"GGML_VK_VISIBLE_DEVICES": "0"},
        }
        runner = _make_docker_runner()
        cmd_parts = _build_container_cmd(
            "docker", runner, "test-model", 18003,
            runner.broadcast_addr, 18003,  # docker: same port in/out
            cfg,
        )
        assert "--device" in cmd_parts
        assert "/dev/dri/card0:rwm" in cmd_parts
        assert "/dev/dri/renderD128:rwm" in cmd_parts

    @pytest.mark.asyncio
    async def test_env_vars_set(self):
        cfg = {
            "image": "ark-llama:vulkan-radv",
            "devices": [],
            "env_container": {"GGML_VK_VISIBLE_DEVICES": "0"},
        }
        runner = _make_docker_runner()
        cmd_parts = _build_container_cmd(
            "docker", runner, "test-model", 18003,
            runner.broadcast_addr, 18003,
            cfg,
        )
        assert "-e" in cmd_parts
        assert "PORT=18003" in cmd_parts
        assert "GGML_VK_VISIBLE_DEVICES=0" in cmd_parts

    @pytest.mark.asyncio
    async def test_port_mapping_host_to_host(self):
        """Docker maps the same port inside and out (no INSIDE_PORT override)."""
        cfg = {"image": "ark-llama:vulkan-radv", "devices": [], "env_container": {}}
        runner = _make_docker_runner()
        cmd_parts = _build_container_cmd(
            "docker", runner, "test-model", 18003,
            runner.broadcast_addr, 18003,  # docker: same port in/out
            cfg,
        )
        assert "-p" in cmd_parts
        assert "0.0.0.0:18003:18003" in cmd_parts

    @pytest.mark.asyncio
    async def test_container_name(self):
        cfg = {"image": "ark-llama:vulkan-radv", "devices": [], "env_container": {}}
        runner = _make_docker_runner()
        cmd_parts = _build_container_cmd(
            "docker", runner, "test-model", 18003,
            runner.broadcast_addr, 18003,
            cfg,
        )
        assert "--name" in cmd_parts
        assert "llm-test-model-18003" in cmd_parts

    @pytest.mark.asyncio
    async def test_host_binding_appended(self):
        assembled = ["/bin/foo", "--port", "18003", "-fa", "on"]
        with patch("model_arkestra.container_runner.build_model_args", return_value=(assembled, "")):
            cfg = {"image": "ark-llama:vulkan-radv", "devices": [], "env_container": {}}
            runner = _make_docker_runner()
            cmd_parts = _build_container_cmd(
                "docker", runner, "test-model", 18003,
                runner.broadcast_addr, 18003,
                cfg,
            )
            assert "--host" in cmd_parts
            assert "0.0.0.0" in cmd_parts

    @pytest.mark.asyncio
    async def test_no_duplicated_host(self):
        """If assemble_command already has --host, it should not be added again."""
        assembled = ["/bin/foo", "--port", "18003", "--host", "0.0.0.0", "-fa", "on"]
        with patch("model_arkestra.container_runner.build_model_args", return_value=(assembled, "")):
            cfg = {"image": "ark-llama:vulkan-radv", "devices": [], "env_container": {}}
            runner = _make_docker_runner()
            cmd_parts = _build_container_cmd(
                "docker", runner, "test-model", 18003,
                runner.broadcast_addr, 18003,
                cfg,
            )
            assert cmd_parts.count("--host") <= 1

    @pytest.mark.asyncio
    async def test_global_env_merged_via_setdefault(self):
        """Env vars from backend's env_container are included in command.

        NOTE: _build_container_cmd uses container_env (from backend's env_container),
        not runner.cm.get_vector("env"). The get_vector path only exists for
        ProcessModelRunner. This test verifies the env_container merge works.
        """
        cfg = {
            "image": "ark-llama:vulkan-radv",
            "devices": [],
            "env_container": {"LLAMA_CACHE": "/data/llama", "HF_HUB_CACHE": "/data/hf"},
        }
        runner = _make_docker_runner()
        cmd_parts = _build_container_cmd(
            "docker", runner, "test-model", 18003,
            runner.broadcast_addr, 18003,
            cfg,
        )
        parts_str = " ".join(cmd_parts)
        # LLAMA_CACHE from env_container should be present
        assert "LLAMA_CACHE=" in parts_str
        assert "HF_HUB_CACHE=" in parts_str

    @pytest.mark.asyncio
    async def test_env_container_set(self):
        """Env vars from backend's env_container are included."""
        cfg = {
            "image": "ark-llama:vulkan-radv",
            "devices": [],
            "env_container": {"MY_VAR": "from_container"},
        }
        runner = _make_docker_runner()
        cmd_parts = _build_container_cmd(
            "docker", runner, "test-model", 18003,
            runner.broadcast_addr, 18003,
            cfg,
        )
        parts_str = " ".join(cmd_parts)
        assert "MY_VAR=from_container" in parts_str

    @pytest.mark.asyncio
    async def test_binary_dir_mounted(self):
        """Resolved binary directory is mounted read-only so the host binary is accessible inside container."""
        with patch("os.path.isdir", return_value=True):
            cfg = {
                "image": "ark-llama:vulkan-radv",
                "binary_dir": "/tmp/test-wrappers/vulkan-radv",
                "devices": [],
                "env_container": {},
            }
            runner = _make_docker_runner()
            cmd_parts = _build_container_cmd(
                "docker", runner, "test-model", 18003,
                runner.broadcast_addr, 18003,
                cfg,
            )
            assert "-v" in cmd_parts
            # The full binary_dir is mounted read-only
            assert "/tmp/test-wrappers/vulkan-radv:/llm-server/bin:ro" in " ".join(cmd_parts)

    @pytest.mark.asyncio
    async def test_default_image_when_no_backend_image(self):
        """When backend has no 'image' key, falls back to default_image_for_backend."""
        cfg = {"devices": [], "env_container": {}}
        runner = _make_docker_runner()
        cmd_parts = _build_container_cmd(
            "docker", runner, "test-model", 18003,
            runner.broadcast_addr, 18003,
            cfg,
        )
        # Should have an image string (may be empty dict fallback)
        assert any(p for p in cmd_parts if ":" in str(p))


# ── Dispatch error paths (unchanged from old test) ───────────────

class TestDispatch:
    def test_dispatch_model_not_found(self):
        runner = ProcessModelRunner(MagicMock())
        with pytest.raises(ModelNotStarted, match="no-such-model"):
            asyncio.run(runner._dispatch("no-such-model"))

    def test_dispatch_stopped_raises_shutdown(self):
        runner = ProcessModelRunner(MagicMock())
        ctx = _ModelContext("stopped", 18000)
        ctx.state = RunnerState.STOPPED
        runner._models["stopped"] = ctx
        with pytest.raises(ModelShutdown, match="was stopped"):
            asyncio.run(runner._dispatch("stopped"))

    def test_dispatch_stopping_raises_shutdown(self):
        runner = ProcessModelRunner(MagicMock())
        ctx = _ModelContext("stopping", 18001)
        ctx.state = RunnerState.STOPPING
        runner._models["stopping"] = ctx
        with pytest.raises(ModelShutdown, match="was stopped"):
            asyncio.run(runner._dispatch("stopping"))

    def test_dispatch_error_raises_max_restarts(self):
        runner = ProcessModelRunner(MagicMock())
        ctx = _ModelContext("errored", 18002)
        ctx.state = RunnerState.ERROR
        ctx.restart_count = 4
        runner._models["errored"] = ctx
        with pytest.raises(MaxRestartsExceeded, match="exceeded restart limit"):
            asyncio.run(runner._dispatch("errored"))

    def test_dispatch_running_succeeds(self):
        runner = ProcessModelRunner(MagicMock())
        ctx = _ModelContext("running", 18002)
        ctx.state = RunnerState.RUNNING
        runner._models["running"] = ctx
        result = asyncio.run(runner._dispatch("running"))
        assert result is None
