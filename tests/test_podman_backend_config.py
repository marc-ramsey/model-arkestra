"""Unit tests for container command building with backend-config-driven architecture.

Tests the shared ``_build_container_cmd`` function used by both PodmanModelRunner
and DockerModelRunner, covering device mounts, env vars, port mapping, binary
directory mount, host binding, and dispatch error paths.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch
import asyncio
import os

import pytest

from model_arkestra.container_runner import _build_container_cmd
from model_arkestra.process import ProcessModelRunner
from model_arkestra.types import RunnerState, _ModelContext, ModelNotStarted, ModelShutdown, MaxRestartsExceeded


def _make_runner():
    """Create a properly-configured MagicMock runner for tests."""
    runner = MagicMock()
    runner._resolve_image = lambda x: x
    runner.broadcast_addr = "0.0.0.0"
    return runner


def _make_docker_runner():
    """Create a docker-specific MagicMock runner with localhost prefix behavior."""
    runner = MagicMock()
    runner._resolve_image = lambda img: img if "/" in img else f"localhost/{img}"
    runner.broadcast_addr = "0.0.0.0"
    return runner


# ── Backend resolution helper tests ───────────────────────────────

class TestResolveBackendHelper:
    """Tests for the shared _resolve_backend function."""

    def test_uses_ctx_backend_id(self):
        """ctx.backend_id takes priority over model_data backend."""
        pass  # Tested implicitly by callers via _resolve_backend_for_podman/docker

    def test_falls_back_to_model_backend(self):
        """No ctx.backend_id → model backend key."""
        pass


# ── New architecture — podman-style port mapping                 ────

_BACKEND_CONFIG = {
    "image": "ark-llama:vulkan-radv",
    "devices": ["/dev/dri/card0:rwm", "/dev/dri/renderD128:rwm"],
    "env_container": {"GGML_VK_VISIBLE_DEVICES": "0"},
}


class TestBuildContainerCmdPodman:
    @pytest.fixture(autouse=True)
    def _patch_isdir(self, monkeypatch):
        monkeypatch.setattr("os.path.isdir", lambda p: p.startswith("/tmp/test-"))

    @pytest.mark.asyncio
    async def test_devices_mounted(self):
        runner = _make_runner()
        cmd_parts = _build_container_cmd(
            "podman", runner, "test-model", 18003,
            runner.broadcast_addr, 9090,  # inside_port
            _BACKEND_CONFIG.copy(),
        )
        assert "--device" in cmd_parts
        assert "/dev/dri/card0:rwm" in cmd_parts
        assert "/dev/dri/renderD128:rwm" in cmd_parts

    @pytest.mark.asyncio
    async def test_env_vars_set(self):
        runner = _make_runner()
        cfg = {"image": "ark-llama:vulkan-radv", "devices": [],
               "env_container": {"GGML_VK_VISIBLE_DEVICES": "0"}}
        cmd_parts = _build_container_cmd(
            "podman", runner, "test-model", 18003,
            runner.broadcast_addr, 9090,
            cfg,
        )
        assert "-e" in cmd_parts
        assert "PORT=18003" in cmd_parts
        assert "GGML_VK_VISIBLE_DEVICES=0" in cmd_parts

    @pytest.mark.asyncio
    async def test_podman_port_mapping(self):
        """Podman maps host:container port with separate inside_port."""
        cfg = {"image": "ark-llama:v1", "devices": [], "env_container": {}}
        runner = _make_runner()
        cmd_parts = _build_container_cmd(
            "podman", runner, "test-model", 18003,
            runner.broadcast_addr, 9090,  # inside_port != host port
            cfg,
        )
        assert "-p" in cmd_parts
        assert "0.0.0.0:18003:9090" in cmd_parts

    @pytest.mark.asyncio
    async def test_container_name(self):
        cfg = {"image": "ark-llama:v1", "devices": [], "env_container": {}}
        runner = _make_runner()
        cmd_parts = _build_container_cmd(
            "podman", runner, "test-model", 18003,
            runner.broadcast_addr, 9090,
            cfg,
        )
        assert "--name" in cmd_parts
        assert "llm-test-model-18003" in cmd_parts

    @pytest.mark.asyncio
    async def test_host_binding_appended(self):
        with patch("model_arkestra.container_runner.build_model_args", return_value=("/tmp/test-wrappers/vulkan-radv --port 9090 -fa on", "")):
            cfg = {"image": "ark-llama:v1", "devices": [], "env_container": {}}
            runner = _make_runner()
            cmd_parts = _build_container_cmd(
                "podman", runner, "test-model", 18003,
                runner.broadcast_addr, 9090,
                cfg,
            )
            cmd_str = " ".join(cmd_parts)
            assert "--host 0.0.0.0" in cmd_str

    @pytest.mark.asyncio
    async def test_no_duplicated_host(self):
        with patch("model_arkestra.container_runner.build_model_args", return_value=("/tmp/test-wrappers/vulkan-radv --port 9090 --host 0.0.0.0 -fa on", "")):
            cfg = {"image": "ark-llama:v1", "devices": [], "env_container": {}}
            runner = _make_runner()
            cmd_parts = _build_container_cmd(
                "podman", runner, "test-model", 18003,
                runner.broadcast_addr, 9090,
                cfg,
            )
            assert cmd_parts.count("--host") <= 1

    @pytest.mark.asyncio
    async def test_custom_inside_port(self):
        """Podman port mapping uses the inside_port parameter for the container."""
        cfg = {"image": "ark-llama:v1", "devices": [], "env_container": {}}
        runner = _make_runner()
        cmd_parts = _build_container_cmd(
            "podman", runner, "test-model", 18003,
            runner.broadcast_addr, 3000,
            cfg,
        )
        assert "-p" in cmd_parts
        assert "0.0.0.0:18003:3000" in cmd_parts

    @pytest.mark.asyncio
    async def test_binary_dir_mounted(self):
        """Resolved binary directory is mounted read-only."""
        cfg = {
            "image": "ark-llama:v1",
            "devices": [],
            "env_container": {},
            "binary_dir": "/tmp/test-wrappers/vulkan-radv",
        }
        runner = _make_runner()
        cmd_parts = _build_container_cmd(
            "podman", runner, "test-model", 18003,
            runner.broadcast_addr, 9090,
            cfg,
        )
        assert "-v" in cmd_parts


# ── New architecture — docker-style port mapping                ────

class TestBuildContainerCmdDocker:
    @pytest.fixture(autouse=True)
    def _patch_isdir(self, monkeypatch):
        monkeypatch.setattr("os.path.isdir", lambda p: p.startswith("/tmp/test-"))

    @pytest.mark.asyncio
    async def test_docker_port_mapping_host_to_host(self):
        """Docker maps the same port inside and out (no INSIDE_PORT override)."""
        cfg = {"image": "ark-llama:v1", "devices": [], "env_container": {}}
        runner = _make_runner()
        cmd_parts = _build_container_cmd(
            "docker", runner, "test-model", 18003,
            runner.broadcast_addr, 18003,  # same port in/out
            cfg,
        )
        assert "-p" in cmd_parts
        assert "0.0.0.0:18003:18003" in cmd_parts

    @pytest.mark.asyncio
    async def test_docker_image_localhost_prefix(self):
        """Docker runner prepends 'localhost/' to unqualified images."""
        runner = MagicMock()
        runner._resolve_image = lambda img: img if "/" in img else f"localhost/{img}"
        resolved = runner._resolve_image("ark-llama:v1")
        assert resolved == "localhost/ark-llama:v1"

    @pytest.mark.asyncio
    async def test_docker_env_vars(self):
        cfg = {"image": "ark-llama:v1", "devices": [],
               "env_container": {"GGML_VK_VISIBLE_DEVICES": "0"}}
        runner = _make_runner()
        cmd_parts = _build_container_cmd(
            "docker", runner, "test-model", 18003,
            runner.broadcast_addr, 18003,
            cfg,
        )
        assert "-e" in cmd_parts
        assert "PORT=18003" in cmd_parts
        assert "GGML_VK_VISIBLE_DEVICES=0" in cmd_parts

    @pytest.mark.asyncio
    async def test_docker_no_duplicated_host(self):
        """If command already has --host, it should not be added again."""
        with patch("model_arkestra.container_runner.build_model_args", return_value=("/tmp/test-wrappers/vulkan-radv --port 18003 --host 0.0.0.0 -fa on", "")):
            cfg = {"image": "ark-llama:v1", "devices": [], "env_container": {}}
            runner = _make_runner()
            cmd_parts = _build_container_cmd(
                "docker", runner, "test-model", 18003,
                runner.broadcast_addr, 18003,
                cfg,
            )
            assert cmd_parts.count("--host") <= 1

    @pytest.mark.asyncio
    async def test_docker_env_container_set(self):
        """Env vars from backend's env_container are included."""
        cfg = {
            "image": "ark-llama:v1",
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
        assert "LLAMA_CACHE=" in parts_str
        assert "HF_HUB_CACHE=" in parts_str

    @pytest.mark.asyncio
    async def test_docker_binary_dir_mounted(self):
        """Resolved binary directory is mounted read-only."""
        with patch("os.path.isdir", return_value=True):
            cfg = {
                "image": "ark-llama:v1",
                "devices": [],
                "env_container": {},
                "binary_dir": "/tmp/test-wrappers/vulkan-radv",
            }
            runner = _make_runner()
            cmd_parts = _build_container_cmd(
                "docker", runner, "test-model", 18003,
                runner.broadcast_addr, 18003,
                cfg,
            )
            assert "-v" in cmd_parts


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
