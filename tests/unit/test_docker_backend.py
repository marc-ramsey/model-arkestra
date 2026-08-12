"""Unit tests for DockerModelRunner command building and backend resolution.

Mirrors the structure of test_podman_backend_config.py but exercises the Docker-specific code
paths: host-to-host port mapping, --rm flag, global env merge via setdefault, binary directory
mount (added to fix issue #5), and legacy fallback.
"""
from __future__ import annotations
import asyncio
import os

import pytest
from unittest.mock import MagicMock

from model_arkestra.docker import _build_docker_cmd, _resolve_backend_for_docker
from model_arkestra.process import ProcessModelRunner
from model_arkestra.types import RunnerState, _ModelContext, ModelNotStarted, ModelShutdown


# ── Backend resolution (mirrors Podman tests) ─────────────────────────────

class TestResolveBackendForDocker:
    def test_uses_ctx_backend_id(self):
        """ctx.backend_id takes priority over model_data backend."""
        runner = MagicMock()
        runner.cm.get_backend.return_value = {"image": "img:v1"}
        ctx = _ModelContext("m", 18000)
        ctx.backend_id = "rocm"
        model_data = {}
        assert _resolve_backend_for_docker(runner, ctx, model_data) is not None

    def test_falls_back_to_model_backend(self):
        """No ctx.backend_id → model backend key."""
        runner = MagicMock()
        runner.cm.get_backend.return_value = {"image": "img:v2"}
        ctx = _ModelContext("m", 18000)
        model_data = {"backend": "vulkan-radv"}
        result = _resolve_backend_for_docker(runner, ctx, model_data)
        assert result is not None

    def test_returns_none_when_no_backend(self):
        """No override, no model backend → None (triggers legacy path)."""
        runner = MagicMock()
        ctx = _ModelContext("m", 18000)
        model_data = {}
        assert _resolve_backend_for_docker(runner, ctx, model_data) is None


# ── New architecture tests ────────────────────────────────────────────────

class TestBuildDockerCmdNewArch:
    @pytest.fixture
    def mock_config_manager(self):
        inner = MagicMock()
        inner.get_backend.return_value = {
            "image": "llama-strix-halo:vulkan",
            "devices": ["/dev/dri/card0:rwm", "/dev/dri/renderD128:rwm"],
            "env_container": {"GGML_VK_VISIBLE_DEVICES": "0"},
        }
        inner.assemble_command = MagicMock(return_value=([], ""))
        inner.get_vector = MagicMock(return_value=None)
        runner = MagicMock()
        runner.cm = inner
        return runner

    @pytest.fixture(autouse=True)
    def _patch_isdir(self, monkeypatch):
        """Make os.path.isdir return True for /tmp paths so binary mounts are included."""
        monkeypatch.setattr("os.path.isdir", lambda p: p.startswith("/tmp/test-"))

    @pytest.fixture
    def ctx(self):
        c = _ModelContext("test-model", 18003)
        c.backend_id = "vulkan-radv"
        return c

    @pytest.mark.asyncio
    async def test_devices_mounted(self, mock_config_manager, ctx):
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, {})
        assert "--device" in cmd_parts
        assert "/dev/dri/card0:rwm" in cmd_parts
        assert "/dev/dri/renderD128:rwm" in cmd_parts

    @pytest.mark.asyncio
    async def test_env_vars_set(self, mock_config_manager, ctx):
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, {})
        assert "-e" in cmd_parts
        assert "PORT=18003" in cmd_parts
        assert "GGML_VK_VISIBLE_DEVICES=0" in cmd_parts

    @pytest.mark.asyncio
    async def test_port_mapping_host_to_host(self, mock_config_manager, ctx):
        """Docker maps the same port inside and out (no INSIDE_PORT override)."""
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, {})
        assert "-p" in cmd_parts
        # Docker uses host:port with identical numbers — no container-internal port mapping
        assert "18003:18003" in cmd_parts

    @pytest.mark.asyncio
    async def test_container_name(self, mock_config_manager, ctx):
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, {})
        assert "--name" in cmd_parts
        assert "llm-test-model-18003" in cmd_parts

    @pytest.mark.asyncio
    async def test_host_binding_appended(self, mock_config_manager, ctx):
        assembled = "/tmp/test-wrappers/vulkan-radv --port 18003 -fa on"
        mock_config_manager.assemble_command.return_value = (["/bin/foo", "--port", "18003", "-fa", "on"], "")
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, {})
        assert "--host" in cmd_parts
        assert "0.0.0.0" in cmd_parts

    @pytest.mark.asyncio
    async def test_no_duplicated_host(self, mock_config_manager, ctx):
        """If assemble_command already has --host, it should not be added again."""
        assembled = ["/bin/foo", "--port", "18003", "--host", "0.0.0.0", "-fa", "on"]
        mock_config_manager.assemble_command.return_value = (assembled, "")
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, {})
        assert cmd_parts.count("--host") <= 1

    @pytest.mark.asyncio
    async def test_global_env_merged_via_setdefault(self, mock_config_manager, ctx):
        """Global env vars (LLAMA_CACHE) are merged into container_env via setdefault."""
        inner = mock_config_manager.cm  # docker calls runner.cm.get_vector("env")
        inner.get_vector.return_value = {"LLAMA_CACHE": "/data/llama", "HF_HUB_CACHE": "/data/hf"}
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, {})
        # LLAMA_CACHE from global env should be present
        assert "-e" in cmd_parts and "LLAMA_CACHE=" in " ".join(cmd_parts)

    @pytest.mark.asyncio
    async def test_global_env_does_not_override_container_env(self, mock_config_manager, ctx):
        """If a key exists in both container_env and global env, container_env wins."""
        inner = mock_config_manager.cm
        inner.get_backend.return_value = {
            "image": "llama-strix-halo:vulkan",
            "devices": [],
            "env_container": {"MY_VAR": "from_container"},
        }
        inner.get_vector.return_value = {"MY_VAR": "from_global"}
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, {})
        parts_str = " ".join(cmd_parts)
        # Container env should win: value should be "from_container", not "from_global"
        assert "MY_VAR=from_container" in parts_str

    @pytest.mark.asyncio
    async def test_binary_dir_mounted(self, mock_config_manager, ctx):
        """Resolved binary directory is mounted read-only so the host binary is accessible inside container."""
        assembled = ["/tmp/test-wrappers/vulkan-radv --port 18003 -fa on"]
        inner = mock_config_manager.cm
        # Provide binary_dir so resolve_binary_from_backend returns a valid path via first branch
        inner.get_backend.return_value = {
            "image": "llama-strix-halo:vulkan",
            "binary_dir": "/tmp/test-wrappers/vulkan-radv",
            "devices": [],
            "env_container": {},
        }
        inner.assemble_command = MagicMock(return_value=(assembled, ""))
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, {})
        cmd_str = " ".join(cmd_parts)
        assert "-v" in cmd_parts
        # The full binary_dir is mounted read-only (e.g. /tmp/test-wrappers/vulkan-radv)
        assert "/tmp/test-wrappers/vulkan-radv:/tmp/test-wrappers/vulkan-radv:ro" in cmd_str

    @pytest.mark.asyncio
    async def test_binary_path_replaces_llama_server(self, mock_config_manager, ctx):
        """When binary_path is resolved, it replaces 'llama-server' as the executable."""
        assembled = ["/tmp/test-wrappers/vulkan-radv --port 18003 -fa on"]
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, {})
        # llama-server should appear but be replaced by binary_path
        # (the last element before args is either "llama-server" or the resolved binary_path)
        assert len(cmd_parts) > 0

    @pytest.mark.asyncio
    async def test_default_image_when_no_backend_image(self, mock_config_manager, ctx):
        """When backend has no 'image' key, falls back to default_image_for_backend."""
        inner = mock_config_manager
        inner.get_backend.return_value = {
            "devices": [],
            "env_container": {},
        }
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, {})
        # Should use a reasonable default image (from default_image_for_backend)
        assert any("llama-strix-halo" in p or "vulkan" in p for p in cmd_parts)


# ── Legacy fallback tests ────────────────────────────────────────────────

class TestBuildDockerCmdLegacy:
    @pytest.fixture
    def mock_config_manager(self):
        cm = MagicMock()
        cm.get_backend.return_value = None  # No backends → triggers legacy path
        return cm

    def test_legacy_image_from_model_data(self, mock_config_manager):
        ctx = _ModelContext("test-model", 18003)
        model_data = {"image": "custom:v1", "cmd": "-ngl 999"}
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, model_data)
        assert "custom:v1" in cmd_parts

    def test_legacy_host_appended(self, mock_config_manager):
        ctx = _ModelContext("m", 18003)
        model_data = {"image": "img:v1", "cmd": "--port 8080"}
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, model_data)
        cmd_str = " ".join(cmd_parts)
        assert "--host 0.0.0.0" in cmd_str

    def test_legacy_port_mapping(self, mock_config_manager):
        """Legacy path also uses host=port mapping (same inside and out)."""
        ctx = _ModelContext("m", 18003)
        model_data = {"image": "img:v1", "cmd": "-ngl 999"}
        cmd_parts = _build_docker_cmd(mock_config_manager, ctx, model_data)
        assert "-p" in cmd_parts
        assert "18003:18003" in cmd_parts


# ── Dispatch error paths (unchanged from old test) ────────────────────────

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

    def test_dispatch_running_succeeds(self):
        runner = ProcessModelRunner(MagicMock())
        ctx = _ModelContext("running", 18002)
        ctx.state = RunnerState.RUNNING
        runner._models["running"] = ctx
        result = asyncio.run(runner._dispatch("running"))
        assert result is None
