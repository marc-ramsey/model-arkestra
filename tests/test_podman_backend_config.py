"""Unit tests for backend-config-driven podman command building."""

from __future__ import annotations
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import os

import pytest

from model_arkestra.podman import _build_podman_cmd, _resolve_backend_for_podman
from model_arkestra.process import ProcessModelRunner
from model_arkestra.types import RunnerState, _ModelContext, ModelNotStarted, ModelShutdown, MaxRestartsExceeded


# ── Backend resolution ──────────────────────────────────────────────────────

class TestResolveBackend:
    def test_uses_ctx_backend_id(self):
        """ctx.backend_id takes priority."""
        runner = MagicMock()
        runner.cm.get_backend.return_value = {"image": "img:v1"}
        ctx = _ModelContext("m", 18000)
        ctx.backend_id = "rocm"
        model_data = {}
        assert _resolve_backend_for_podman(runner, ctx, model_data) is not None

    def test_falls_back_to_model_backend(self):
        """No ctx.backend_id → model backend key."""
        runner = MagicMock()
        runner.cm.get_backend.return_value = {"image": "img:v2"}
        ctx = _ModelContext("m", 18000)
        model_data = {"backend": "vulkan-radv"}
        result = _resolve_backend_for_podman(runner, ctx, model_data)
        assert result is not None

    def test_returns_none_when_no_backend(self):
        """No override, no model backend → None (legacy path)."""
        runner = MagicMock()
        ctx = _ModelContext("m", 18000)
        model_data = {}
        assert _resolve_backend_for_podman(runner, ctx, model_data) is None


# ── New architecture — wrapper present ───────────────────────────────────────

class TestBuildPodmanCmdNewArch:
    @pytest.fixture
    def mock_config_manager(self):
        inner = MagicMock()
        inner.get_backend.return_value = {
            "image": "llama-strix-halo:vulkan",
            "devices": ["/dev/dri/card0:rwm", "/dev/dri/renderD128:rwm"],
            "env_container": {"GGML_VK_VISIBLE_DEVICES": "0"},
        }
        inner.assemble_command = MagicMock(return_value=([], ""))
        runner = MagicMock()
        runner.cm = inner
        runner.INSIDE_PORT = 9091
        return runner

    @pytest.fixture(autouse=True)
    def _patch_isfile(self, monkeypatch):
        monkeypatch.setattr("os.path.isfile", lambda p: p.startswith("/tmp/test-"))

    @pytest.fixture
    def ctx(self):
        c = _ModelContext("test-model", 18003)
        c.backend_id = "vulkan-radv"
        return c

    @pytest.mark.asyncio
    async def test_devices_mounted(self, mock_config_manager, ctx):
        cmd_parts = _build_podman_cmd(mock_config_manager, ctx, {})
        assert "--device" in cmd_parts
        assert "/dev/dri/card0:rwm" in cmd_parts
        assert "/dev/dri/renderD128:rwm" in cmd_parts

    @pytest.mark.asyncio
    async def test_env_vars_set(self, mock_config_manager, ctx):
        cmd_parts = _build_podman_cmd(mock_config_manager, ctx, {})
        assert "-e" in cmd_parts
        assert "PORT=18003" in cmd_parts
        assert "GGML_VK_VISIBLE_DEVICES=0" in cmd_parts

    @pytest.mark.asyncio
    async def test_port_mapping(self, mock_config_manager, ctx):
        cmd_parts = _build_podman_cmd(mock_config_manager, ctx, {})
        assert "-p" in cmd_parts
        assert "18003:9091" in cmd_parts

    @pytest.mark.asyncio
    async def test_container_name(self, mock_config_manager, ctx):
        cmd_parts = _build_podman_cmd(mock_config_manager, ctx, {})
        assert "--name" in cmd_parts
        assert "llm-test-model-18003" in cmd_parts

    @pytest.mark.asyncio
    async def test_host_binding_appended(self, mock_config_manager, ctx):
        assembled = "/tmp/test-wrappers/vulkan-radv --port 18003 -fa on"
        mock_config_manager.assemble_command.return_value = assembled
        cmd_parts = _build_podman_cmd(mock_config_manager, ctx, {})
        cmd_str = " ".join(cmd_parts)
        assert "--host 0.0.0.0" in cmd_str

    @pytest.mark.asyncio
    async def test_no_duplicated_host(self, mock_config_manager, ctx):
        """If assemble_command already has --host, it should not be added again."""
        assembled = "/tmp/test-wrappers/vulkan-radv --port 18003 --host 0.0.0.0 -fa on"
        mock_config_manager.assemble_command.return_value = assembled
        cmd_parts = _build_podman_cmd(mock_config_manager, ctx, {})
        assert cmd_parts.count("--host") <= 1

    @pytest.mark.asyncio
    async def test_custom_container_port(self, mock_config_manager, ctx):
        model_data = {"container_port": 9091}
        cmd_parts = _build_podman_cmd(mock_config_manager, ctx, model_data)
        assert "-p" in cmd_parts
        assert "18003:9091" in cmd_parts


# ── Legacy fallback (no backends section) ───────────────────────────────────

class TestBuildPodmanCmdLegacy:
    @pytest.fixture
    def mock_config_manager(self):
        cm = MagicMock()
        # No backends → get_backend returns None
        cm.get_backend.return_value = None
        return cm

    def test_legacy_image_from_model_data(self, mock_config_manager):
        ctx = _ModelContext("test-model", 18003)
        model_data = {"image": "custom:v1", "cmd": "-ngl 999"}
        cmd_parts = _build_podman_cmd(mock_config_manager, ctx, model_data)
        assert "custom:v1" in cmd_parts

    def test_legacy_host_appended(self, mock_config_manager):
        ctx = _ModelContext("m", 18003)
        model_data = {"image": "img:v1", "cmd": "--port 8080"}
        cmd_parts = _build_podman_cmd(mock_config_manager, ctx, model_data)
        cmd_str = " ".join(cmd_parts)
        assert "--host 0.0.0.0" in cmd_str

    def test_legacy_custom_port(self, mock_config_manager):
        ctx = _ModelContext("m", 18003)
        model_data = {"image": "img:v1", "cmd": "foo", "container_port": 3000}
        cmd_parts = _build_podman_cmd(mock_config_manager, ctx, model_data)
        assert "-p" in cmd_parts
        assert "18003:3000" in cmd_parts


# ── Dispatch error paths (unchanged from old test) ──────────────────────────

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
