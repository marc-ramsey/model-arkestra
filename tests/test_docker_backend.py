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
    runner.arkestra.resolve_config.return_value = None
    runner._inference_kwargs = {}
    # Provide a minimal cm stub so build_model_args can find the model.
    runner.cm.data.get.side_effect = lambda key, default=None: (
        {"test-model": {"args": {"ngl": 0}}, "default": {}}
        if key == "models" else default
    )
    return runner


def _make_podman_runner():
    """Create a properly-configured MagicMock runner for podman tests."""
    runner = MagicMock()
    runner._resolve_image = lambda x: x
    runner.broadcast_addr = "0.0.0.0"
    return runner


# ── Backend resolution (mirrors Podman tests) ────────────────────




# ── Command building (consolidated) ───────────────────────────────

class TestBuildDockerCmdNewArch:
    @pytest.fixture(autouse=True)
    def _patch_isdir(self, monkeypatch):
        monkeypatch.setattr("os.path.isdir", lambda p: p.startswith("/tmp/test-"))

    @pytest.mark.asyncio
    async def test_devices_mounted(self):
        """GPU device nodes are mounted into the container."""
        cfg = {
            "image": "ark-llama:vulkan-radv",
            "devices": ["/dev/dri/card0:rwm", "/dev/dri/renderD128:rwm"],
        }
        runner = _make_docker_runner()
        cmd_parts = _build_container_cmd(
            "docker", runner, "test-model", 18003,
            runner.broadcast_addr, 8080,
            cfg,
        )
        assert "--device" in cmd_parts
        assert "/dev/dri/card0:rwm" in cmd_parts

    @pytest.mark.asyncio
    async def test_port_mapping(self):
        """Docker maps the same port inside and out (no INSIDE_PORT override)."""
        cfg = {"image": "ark-llama:vulkan-radv", "devices": [], "env_container": {}}
        runner = _make_docker_runner()
        cmd_parts = _build_container_cmd(
            "docker", runner, "test-model", 18003,
            runner.broadcast_addr, 8080,
            cfg,
        )
        assert "-p" in cmd_parts
        assert "0.0.0.0:18003:8080" in cmd_parts

    @pytest.mark.asyncio
    async def test_env_container_merged(self):
        """Backend env_container vars are included in the container command."""
        cfg = {
            "image": "ark-llama:vulkan-radv",
            "env_container": {"LLAMA_CACHE": "/data/llama", "HF_HUB_CACHE": "/data/hf"},
        }
        runner = _make_docker_runner()
        cmd_parts = _build_container_cmd(
            "docker", runner, "test-model", 18003,
            runner.broadcast_addr, 8080,
            cfg,
        )
        parts_str = " ".join(cmd_parts)
        assert "LLAMA_CACHE=" in parts_str
        assert "HF_HUB_CACHE=" in parts_str

    @pytest.mark.asyncio
    async def test_binary_dir_mounted(self):
        """Resolved binary directory is mounted read-only into the container."""
        with patch("os.path.isdir", return_value=True):
            cfg = {
                "image": "ark-llama:vulkan-radv",
                "binary_dir": "/tmp/test-wrappers/vulkan-radv",
                "env_container": {},
            }
            runner = _make_docker_runner()
            cmd_parts = _build_container_cmd(
                "docker", runner, "test-model", 18003,
                runner.broadcast_addr, 8080,
                cfg,
            )
            assert "-v" in cmd_parts
            assert "/tmp/test-wrappers/vulkan-radv:/llm-server/bin:ro" in " ".join(cmd_parts)

    @pytest.mark.asyncio
    async def test_container_name_and_host_flag(self):
        """Container gets a named identifier and host binding is applied once."""
        assembled = {"fa": "on", "ngl": "99"}
        with patch("model_arkestra.container_runner.build_model_args", return_value=assembled):
            cfg = {"image": "ark-llama:vulkan-radv", "devices": [], "env_container": {}}
            runner = _make_docker_runner()
            cmd_parts = _build_container_cmd(
                "docker", runner, "test-model", 18003,
                runner.broadcast_addr, 8080,
                cfg,
            )
            assert "--name" in cmd_parts
            assert "llm-test-model-18003" in cmd_parts
            assert "--host" in cmd_parts
            assert "0.0.0.0" in cmd_parts
            # No duplicate --host
            assert cmd_parts.count("--host") <= 1


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
