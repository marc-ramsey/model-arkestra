"""Comprehensive tests for all arkestra-admin sub-commands.

Mock the HTTP layer (_request) and verify each command's output/behaviour.
No real network calls are made.
"""
from __future__ import annotations

import argparse
from unittest.mock import patch, MagicMock

import pytest

import model_arkestra.admin_cli as admin_cli


# ── Helpers ────────────────────────────────────────────────────────

def _mock(return_value):
    """Return an async function that returns *return_value* from _request."""
    async def _inner(*_a, **_kw):
        return return_value
    return _inner


# ════════════════════════════════════════════════════════════════════
# models
# ════════════════════════════════════════════════════════════════════


class TestCmdModels:
    def test_table_output(self, monkeypatch):
        data = {
            "models": [
                {"id": "qwen3-4b", "status": {"value": "loaded"}, "port": 12000, "backend_id": "rocm", "runner_type": "process"},
                {"id": "gemma-4-e2b", "status": {"value": "sleeping"}, "port": None, "backend_id": "vulkan", "runner_type": "podman"},
            ],
        }
        monkeypatch.setattr(admin_cli, "_request", _mock(data))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        async def run():
            await admin_cli.cmd_models(argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False))
        import asyncio
        asyncio.run(run())

        out = captured.getvalue()
        assert "qwen3-4b" in out
        assert "loaded" in out
        assert "12000" in out
        assert "rocm" in out
        assert "gemma-4-e2b" in out

    def test_json_output(self, monkeypatch):
        data = {"models": [{"id": "foo"}]}
        monkeypatch.setattr(admin_cli, "_request", _mock(data))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=True)
        async def run():
            await admin_cli.cmd_models(args)
        import asyncio
        asyncio.run(run())
        assert '"id": "foo"' in captured.getvalue()

    def test_no_models_message(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"models": []}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        async def run():
            await admin_cli.cmd_models(argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False))
        import asyncio
        asyncio.run(run())
        assert "No models in config." in captured.getvalue()


# ════════════════════════════════════════════════════════════════════
# start
# ════════════════════════════════════════════════════════════════════


class TestCmdStart:
    def test_ok_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"ok": True, "port": 12005}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, name="qwen3-4b", port=12005, backend=None, runner=None, args=[])
        async def run():
            await admin_cli.cmd_start(args)
        import asyncio
        asyncio.run(run())
        assert "ok" in captured.getvalue().lower()

    def test_calls_correct_endpoint(self, monkeypatch):
        called_url = []
        async def _req(*_a, **_kw):
            called_url.append(_kw.get("json_body", {}).get("name") or _a[2] if len(_a) > 2 else "?")
            return {"ok": True}
        monkeypatch.setattr(admin_cli, "_request", _req)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, name="gemma-4-e2b", port=13000, backend="rocm", runner=None, args=[])
        async def run():
            await admin_cli.cmd_start(args)
        import asyncio
        asyncio.run(run())
        assert "/admin/start/gemma-4-e2b" in called_url[0]


# ════════════════════════════════════════════════════════════════════
# stop
# ════════════════════════════════════════════════════════════════════


class TestCmdStop:
    def test_ok_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"ok": True, "previous_state": "loaded"}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, name="qwen3-4b")
        async def run():
            await admin_cli.cmd_stop(args)
        import asyncio
        asyncio.run(run())
        assert "stopped" in captured.getvalue()

    def test_not_found_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"ok": False}))

        from io import StringIO
        import sys
        stderr = StringIO()
        monkeypatch.setattr(sys, "stderr", stderr)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, name="missing")
        async def run():
            await admin_cli.cmd_stop(args)
        import asyncio
        asyncio.run(run())
        assert "not found" in stderr.getvalue().lower()


# ════════════════════════════════════════════════════════════════════
# stop-all
# ════════════════════════════════════════════════════════════════════


class TestCmdStopAll:
    def test_ok_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"stopped": ["a", "b"], "message": "Stopped 2 model(s)"}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False)
        async def run():
            await admin_cli.cmd_stop_all(args)
        import asyncio
        asyncio.run(run())
        assert "2" in captured.getvalue()


# ════════════════════════════════════════════════════════════════════
# config list
# ════════════════════════════════════════════════════════════════════


class TestCmdConfigList:
    def test_list_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"models": ["foo", "bar", "baz"]}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, name=None, config_cmd="list")
        async def run():
            await admin_cli.cmd_config_list(args)
        import asyncio
        asyncio.run(run())
        assert "foo" in captured.getvalue()


# ════════════════════════════════════════════════════════════════════
# config get
# ════════════════════════════════════════════════════════════════════


class TestCmdConfigGet:
    def test_get_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({
            "model": "qwen3-4b",
            "status": "loaded",
            "config": {"checkpoint": "foo", "temp": 0.7},
        }))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, name="qwen3-4b", config_cmd="get")
        async def run():
            await admin_cli.cmd_config_get(args)
        import asyncio
        asyncio.run(run())
        out = captured.getvalue()
        assert "qwen3-4b" in out


# ════════════════════════════════════════════════════════════════════
# config set
# ════════════════════════════════════════════════════════════════════


class TestCmdConfigSet:
    def test_set_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"ok": True}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, name="qwen3-4b", args=["temp=0.9"], config_cmd="set")
        async def run():
            await admin_cli.cmd_config_set(args)
        import asyncio
        asyncio.run(run())
        assert "updated" in captured.getvalue()


# ════════════════════════════════════════════════════════════════════
# config create
# ════════════════════════════════════════════════════════════════════


class TestCmdConfigCreate:
    def test_create_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"ok": True, "model": "new-model"}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, name=None, checkpoint="foo:Q4", backend=None, args=[], config_cmd="create")
        async def run():
            await admin_cli.cmd_config_create(args)
        import asyncio
        asyncio.run(run())
        assert "created" in captured.getvalue()


# ════════════════════════════════════════════════════════════════════
# config rm
# ════════════════════════════════════════════════════════════════════


class TestCmdConfigRm:
    def test_rm_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"ok": True}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, name="old-model", config_cmd="rm")
        async def run():
            await admin_cli.cmd_config_rm(args)
        import asyncio
        asyncio.run(run())
        assert "removed" in captured.getvalue()


# ════════════════════════════════════════════════════════════════════
# logs
# ════════════════════════════════════════════════════════════════════


class TestCmdLogs:
    def test_model_logs(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({
            "lines": [{"text": "[info] loaded"}, {"text": "[info] warmed up"}],
            "missed_lines": 0, "seq": 5,
        }))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, name="qwen3-4b", lines=100)
        async def run():
            await admin_cli.cmd_logs(args)
        import asyncio
        asyncio.run(run())
        assert "[info]" in captured.getvalue()

    def test_all_logs(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"lines": [{"text": "global line"}], "missed_lines": 0, "seq": 1}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, name="all", lines=100)
        async def run():
            await admin_cli.cmd_logs(args)
        import asyncio
        asyncio.run(run())
        assert "global line" in captured.getvalue()


# ════════════════════════════════════════════════════════════════════
# eject
# ════════════════════════════════════════════════════════════════════


class TestCmdEject:
    def test_ok_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"ok": True}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, name="qwen3-4b")
        async def run():
            await admin_cli.cmd_eject(args)
        import asyncio
        asyncio.run(run())
        assert "ejected" in captured.getvalue()


# ════════════════════════════════════════════════════════════════════
# images list
# ════════════════════════════════════════════════════════════════════


class TestCmdImagesList:
    def test_table_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock([
            {"backend_id": "rocm", "runner": "podman", "image": "docker.io/foo/rocm:v1", "available": True},
            {"backend_id": "vulkan", "runner": "process", "image": None, "available": False},
        ]))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, image_cmd="list")
        async def run():
            await admin_cli.cmd_images_list(args)
        import asyncio
        asyncio.run(run())
        out = captured.getvalue()
        assert "rocm" in out


# ════════════════════════════════════════════════════════════════════
# images build
# ════════════════════════════════════════════════════════════════════


class TestCmdImagesBuild:
    def test_build_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"ok": True, "image": "rocm:v2", "skipped": False}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, backend="rocm", image_cmd="build", tag=None)
        async def run():
            await admin_cli.cmd_images_build(args)
        import asyncio
        asyncio.run(run())
        assert "built" in captured.getvalue()


# ════════════════════════════════════════════════════════════════════
# images rm
# ════════════════════════════════════════════════════════════════════


class TestCmdImagesRm:
    def test_rm_output(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"removed": True}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False, tag="docker.io/foo:v1", image_cmd="rm")
        async def run():
            await admin_cli.cmd_images_rm(args)
        import asyncio
        asyncio.run(run())
        assert "removed" in captured.getvalue()


# ════════════════════════════════════════════════════════════════════
# shutdown
# ════════════════════════════════════════════════════════════════════


class TestCmdShutdown:
    def test_shutdown_response(self, monkeypatch):
        monkeypatch.setattr(admin_cli, "_request", _mock({"message": "Server shutting down"}))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False)
        async def run():
            await admin_cli.cmd_shutdown(args)
        import asyncio
        asyncio.run(run())
        assert "shutting down" in captured.getvalue().lower()

    def test_shutdown_connection_error(self, monkeypatch):
        """cmd_shutdown uses aiohttp directly — mock ClientSession to raise."""
        from unittest.mock import AsyncMock
        # Patch at the module level where admin_cli imported it
        async def _raise(*_a, **_kw):
            raise ConnectionError("Connection refused")
        monkeypatch.setattr(admin_cli, "ClientSession", AsyncMock(side_effect=ConnectionError("refused")))

        from io import StringIO
        import sys
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        args = argparse.Namespace(server="http://localhost:9090", api_key="secret", config=None, json=False)
        async def run():
            await admin_cli.cmd_shutdown(args)
        import asyncio
        asyncio.run(run())
        assert "Shutdown signal sent" in captured.getvalue()


# ════════════════════════════════════════════════════════════════════
# _dispatch routing — verify every sub-command reaches its handler
# ════════════════════════════════════════════════════════════════════


class TestDispatchRouting:
    """Verify that each known command routes to the correct handler."""

    def test_all_commands_dispatch(self, monkeypatch):
        called = {}
        for name in ["models", "start", "stop", "stop-all", "logs", "eject", "shutdown"]:
            async def fake(_a):
                called[name] = True
            monkeypatch.setattr(admin_cli, f"cmd_{name.replace('-', '_')}", fake)

        args = argparse.Namespace(command="models")  # default to one; will be overridden by per-test patching
        import asyncio

        for name in ["models", "start", "stop", "stop-all", "logs", "eject", "shutdown"]:
            called.clear()
            args.command = name
            asyncio.run(admin_cli._dispatch(args))
            assert name in called, f"Command '{name}' was not dispatched"
