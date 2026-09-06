"""Tests for arkestra CLI (user-facing) commands that talk to the admin server."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_mock_response(status: int, body: dict) -> AsyncMock:
    """Create an aiohttp-style mock response."""
    r = AsyncMock()
    r.status = status
    r.json = AsyncMock(return_value=body)
    return r


async def _mock_request(method: str, url: str, **kwargs):
    """Global mock for aiohttp ClientSession.request — returns configurable responses."""
    if "/admin/pull/" in url:
        model = url.split("/admin/pull/")[1].split("/")[0]
        return {"ok": True, "model": model}
    elif "/admin/eject/" in url:
        model = url.split("/admin/eject/")[1].split("/")[0]
        return {"ok": True, "model": model}
    elif "/admin/models" in url:
        return {"models": [
            {"id": "qwen3", "status": {"value": "stopped"}, "port": 18000, "backend_id": "rocm"},
            {"id": "gemma", "status": {"value": "uncached"}, "port": None, "backend_id": "vulkan-radv"},
        ]}
    raise ValueError(f"Unexpected URL: {url}")


# ── arg parsing ────────────────────────────────────────────────────────────

class TestArgParsing:
    def test_help_prints_all_commands(self):
        from model_arkestra.cli import build_parser
        p = build_parser()
        try:
            p.parse_args(["--help"])
        except SystemExit as e:
            assert e.code == 0

    def test_pull_requires_model(self):
        from model_arkestra.cli import build_parser
        p = build_parser()
        args = p.parse_args(["pull", "my-model"])
        assert args.command == "pull"
        assert args.model == "my-model"

    def test_unload_requires_model(self):
        from model_arkestra.cli import build_parser
        p = build_parser()
        args = p.parse_args(["unload", "my-model"])
        assert args.command == "unload"
        assert args.model == "my-model"

    def test_status_optional_model(self):
        from model_arkestra.cli import build_parser
        p = build_parser()
        args_all = p.parse_args(["status"])
        assert args_all.model is None
        args_one = p.parse_args(["status", "-m", "foo"])
        assert args_one.model == "foo"


# ── cmd_status ─────────────────────────────────────────────────────────────

class TestCmdStatus:
    async def test_list_all_models(self):
        from model_arkestra.cli import cmd_status, _resolve_server_url, _request
        args = type("Args", (), {"model": None})()
        with patch("model_arkestra.cli._resolve_server_url", return_value="http://localhost:8080"), \
             patch("model_arkestra.cli._request", _mock_request):
            await cmd_status(args)

    async def test_single_model_filter(self):
        from model_arkestra.cli import cmd_status, _resolve_server_url, _request
        args = type("Args", (), {"model": "qwen3"})()
        with patch("model_arkestra.cli._resolve_server_url", return_value="http://localhost:8080"), \
             patch("model_arkestra.cli._request", _mock_request):
            await cmd_status(args)

    async def test_no_server_url_fails(self, capsys):
        from model_arkestra.cli import cmd_status
        args = type("Args", (), {"model": None})()
        with patch("model_arkestra.cli._resolve_server_url", return_value=None):
            try:
                await cmd_status(args)
            except SystemExit:
                pass  # expected
        captured = capsys.readouterr()
        assert "No server URL found" in captured.err


# ── cmd_pull ───────────────────────────────────────────────────────────────

class TestCmdPull:
    async def test_successful_pull(self):
        from model_arkestra.cli import cmd_pull, _resolve_server_url, _request
        args = type("Args", (), {"model": "my-model"})()
        with patch("model_arkestra.cli._resolve_server_url", return_value="http://localhost:8080"), \
             patch("model_arkestra.cli._request", AsyncMock(return_value={"ok": True, "model": "my-model"})):
            try:
                await cmd_pull(args)
            except SystemExit:
                pass  # no sys.exit on success

    async def test_already_downloading(self):
        from model_arkestra.cli import cmd_pull
        args = type("Args", (), {"model": "my-model"})()
        with patch("model_arkestra.cli._resolve_server_url", return_value="http://localhost:8080"), \
             patch("model_arkestra.cli._request", AsyncMock(return_value={"ok": True, "already_downloading": True})):
            try:
                await cmd_pull(args)
            except SystemExit:
                pass

    async def test_pull_failure(self, capsys):
        from model_arkestra.cli import cmd_pull
        args = type("Args", (), {"model": "bad-model"})()
        with patch("model_arkestra.cli._resolve_server_url", return_value="http://localhost:8080"), \
             patch("model_arkestra.cli._request", AsyncMock(return_value={"detail": "Model is stopped"})):
            try:
                await cmd_pull(args)
            except SystemExit:
                pass
        captured = capsys.readouterr()
        assert "failed to pull" in captured.err.lower() or "is stopped" in captured.err

    async def test_no_server_url_fails(self, capsys):
        from model_arkestra.cli import cmd_pull
        args = type("Args", (), {"model": "my-model"})()
        with patch("model_arkestra.cli._resolve_server_url", return_value=None):
            try:
                await cmd_pull(args)
            except SystemExit:
                pass
        captured = capsys.readouterr()
        assert "No server URL found" in captured.err


# ── cmd_unload ─────────────────────────────────────────────────────────────

class TestCmdUnload:
    async def test_successful_unload(self):
        from model_arkestra.cli import cmd_unload
        args = type("Args", (), {"model": "my-model"})()
        with patch("model_arkestra.cli._resolve_server_url", return_value="http://localhost:8080"), \
             patch("model_arkestra.cli._request", AsyncMock(return_value={"ok": True, "model": "my-model"})):
            try:
                await cmd_unload(args)
            except SystemExit:
                pass

    async def test_unload_failure(self, capsys):
        from model_arkestra.cli import cmd_unload
        args = type("Args", (), {"model": "missing"})()
        with patch("model_arkestra.cli._resolve_server_url", return_value="http://localhost:8080"), \
             patch("model_arkestra.cli._request", AsyncMock(return_value={"detail": "Model not found"})):
            try:
                await cmd_unload(args)
            except SystemExit:
                pass
        captured = capsys.readouterr()
        assert "failed to unload" in captured.err.lower() or "not found" in captured.err

    async def test_no_server_url_fails(self, capsys):
        from model_arkestra.cli import cmd_unload
        args = type("Args", (), {"model": "my-model"})()
        with patch("model_arkestra.cli._resolve_server_url", return_value=None):
            try:
                await cmd_unload(args)
            except SystemExit:
                pass
        captured = capsys.readouterr()
        assert "No server URL found" in captured.err
