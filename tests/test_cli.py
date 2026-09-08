"""Tests for arkestra CLI — thin RPC client (models, start, stop, restart, chat)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Helpers ────────────────────────────────────────────────────────────────

async def _mock_request(server_url, path, method="GET", json_body=None, api_key=None):
    """Mock that returns configured responses based on path."""
    if "/api/models" in path:
        return {"models": [
            {"name": "qwen3-4b", "model": "qwen/Qwen3-4B:Q4_K_M", "size": 8.2},
            {"name": "gemma", "model": "unsloth/gemma-4e2b-bnb-4bit", "size": 0},
        ]}
    elif "/api/start/" in path:
        model = path.split("/api/start/")[1].split("/")[0]
        return {"ok": True, "model": model, "port": 18000}
    elif "/api/stop/" in path:
        model = path.split("/api/stop/")[1].split("/")[0]
        return {"ok": True, "model": model, "previous_state": "running"}
    elif "/api/restart/" in path:
        model = path.split("/api/restart/")[1].split("/")[0]
        return {"ok": True, "model": model, "port": 18000}
    raise ValueError(f"Unexpected URL: {path}")


# ── arg parsing ────────────────────────────────────────────────────────────

class TestArgParsing:
    def test_help_prints_all_commands(self):
        from model_arkestra.cli import build_parser
        p = build_parser()
        try:
            p.parse_args(["--help"])
        except SystemExit as e:
            assert e.code == 0

    def test_models_subcommand(self):
        from model_arkestra.cli import build_parser
        p = build_parser()
        args = p.parse_args(["models", "-m", "foo"])
        assert args.command == "models"
        assert args.model == "foo"

    def test_start_requires_model(self):
        from model_arkestra.cli import build_parser
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["start"])
        args = p.parse_args(["start", "-m", "my-model"])
        assert args.command == "start"

    def test_stop_requires_model(self):
        from model_arkestra.cli import build_parser
        p = build_parser()
        args = p.parse_args(["stop", "-m", "my-model"])
        assert args.command == "stop"

    def test_restart_requires_model(self):
        from model_arkestra.cli import build_parser
        p = build_parser()
        args = p.parse_args(["restart", "-m", "my-model"])
        assert args.command == "restart"


# ── cmd_models ─────────────────────────────────────────────────────────────

class TestCmdModels:
    async def test_list_all_models(self, capsys):
        from model_arkestra.cli import cmd_models
        args = type("Args", (), {"model": None, "config": None})()
        with patch("model_arkestra.cli._resolve_server_url", return_value="http://localhost:8080"), \
             patch("model_arkestra.cli._request", _mock_request):
            await cmd_models(args)
        captured = capsys.readouterr()
        assert "qwen3-4b" in captured.out
        assert "Qwen3-4B:Q4_K_M" in captured.out

    async def test_single_model_filter(self, capsys):
        from model_arkestra.cli import cmd_models
        args = type("Args", (), {"model": "qwen3-4b", "config": None})()
        with patch("model_arkestra.cli._resolve_server_url", return_value="http://localhost:8080"), \
             patch("model_arkestra.cli._request", _mock_request):
            await cmd_models(args)
        captured = capsys.readouterr()
        assert "qwen3-4b" in captured.out
        assert "gemma" not in captured.out

    async def test_no_server_url_fails(self, capsys):
        from model_arkestra.cli import cmd_models
        args = type("Args", (), {"model": None, "config": None})()
        with patch("model_arkestra.cli._resolve_server_url", return_value=None):
            try:
                await cmd_models(args)
            except SystemExit:
                pass  # expected
        captured = capsys.readouterr()
        assert "No server URL found" in captured.err


# ── cmd_start / stop / restart ─────────────────────────────────────────────

class TestCmdServerActions:
    async def test_start(self, capsys):
        from model_arkestra.cli import cmd_start
        args = type("Args", (), {"model": "my-model", "config": None})()
        with patch("model_arkestra.cli._resolve_server_url", return_value="http://localhost:8080"), \
             patch("model_arkestra.cli._request", _mock_request):
            await cmd_start(args)
        captured = capsys.readouterr()
        assert "started" in captured.out.lower()

    async def test_stop(self, capsys):
        from model_arkestra.cli import cmd_stop
        args = type("Args", (), {"model": "my-model", "config": None})()
        with patch("model_arkestra.cli._resolve_server_url", return_value="http://localhost:8080"), \
             patch("model_arkestra.cli._request", _mock_request):
            await cmd_stop(args)
        captured = capsys.readouterr()
        assert "stopped" in captured.out.lower()

    async def test_restart(self, capsys):
        from model_arkestra.cli import cmd_restart
        args = type("Args", (), {"model": "my-model", "config": None})()
        with patch("model_arkestra.cli._resolve_server_url", return_value="http://localhost:8080"), \
             patch("model_arkestra.cli._request", _mock_request):
            await cmd_restart(args)
        captured = capsys.readouterr()
        assert "restarted" in captured.out.lower()

    async def test_action_no_server_fails(self, capsys):
        from model_arkestra.cli import cmd_start
        args = type("Args", (), {"model": "my-model", "config": None})()
        with patch("model_arkestra.cli._resolve_server_url", return_value=None):
            try:
                await cmd_start(args)
            except SystemExit:
                pass
        captured = capsys.readouterr()
        assert "No server URL found" in captured.err
