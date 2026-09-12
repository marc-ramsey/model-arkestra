"""Tests for arkestra CLI — user client (models, chat, init)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Helpers ────────────────────────────────────────────────────────────────

async def _mock_request(conn, path, method="GET", json_body=None):
    if "/api/models" in path:
        return {"models": [
            {"name": "qwen3-4b", "model": "qwen/Qwen3-4B:Q4_K_M", "size": 8.2},
            {"name": "gemma", "model": "unsloth/gemma-4e2b-bnb-4bit", "size": 0},
        ]}
    raise ValueError(f"Unexpected URL: {path}")


def _args(**kw):
    base = {"model": None, "config": None, "url": None,
            "api_key": None, "temperature": None, "top_p": None,
            "max_tokens": None, "frequency_penalty": None,
            "presence_penalty": None, "stop": None, "force": False}
    base.update(kw)
    return type("Args", (), base)()


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
        args = build_parser().parse_args(["models", "-m", "foo"])
        assert args.command == "models"
        assert args.model == "foo"

    def test_chat_requires_model(self):
        from model_arkestra.cli import build_parser
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["chat"])
        args = p.parse_args(["chat", "-m", "my-model"])
        assert args.command == "chat"

    def test_chat_inference_flags(self):
        from model_arkestra.cli import build_parser
        args = build_parser().parse_args(
            ["chat", "-m", "m", "-T", "0.5", "--top-p", "0.9",
             "--max-tokens", "100", "--stop", "END"])
        assert args.temperature == 0.5
        assert args.top_p == 0.9
        assert args.max_tokens == 100
        assert args.stop == "END"

    def test_shared_flags_on_all_subcommands(self):
        from model_arkestra.cli import build_parser
        for sub in ("models", "chat", "init"):
            p = build_parser()
            argv = [sub] + (["-m", "x"] if sub == "chat" else [])
            args = p.parse_args(argv + ["-c", "/x.yaml", "--url", "http://1.2.3.4:1234",
                                        "--api-key", "k"])
            assert args.config == "/x.yaml"
            assert args.url == "http://1.2.3.4:1234"
            assert args.api_key == "k"

    def test_init_force_flag(self):
        from model_arkestra.cli import build_parser
        args = build_parser().parse_args(["init", "--force"])
        assert args.command == "init"
        assert args.force is True


# ── connection resolution ──────────────────────────────────────────────────

class TestMakeConn:
    def test_api_key_from_config(self, tmp_path, monkeypatch):
        from model_arkestra.cli import _make_conn
        cfg = tmp_path / "config.yaml"
        cfg.write_text("default-env:\n  api-key: cfg-key\n")
        monkeypatch.setenv("ARKESTRA_CONFIG", str(cfg))
        conn = _make_conn(_args())
        assert conn.api_key == "cfg-key"

    def test_flag_key_beats_config(self, tmp_path, monkeypatch):
        from model_arkestra.cli import _make_conn
        cfg = tmp_path / "config.yaml"
        cfg.write_text("default-env:\n  api-key: cfg-key\n")
        monkeypatch.setenv("ARKESTRA_CONFIG", str(cfg))
        conn = _make_conn(_args(api_key="flag-key"))
        assert conn.api_key == "flag-key"

    def test_port_from_config(self, tmp_path, monkeypatch):
        from model_arkestra.cli import _make_conn
        cfg = tmp_path / "config.yaml"
        cfg.write_text("default:\n  url: http://h:9999\n")
        monkeypatch.setenv("ARKESTRA_CONFIG", str(cfg))
        conn = _make_conn(_args())
        assert conn.port == 9999


# ── cmd_models ─────────────────────────────────────────────────────────────

class TestCmdModels:
    async def test_list_all_models(self, capsys, tmp_path, monkeypatch):
        from model_arkestra.cli import cmd_models
        cfg = tmp_path / "config.yaml"
        cfg.write_text("default:\n  url: http://127.0.0.1:8080\n")
        monkeypatch.setenv("ARKESTRA_CONFIG", str(cfg))
        with patch("model_arkestra.cli._request", _mock_request):
            await cmd_models(_args())
        captured = capsys.readouterr()
        assert "qwen3-4b" in captured.out
        assert "Qwen3-4B:Q4_K_M" in captured.out

    async def test_single_model_filter(self, capsys, tmp_path, monkeypatch):
        from model_arkestra.cli import cmd_models
        cfg = tmp_path / "config.yaml"
        cfg.write_text("default:\n  url: http://127.0.0.1:8080\n")
        monkeypatch.setenv("ARKESTRA_CONFIG", str(cfg))
        with patch("model_arkestra.cli._request", _mock_request):
            await cmd_models(_args(model="qwen3-4b"))
        captured = capsys.readouterr()
        assert "qwen3-4b" in captured.out
        assert "gemma" not in captured.out


# ── chat /-commands ────────────────────────────────────────────────────────

class TestChatCommands:
    def test_quit(self):
        from model_arkestra.cli import _handle_command
        assert _handle_command("/quit", [], {}) == "quit"
        assert _handle_command("/exit", [], {}) == "quit"

    def test_clear(self, capsys):
        from model_arkestra.cli import _handle_command
        history = [{"role": "user", "content": "hi"}]
        assert _handle_command("/clear", history, {}) == "handled"
        assert history[0]["role"] == "system"
        assert len(history) == 1

    def test_system_prompt(self):
        from model_arkestra.cli import _handle_command
        history = [{"role": "system", "content": "old"}]
        _handle_command("/system new prompt", history, {})
        assert history[0]["content"] == "new prompt"

    def test_param_set_and_show(self, capsys):
        from model_arkestra.cli import _handle_command
        params = {}
        _handle_command("/temperature 0.7", [], params)
        assert params["temperature"] == 0.7
        _handle_command("/top-p 0.9", [], params)
        assert params["top_p"] == 0.9
        _handle_command("/max-tokens 100", [], params)
        assert params["max_tokens"] == 100
        _handle_command("/temperature", [], params)
        assert "0.7" in capsys.readouterr().out

    def test_param_invalid_number(self, capsys):
        from model_arkestra.cli import _handle_command
        params = {}
        _handle_command("/temperature abc", [], params)
        assert "Invalid number" in capsys.readouterr().out
        assert "temperature" not in params

    def test_stop_command(self):
        from model_arkestra.cli import _handle_command
        params = {}
        _handle_command("/stop END", [], params)
        assert params["stop"] == "END"

    def test_plain_message_not_a_command(self):
        from model_arkestra.cli import _handle_command
        assert _handle_command("hello world", [], {}) is None

    def test_unknown_slash_command_is_message(self):
        from model_arkestra.cli import _handle_command
        assert _handle_command("/unknown", [], {}) is None


# ── chat params from args ──────────────────────────────────────────────────

class TestParamsFromArgs:
    def test_collects_set_flags(self):
        from model_arkestra.cli import _params_from_args
        args = _args(temperature=0.5, max_tokens=42, stop="END")
        params = _params_from_args(args)
        assert params == {"temperature": 0.5, "max_tokens": 42, "stop": "END"}

    def test_empty_when_unset(self):
        from model_arkestra.cli import _params_from_args
        assert _params_from_args(_args()) == {}


# ── init ───────────────────────────────────────────────────────────────────

class TestInit:
    def test_scaffolds_both_files(self, tmp_path, monkeypatch):
        from model_arkestra.cli import cmd_init
        monkeypatch.setenv("ARKESTRA_DIR", str(tmp_path))
        with patch("model_arkestra.gpu_detect.detect_all",
                   return_value={"recommendation": ("cpu", "No GPU found")}):
            cmd_init(_args())
        assert (tmp_path / "config.yaml").exists()
        assert (tmp_path / "backends.yaml").exists()
        text = (tmp_path / "config.yaml").read_text()
        assert "default: cpu" in text

    def test_refuses_overwrite_without_force(self, tmp_path, monkeypatch):
        from model_arkestra.cli import cmd_init
        monkeypatch.setenv("ARKESTRA_DIR", str(tmp_path))
        (tmp_path / "config.yaml").write_text("existing")
        with pytest.raises(SystemExit):
            cmd_init(_args())

    def test_force_overwrites(self, tmp_path, monkeypatch):
        from model_arkestra.cli import cmd_init
        monkeypatch.setenv("ARKESTRA_DIR", str(tmp_path))
        (tmp_path / "config.yaml").write_text("existing")
        with patch("model_arkestra.gpu_detect.detect_all", return_value={}):
            cmd_init(_args(force=True))
        assert "existing" not in (tmp_path / "config.yaml").read_text()

    def test_no_detection_still_scaffolds(self, tmp_path, monkeypatch):
        from model_arkestra.cli import cmd_init
        monkeypatch.setenv("ARKESTRA_DIR", str(tmp_path))
        with patch("model_arkestra.gpu_detect.detect_all",
                   side_effect=Exception("no pci")):
            cmd_init(_args())
        assert (tmp_path / "config.yaml").exists()
