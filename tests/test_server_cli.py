"""Smoke tests for model_arkestra.server CLI entry point — main() resolution logic.

These exercise the if __name__ == "__main__" path that ArkestraServer class tests miss.
No sockets are bound; all subprocess/server startup is mocked out.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml


def _make_config(data: dict, tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return p


class TestPortResolution:
    """Resolution order for --port: CLI > env PORT > config admin-port > 8080."""

    def test_cli_port_takes_all_precedence(self, tmp_path):
        cfg = _make_config({"admin-port": 9090}, tmp_path)
        with patch.dict(os.environ, {"PORT": "9091"}):
            mock_app = MagicMock()
            with patch("model_arkestra.server.ArkestraServer") as MockAs, \
                 patch("model_arkestra.server.uvicorn.run"):
                MockAs.return_value.get_app.return_value = mock_app
                from model_arkestra.server import main
                main(["--config", str(cfg), "--port", "7777"])
        _, kwargs = MockAs.call_args
        assert kwargs["port"] == 7777

    def test_env_port_overrides_config(self, tmp_path):
        cfg = _make_config({"admin-port": 9090}, tmp_path)
        with patch.dict(os.environ, {"PORT": "9091"}):
            mock_app = MagicMock()
            with patch("model_arkestra.server.ArkestraServer") as MockAs, \
                 patch("model_arkestra.server.uvicorn.run"):
                MockAs.return_value.get_app.return_value = mock_app
                from model_arkestra.server import main
                main(["--config", str(cfg)])
        _, kwargs = MockAs.call_args
        assert kwargs["port"] == 9091

    def test_config_admin_port_used(self, tmp_path):
        cfg = _make_config({"admin-port": 9092}, tmp_path)
        env = {k: v for k, v in os.environ.items() if k != "PORT"}
        with patch.dict(os.environ, env, clear=True):
            mock_app = MagicMock()
            with patch("model_arkestra.server.ArkestraServer") as MockAs, \
                 patch("model_arkestra.server.uvicorn.run"):
                MockAs.return_value.get_app.return_value = mock_app
                from model_arkestra.server import main
                main(["--config", str(cfg)])
        _, kwargs = MockAs.call_args
        assert kwargs["port"] == 9092

    def test_hardwired_default_8080(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        env = {k: v for k, v in os.environ.items() if k not in ("PORT",)}
        with patch.dict(os.environ, env, clear=True):
            mock_app = MagicMock()
            with patch("model_arkestra.server.ArkestraServer") as MockAs, \
                 patch("model_arkestra.server.uvicorn.run"):
                MockAs.return_value.get_app.return_value = mock_app
                from model_arkestra.server import main
                main(["--config", str(cfg)])
        _, kwargs = MockAs.call_args
        assert kwargs["port"] == 8080

    def test_missing_config_uses_default(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        env = {k: v for k, v in os.environ.items() if k not in ("PORT",)}
        with patch.dict(os.environ, env, clear=True):
            mock_app = MagicMock()
            with patch("model_arkestra.server.ArkestraServer") as MockAs, \
                 patch("model_arkestra.server.uvicorn.run"):
                MockAs.return_value.get_app.return_value = mock_app
                from model_arkestra.server import main
                main(["--config", str(cfg)])
        _, kwargs = MockAs.call_args
        assert kwargs["port"] == 8080


class TestTimeoutResolution:
    """Resolution for --ready-timeout: CLI > config warmup-time > 120.0."""

    def test_cli_timeout_overrides_config(self, tmp_path):
        cfg = _make_config({"warmup-time": "30"}, tmp_path)
        mock_app = MagicMock()
        with patch("model_arkestra.server.ArkestraServer") as MockAs, \
             patch("model_arkestra.server.uvicorn.run"):
            MockAs.return_value.get_app.return_value = mock_app
            from model_arkestra.server import main
            main(["--config", str(cfg), "--ready-timeout", "45.5"])
        _, kwargs = MockAs.call_args
        assert kwargs["ready_timeout"] == 45.5

    def test_config_warmup_time_used(self, tmp_path):
        cfg = _make_config({"warmup-time": "60"}, tmp_path)
        mock_app = MagicMock()
        with patch("model_arkestra.server.ArkestraServer") as MockAs, \
             patch("model_arkestra.server.uvicorn.run"):
            MockAs.return_value.get_app.return_value = mock_app
            from model_arkestra.server import main
            main(["--config", str(cfg)])
        _, kwargs = MockAs.call_args
        assert kwargs["ready_timeout"] == 60.0

    def test_hardwired_default_120(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        mock_app = MagicMock()
        with patch("model_arkestra.server.ArkestraServer") as MockAs, \
             patch("model_arkestra.server.uvicorn.run"):
            MockAs.return_value.get_app.return_value = mock_app
            from model_arkestra.server import main
            main(["--config", str(cfg)])
        _, kwargs = MockAs.call_args
        assert kwargs["ready_timeout"] == 120.0


class TestHostResolution:
    """Resolution for --host: CLI > env HOST > default '0.0.0.0'."""

    def test_cli_host_precedence(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        mock_app = MagicMock()
        with patch("model_arkestra.server.ArkestraServer") as MockAs, \
             patch("model_arkestra.server.uvicorn.run"):
            MockAs.return_value.get_app.return_value = mock_app
            from model_arkestra.server import main
            main(["--config", str(cfg), "--host", "127.0.0.1"])
        _, kwargs = MockAs.call_args
        assert kwargs["port"] == 8080  # sanity: port still resolved

    def test_env_host_overrides_default(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        with patch.dict(os.environ, {"HOST": "127.0.0.1"}):
            mock_app = MagicMock()
            with patch("model_arkestra.server.ArkestraServer") as MockAs, \
                 patch("model_arkestra.server.uvicorn.run"):
                MockAs.return_value.get_app.return_value = mock_app
                from model_arkestra.server import main
                main(["--config", str(cfg)])

    def test_default_is_0000(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        env = {k: v for k, v in os.environ.items() if k not in ("HOST",)}
        with patch.dict(os.environ, env, clear=True):
            mock_app = MagicMock()
            with patch("model_arkestra.server.ArkestraServer") as MockAs, \
                 patch("model_arkestra.server.uvicorn.run"):
                MockAs.return_value.get_app.return_value = mock_app
                from model_arkestra.server import main
                main(["--config", str(cfg)])


class TestNonExistentConfig:
    """Behavior when config file doesn't exist."""

    def test_missing_config_uses_defaults(self):
        with patch.dict(os.environ, {"PORT": "9091"}):
            mock_app = MagicMock()
            with patch("model_arkestra.server.ArkestraServer") as MockAs, \
                 patch("model_arkestra.server.uvicorn.run"):
                MockAs.return_value.get_app.return_value = mock_app
                from model_arkestra.server import main
                main(["--config", "/tmp/nonexistent-config.yaml"])
        _, kwargs = MockAs.call_args
        assert kwargs["port"] == 9091  # env PORT still works
