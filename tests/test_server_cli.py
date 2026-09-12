"""Smoke tests for model_arkestra.server CLI entry point — main() resolution logic.

These exercise the if __name__ == "__main__" path that ArkestraServer class tests miss.
No sockets are bound; all subprocess/server startup is mocked out.

Connection is now a single public URL (--url / ARKESTRA_URL / config default.url)
plus a separate bind address (--bind / config default.bind).  The resolved port and
base-path prefix are passed to ArkestraServer as ``port`` and ``base_url``.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml


def _make_config(data: dict, tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return p


def _run_main(argv, cfg, env=None):
    """Run server.main() with ArkestraServer + uvicorn.run mocked; return call kwargs."""
    mock_app = MagicMock()
    with patch("model_arkestra.server.ArkestraServer") as MockAs, \
         patch("model_arkestra.server.uvicorn.run"):
        MockAs.return_value.get_app.return_value = mock_app
        from model_arkestra.server import main
        if env is not None:
            with patch.dict(os.environ, env, clear=True):
                main(argv)
        else:
            main(argv)
    _, kwargs = MockAs.call_args
    return kwargs


class TestPortResolution:
    """Port is the port component of the public URL: CLI > env > config default.url > 8080."""

    def test_cli_url_takes_all_precedence(self, tmp_path):
        cfg = _make_config({"default": {"url": "http://h:9090"}}, tmp_path)
        with patch.dict(os.environ, {"ARKESTRA_URL": "http://h:9091"}):
            kwargs = _run_main(["--config", str(cfg), "--url", "http://h:7777"], cfg)
        assert kwargs["port"] == 7777

    def test_env_url_overrides_config(self, tmp_path):
        cfg = _make_config({"default": {"url": "http://h:9090"}}, tmp_path)
        with patch.dict(os.environ, {"ARKESTRA_URL": "http://h:9091"}):
            kwargs = _run_main(["--config", str(cfg)], cfg)
        assert kwargs["port"] == 9091

    def test_config_url_used(self, tmp_path):
        cfg = _make_config({"default": {"url": "http://h:9092"}}, tmp_path)
        env = {k: v for k, v in os.environ.items() if k != "ARKESTRA_URL"}
        kwargs = _run_main(["--config", str(cfg)], cfg, env=env)
        assert kwargs["port"] == 9092

    def test_hardwired_default_8080(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        env = {k: v for k, v in os.environ.items() if k != "ARKESTRA_URL"}
        kwargs = _run_main(["--config", str(cfg)], cfg, env=env)
        assert kwargs["port"] == 8080


class TestBasePathPrefix:
    """The URL path component becomes the server's base_url prefix."""

    def test_prefix_from_cli_url(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        kwargs = _run_main(["--config", str(cfg), "--url", "http://127.0.0.1:9090/base"], cfg)
        assert kwargs["base_url"] == "/base"

    def test_no_prefix_defaults_empty(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        env = {k: v for k, v in os.environ.items() if k != "ARKESTRA_URL"}
        kwargs = _run_main(["--config", str(cfg)], cfg, env=env)
        assert kwargs["base_url"] == ""


class TestBindResolution:
    """Bind address: CLI --bind > config default.bind > 127.0.0.1."""

    def test_cli_bind(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        kwargs = _run_main(["--config", str(cfg), "--bind", "0.0.0.0"], cfg)
        assert kwargs["bind_host"] == "0.0.0.0"

    def test_config_bind(self, tmp_path):
        cfg = _make_config({"default": {"bind": "10.0.0.9"}}, tmp_path)
        env = {k: v for k, v in os.environ.items() if k != "ARKESTRA_URL"}
        kwargs = _run_main(["--config", str(cfg)], cfg, env=env)
        assert kwargs["bind_host"] == "10.0.0.9"

    def test_default_bind_loopback(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        env = {k: v for k, v in os.environ.items() if k != "ARKESTRA_URL"}
        kwargs = _run_main(["--config", str(cfg)], cfg, env=env)
        assert kwargs["bind_host"] == "127.0.0.1"


class TestTimeoutResolution:
    """Resolution for --ready-timeout: CLI > config warmup-time > 120.0."""

    def test_cli_timeout_overrides_config(self, tmp_path):
        cfg = _make_config({"warmup-time": "30"}, tmp_path)
        kwargs = _run_main(["--config", str(cfg), "--ready-timeout", "45.5"], cfg)
        assert kwargs["ready_timeout"] == 45.5

    def test_config_warmup_time_used(self, tmp_path):
        cfg = _make_config({"default": {"warmup-time": "60"}}, tmp_path)
        kwargs = _run_main(["--config", str(cfg)], cfg)
        assert kwargs["ready_timeout"] == 60.0

    def test_hardwired_default_120(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        kwargs = _run_main(["--config", str(cfg)], cfg)
        assert kwargs["ready_timeout"] == 120.0


class TestNonExistentConfig:
    """Behavior when config file doesn't exist."""

    def test_missing_config_raises(self):
        with patch("model_arkestra.server.uvicorn.run"):
            with pytest.raises(RuntimeError, match="Config file not found"):
                from model_arkestra.server import main
                main(["--config", "/tmp/nonexistent-config.yaml"])
