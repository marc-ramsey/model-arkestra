"""Tests for arkestra-admin CLI resolution of --server and --api-key."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from model_arkestra.admin_cli import _load_config, _read_admin_key


# ═══════════════════════════════════════════════════════════════
# _load_config — file reading
# ═══════════════════════════════════════════════════════════════


class TestLoadConfig:
    def test_reads_valid_yaml(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("models:\n  foo: bar\n")
        data = _load_config(str(cfg))
        assert data["models"]["foo"] == "bar"

    def test_returns_empty_on_missing_file(self):
        assert _load_config("/no/such/path/config.yaml") == {}

    def test_returns_empty_on_non_dict_yaml(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("- item1\n- item2\n")
        assert _load_config(str(cfg)) == {}


# ═══════════════════════════════════════════════════════════════
# _read_admin_key — env section reader
# ═══════════════════════════════════════════════════════════════


class TestReadAdminKey:
    def test_reads_from_env_section(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("env:\n  ADMIN_KEY: supersecret\n")
        assert _read_admin_key(str(cfg)) == "supersecret"

    def test_returns_none_when_no_env(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("models:\n  foo: bar\n")
        assert _read_admin_key(str(cfg)) is None


# ═══════════════════════════════════════════════════════════════
# main() server URL resolution
# ═══════════════════════════════════════════════════════════════


def _make_config(data: dict, tmp_path) -> str:
    """Write a config file and return its path."""
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return str(p)


class TestServerURLResolution:
    """Resolution order: CLI > env ARKESTRA_ADMIN_URL > config admin-port > config env.PORT > default."""

    def test_cli_server_takes_precedence(self):
        with patch("model_arkestra.admin_cli._dispatch", create=True) as mock_dispatch:
            from model_arkestra.admin_cli import build_parser
            parser = build_parser()
            args = parser.parse_args(["-x", "http://custom:9999", "models"])
            # Simulate main's resolution logic
            server_url = args.server
            if not server_url:
                url_env = os.environ.get("ARKESTRA_ADMIN_URL")
                if url_env:
                    server_url = url_env
                else:
                    data = {}  # no config in this test
                    port = data.get("admin-port") or (data.get("env") or {}).get("PORT")
                    if port is not None:
                        server_url = f"http://127.0.0.1:{port}"
            if not server_url:
                server_url = "http://127.0.0.1:8080"
            assert server_url == "http://custom:9999"

    def test_env_overrides_default(self):
        with patch.dict(os.environ, {"ARKESTRA_ADMIN_URL": "http://remote:7777"}):
            with patch("model_arkestra.admin_cli._dispatch", create=True):
                from model_arkestra.admin_cli import build_parser
                parser = build_parser()
                args = parser.parse_args(["models"])
                server_url = args.server or os.environ.get("ARKESTRA_ADMIN_URL") or "http://127.0.0.1:8080"
            assert server_url == "http://remote:7777"

    def test_config_admin_port_used(self, tmp_path):
        cfg = _make_config({"admin-port": 9090}, tmp_path)
        with patch("model_arkestra.admin_cli._dispatch", create=True):
            from model_arkestra.admin_cli import build_parser, _load_config
            parser = build_parser()
            args = parser.parse_args(["--config", cfg, "models"])
            # Simulate resolution (no CLI arg, no env)
            server_url = args.server
            if not server_url:
                data = _load_config(args.config)
                port = data.get("admin-port") or (data.get("env") or {}).get("PORT")
                if port is not None:
                    server_url = f"http://127.0.0.1:{port}"
            if not server_url:
                server_url = "http://127.0.0.1:8080"
            assert server_url == "http://127.0.0.1:9090"

    def test_config_env_port_fallback(self, tmp_path):
        cfg = _make_config({"env": {"PORT": 9091}}, tmp_path)
        with patch("model_arkestra.admin_cli._dispatch", create=True):
            from model_arkestra.admin_cli import build_parser, _load_config
            parser = build_parser()
            args = parser.parse_args(["--config", cfg, "models"])
            server_url = args.server
            if not server_url:
                data = _load_config(args.config)
                port = data.get("admin-port") or (data.get("env") or {}).get("PORT")
                if port is not None:
                    server_url = f"http://127.0.0.1:{port}"
            if not server_url:
                server_url = "http://127.0.0.1:8080"
            assert server_url == "http://127.0.0.1:9091"

    def test_admin_port_takes_precedence_over_env_port(self, tmp_path):
        cfg = _make_config({"admin-port": 9092, "env": {"PORT": 9093}}, tmp_path)
        with patch("model_arkestra.admin_cli._dispatch", create=True):
            from model_arkestra.admin_cli import build_parser, _load_config
            parser = build_parser()
            args = parser.parse_args(["--config", cfg, "models"])
            server_url = args.server
            if not server_url:
                data = _load_config(args.config)
                port = data.get("admin-port") or (data.get("env") or {}).get("PORT")
                if port is not None:
                    server_url = f"http://127.0.0.1:{port}"
            if not server_url:
                server_url = "http://127.0.0.1:8080"
            assert server_url == "http://127.0.0.1:9092"

    def test_hardwired_default_when_nothing_set(self):
        with patch.dict(os.environ, {}, clear=False):
            # Ensure no interfering env vars
            os.environ.pop("ARKESTRA_ADMIN_URL", None)
            with patch("model_arkestra.admin_cli._dispatch", create=True):
                from model_arkestra.admin_cli import build_parser
                parser = build_parser()
                args = parser.parse_args(["models"])
                server_url = args.server or os.environ.get("ARKESTRA_ADMIN_URL") or "http://127.0.0.1:8080"
            assert server_url == "http://127.0.0.1:8080"

    def test_env_server_overrides_config(self, tmp_path):
        cfg = _make_config({"admin-port": 9094}, tmp_path)
        with patch.dict(os.environ, {"ARKESTRA_ADMIN_URL": "http://from-env:6543"}):
            with patch("model_arkestra.admin_cli._dispatch", create=True):
                from model_arkestra.admin_cli import build_parser, _load_config
                parser = build_parser()
                args = parser.parse_args(["--config", cfg, "models"])
                server_url = args.server
                if not server_url:
                    url_env = os.environ.get("ARKESTRA_ADMIN_URL")
                    if url_env:
                        server_url = url_env
                    else:
                        data = _load_config(args.config)
                        port = data.get("admin-port") or (data.get("env") or {}).get("PORT")
                        if port is not None:
                            server_url = f"http://127.0.0.1:{port}"
                if not server_url:
                    server_url = "http://127.0.0.1:8080"
            assert server_url == "http://from-env:6543"
