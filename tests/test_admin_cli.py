"""Tests for arkestra-admin CLI connection resolution (shared conn surface).

arkestra-admin now resolves its target via the shared ``conn.resolve_conn``:
    --url  >  ARKESTRA_URL env  >  config default.url  >  http://127.0.0.1:8080
and auth via  --api-key  >  ARKESTRA_API_KEY  >  config default-env.admin_key.
The resolved ``Conn`` is attached to ``args.conn`` before dispatch.
"""
from __future__ import annotations

import os
from pathlib import Path
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
# _read_admin_key — default-env section reader
# ═══════════════════════════════════════════════════════════════


class TestReadAdminKey:
    def test_reads_from_default_env_section(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("default-env:\n  admin_key: supersecret\n")
        assert _read_admin_key(str(cfg)) == "supersecret"

    def test_returns_none_when_no_env(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("models:\n  foo: bar\n")
        assert _read_admin_key(str(cfg)) is None


def _make_config(data: dict, tmp_path: Path) -> str:
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return str(p)


def _run_main(argv, env=None):
    """Run admin_cli.main() with dispatch mocked; return the args namespace."""
    captured = {}

    def fake_dispatch(args):
        captured["args"] = args

    with patch("model_arkestra.admin_cli._dispatch", side_effect=fake_dispatch):
        from model_arkestra.admin_cli import main
        if env is not None:
            with patch.dict(os.environ, env, clear=True):
                main(argv)
        else:
            main(argv)
    return captured["args"]


class TestConnResolution:
    """The resolved Conn attached to args.conn reflects the shared precedence."""

    def test_cli_url_takes_precedence(self):
        with patch.dict(os.environ, {"ARKESTRA_API_KEY": "k"}):
            args = _run_main(["--url", "http://custom:9999/base", "models"])
        assert args.conn.host == "custom"
        assert args.conn.port == 9999
        assert args.conn.base_path == "/base"

    def test_env_url_overrides_default(self):
        with patch.dict(os.environ, {"ARKESTRA_URL": "http://remote:7777", "ARKESTRA_API_KEY": "k"}):
            args = _run_main(["models"])
        assert (args.conn.host, args.conn.port) == ("remote", 7777)

    def test_config_default_url_used(self, tmp_path):
        cfg = _make_config({"default": {"url": "http://cfg:9090/pfx"}}, tmp_path)
        with patch.dict(os.environ, {"ARKESTRA_API_KEY": "k"}):
            args = _run_main(["--config", cfg, "models"])
        assert (args.conn.host, args.conn.port, args.conn.base_path) == ("cfg", 9090, "/pfx")

    def test_hardwired_default_when_nothing_set(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        env = {k: v for k, v in os.environ.items() if k not in ("ARKESTRA_URL",)}
        with patch.dict(os.environ, env, clear=True), \
             patch("model_arkestra.admin_cli._read_admin_key", return_value="k"):
            args = _run_main(["--config", cfg, "models"])
        assert (args.conn.host, args.conn.port) == ("127.0.0.1", 8080)

    def test_env_url_overrides_config(self, tmp_path):
        cfg = _make_config({"default": {"url": "http://cfg:9094"}}, tmp_path)
        with patch.dict(os.environ, {"ARKESTRA_URL": "http://from-env:6543", "ARKESTRA_API_KEY": "k"}):
            args = _run_main(["--config", cfg, "models"])
        assert (args.conn.host, args.conn.port) == ("from-env", 6543)


class TestApiKeyResolution:
    def test_api_key_flag(self):
        with patch.dict(os.environ, {}, clear=True):
            args = _run_main(["--api-key", "flagkey", "models"])
        assert args.api_key == "flagkey"

    def test_api_key_env(self):
        with patch.dict(os.environ, {"ARKESTRA_API_KEY": "envkey"}, clear=True):
            args = _run_main(["models"])
        assert args.conn.api_key == "envkey"

    def test_api_key_from_config(self, tmp_path):
        cfg = _make_config({"default-env": {"admin_key": "cfgkey"}}, tmp_path)
        env = {k: v for k, v in os.environ.items() if k != "ARKESTRA_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            args = _run_main(["--config", cfg, "models"])
        assert args.conn.api_key == "cfgkey"

    def test_missing_api_key_exits(self, tmp_path):
        cfg = _make_config({}, tmp_path)
        env = {k: v for k, v in os.environ.items() if k != "ARKESTRA_API_KEY"}
        with patch.dict(os.environ, env, clear=True), \
             pytest.raises(SystemExit):
            _run_main(["--config", cfg, "models"])
