"""Tests for the shared connection surface (model_arkestra.conn)."""
import argparse

import pytest

from model_arkestra.conn import add_common_args, resolve_conn

_ENV_VARS = [
    "ARKESTRA_CONFIG", "ARKESTRA_DIR", "ARKESTRA_HOST",
    "ARKESTRA_PORT", "ARKESTRA_API_KEY", "ARKESTRA_BASE_PATH",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _args(**kw):
    base = {"host": None, "port": None, "api_key": None, "config": None}
    base.update(kw)
    return argparse.Namespace(**base)


class TestResolveConn:
    def test_client_defaults(self):
        c = resolve_conn(_args(), server=False)
        assert c.host == "127.0.0.1"
        assert c.port == 8080
        assert c.base_path == ""
        assert c.api_key is None
        assert str(c.config_path).endswith(".config/arkestra/config.yaml")

    def test_server_defaults_host(self):
        assert resolve_conn(_args(), server=True).host == "0.0.0.0"

    def test_env_host_port(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_HOST", "10.0.0.5")
        monkeypatch.setenv("ARKESTRA_PORT", "9999")
        c = resolve_conn(_args(), server=False)
        assert c.host == "10.0.0.5"
        assert c.port == 9999

    def test_flag_beats_env(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_HOST", "10.0.0.5")
        monkeypatch.setenv("ARKESTRA_PORT", "9999")
        c = resolve_conn(_args(host="1.2.3.4", port=1111), server=False)
        assert c.host == "1.2.3.4"
        assert c.port == 1111

    def test_cfg_get_provides_port(self):
        c = resolve_conn(_args(), server=False, cfg_get=lambda p, d: 1234)
        assert c.port == 1234

    def test_env_beats_cfg_get(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_PORT", "2222")
        c = resolve_conn(_args(), server=False, cfg_get=lambda p, d: 1234)
        assert c.port == 2222

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_API_KEY", "sekret")
        assert resolve_conn(_args(), server=False).api_key == "sekret"

    def test_api_key_flag_beats_env(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_API_KEY", "envkey")
        assert resolve_conn(_args(api_key="flagkey"), server=False).api_key == "flagkey"

    def test_base_path_env(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_BASE_PATH", "/prefix")
        c = resolve_conn(_args(host="h"), server=False)
        assert c.base_path == "/prefix"
        assert c.url_for("/v1/chat/completions") == "http://h:8080/prefix/v1/chat/completions"


class TestConfigPath:
    def test_dir_env(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_DIR", "/tmp/arak")
        assert str(resolve_conn(_args(), server=False).config_path) == "/tmp/arak/config.yaml"

    def test_full_env_beats_dir(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_DIR", "/tmp/arak")
        monkeypatch.setenv("ARKESTRA_CONFIG", "/tmp/custom.yaml")
        assert str(resolve_conn(_args(), server=False).config_path) == "/tmp/custom.yaml"

    def test_config_flag_beats_env(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_CONFIG", "/tmp/custom.yaml")
        assert str(resolve_conn(_args(config="/tmp/flag.yaml"), server=False).config_path) == "/tmp/flag.yaml"


class TestAddCommonArgs:
    def test_registers_long_flags(self):
        p = argparse.ArgumentParser()
        add_common_args(p)
        ns = p.parse_args(["--host", "h", "--port", "5", "--config", "c", "--api-key", "k"])
        assert (ns.host, ns.port, ns.config, ns.api_key) == ("h", 5, "c", "k")

    def test_registers_short_flags(self):
        p = argparse.ArgumentParser()
        add_common_args(p)
        ns = p.parse_args(["-H", "h", "-p", "7", "-c", "c"])
        assert (ns.host, ns.port, ns.config) == ("h", 7, "c")
