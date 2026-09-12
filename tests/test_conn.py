"""Tests for the shared connection surface (model_arkestra.conn)."""
import argparse

import pytest

from model_arkestra.conn import (
    add_common_args, add_server_args, parse_url, resolve_conn,
)

_ENV_VARS = [
    "ARKESTRA_CONFIG", "ARKESTRA_DIR", "ARKESTRA_URL", "ARKESTRA_API_KEY",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _args(**kw):
    base = {"url": None, "api_key": None, "config": None, "bind": None}
    base.update(kw)
    return argparse.Namespace(**base)


class TestParseUrl:
    def test_full(self):
        assert parse_url("http://127.0.0.1:9090/base") == ("http", "127.0.0.1", 9090, "/base")

    def test_https_and_hostcase(self):
        assert parse_url("HTTPS://MyHost:8443/x/") == ("https", "myhost", 8443, "/x")

    def test_default_port_when_omitted(self):
        assert parse_url("http://localhost/base") == ("http", "localhost", 8080, "/base")

    def test_bare_host_no_scheme(self):
        assert parse_url("myhost:9000") == ("http", "myhost", 9000, "")

    def test_empty(self):
        assert parse_url("") == ("http", "127.0.0.1", 8080, "")


class TestResolveConn:
    def test_client_defaults(self):
        c = resolve_conn(_args(), server=False)
        assert c.scheme == "http"
        assert c.host == "127.0.0.1"
        assert c.port == 8080
        assert c.base_path == ""
        assert c.api_key is None
        assert str(c.config_path).endswith(".config/arkestra/config.yaml")

    def test_server_default_bind(self):
        assert resolve_conn(_args(), server=True).bind_host == "127.0.0.1"

    def test_url_flag(self):
        c = resolve_conn(_args(url="http://lanbox:9090/base"), server=False)
        assert (c.scheme, c.host, c.port, c.base_path) == ("http", "lanbox", 9090, "/base")

    def test_env_url(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_URL", "https://h:8443/pfx")
        c = resolve_conn(_args(), server=False)
        assert (c.scheme, c.host, c.port, c.base_path) == ("https", "h", 8443, "/pfx")

    def test_flag_beats_env(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_URL", "http://envhost:1/base")
        c = resolve_conn(_args(url="http://flaghost:2/two"), server=False)
        assert (c.host, c.port, c.base_path) == ("flaghost", 2, "/two")

    def test_cfg_get_provides_url(self):
        c = resolve_conn(_args(), server=False, cfg_get=lambda p, d=None: "http://cfg:7000/cfgpfx")
        assert (c.host, c.port, c.base_path) == ("cfg", 7000, "/cfgpfx")

    def test_env_beats_cfg_get(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_URL", "http://envhost:1/base")
        c = resolve_conn(_args(), server=False, cfg_get=lambda p, d=None: "http://cfg:7000/x")
        assert (c.host, c.port) == ("envhost", 1)

    def test_bind_flag_server_only(self):
        c = resolve_conn(_args(bind="0.0.0.0"), server=True)
        assert c.bind_host == "0.0.0.0"

    def test_bind_from_cfg(self):
        c = resolve_conn(_args(), server=True, cfg_get=lambda p, d=None: "10.0.0.9")
        assert c.bind_host == "10.0.0.9"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_API_KEY", "sekret")
        assert resolve_conn(_args(), server=False).api_key == "sekret"

    def test_api_key_flag_beats_env(self, monkeypatch):
        monkeypatch.setenv("ARKESTRA_API_KEY", "envkey")
        assert resolve_conn(_args(api_key="flagkey"), server=False).api_key == "flagkey"

    def test_prefix_roundtrip_url_for(self):
        c = resolve_conn(_args(url="http://127.0.0.1:9090/base"), server=False)
        assert c.url_for("/v1/chat/completions") == "http://127.0.0.1:9090/base/v1/chat/completions"
        assert c.url_for("admin/models") == "http://127.0.0.1:9090/base/admin/models"
        assert c.url == "http://127.0.0.1:9090/base"


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
    def test_registers_flags(self):
        p = argparse.ArgumentParser()
        add_common_args(p)
        ns = p.parse_args(["--url", "http://h:1/x", "--config", "c", "--api-key", "k"])
        assert (ns.url, ns.config, ns.api_key) == ("http://h:1/x", "c", "k")

    def test_server_args_bind(self):
        p = argparse.ArgumentParser()
        add_common_args(p)
        add_server_args(p)
        ns = p.parse_args(["--url", "http://h:1", "--bind", "0.0.0.0"])
        assert (ns.url, ns.bind) == ("http://h:1", "0.0.0.0")
