"""Tests for plugin loading — real ArkestraServer, no mocks.

Covers: dir- and file-style entries, warn+skip failure policy, mount prefix,
api_key gating inherited from the /api/* middleware, register(ctx) contract
(narrow surface — no raw app handle), name validation (identifier + duplicate).
"""
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from model_arkestra.server import ArkestraServer

ADMIN_KEY = "test-admin-key"
API_KEY = "test-api-key"


def _admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


# ── Plugin fixtures on disk (written once per module) ─────────────

_TMPDIR = tempfile.mkdtemp(prefix="arkestra-plugins-")
PLUGINS_DIR = Path(_TMPDIR) / "plugins"
PLUGINS_DIR.mkdir()


def _write(rel: str, body: str):
    path = PLUGINS_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


# Dir-style plugin with a router + register hook. The module records what it
# was handed so tests can assert on the narrow surface afterwards.
_write("stats/plugin.py", """\
from fastapi import APIRouter

router = APIRouter()

def register(ctx):
    globals()["saw_ctx"] = ctx

@router.get("/hello")
async def hello():
    return {"ok": True}
""")

# Dir-style plugin: register hook only, no router. Records the models view.
_write("uptime/plugin.py", """\
def register(ctx):
    globals()["saw_models"] = dict(ctx.models)
""")

# Broken plugin: raises on import (SyntaxError)
_write("broken/plugin.py", "1/0\n")

# Plugin whose register() raises → must be recorded as error and its router
# must NOT be mounted.
_write("badsetup/plugin.py", """\
from fastapi import APIRouter

router = APIRouter()

@router.get("/hello")
async def hello():
    return {"ok": True}

def register(ctx):
    raise RuntimeError("boom in setup")
""")

# File-style entry — name is the file stem ("filestyle")
_write("filestyle.py", """\
from fastapi import APIRouter

router = APIRouter()

@router.get("/ping")
async def ping():
    return {"pong": True}
""")


_CONFIG_TEMPLATE = """\
default:
  model-start-port: 18300
  model-ports: 2

env:
  admin-key: {admin_key}
  api-key: {api_key}

backends:
  vulkan-radv:
    runner: podman
    image: ark-llama:vulkan-radv

models:
  test-model:
    model: unsloth/Qwen3.5-4B-GGUF:Q4_K_M
    backend: vulkan-radv

plugins:
{plugin_lines}
"""


def _write_config(tmpdir: str, plugin_lines: str) -> str:
    cfg = os.path.join(tmpdir, "config.yaml")
    Path(cfg).write_text(_CONFIG_TEMPLATE.format(
        admin_key=ADMIN_KEY, api_key=API_KEY, plugin_lines=plugin_lines))
    return cfg


@pytest.fixture(scope="module")
def live_server():
    """Real ArkestraServer with plugins: two good, one broken import."""
    lines = "\n".join(f"  - {PLUGINS_DIR / name}" for name in
                      ("stats", "uptime", "broken"))
    cfg_path = _write_config(_TMPDIR, lines)
    server = ArkestraServer(cfg_path, port=18006)
    yield {"server": server, "client": TestClient(server.get_app())}
    shutil.rmtree(_TMPDIR, ignore_errors=True)


# ── Loading behaviour ─────────────────────────────────────────────

class TestPluginLoading:

    def test_states(self, live_server):
        r = live_server["client"].get("/admin/plugins", headers=_admin_headers())
        assert r.status_code == 200
        by_name = {p["name"]: p for p in r.json()["plugins"]}
        assert by_name["stats"]["state"] == "ok"
        assert by_name["uptime"]["state"] == "ok"
        assert by_name["broken"]["state"] == "error"

    def test_broken_error_captures_reason(self, live_server):
        r = live_server["client"].get("/admin/plugins", headers=_admin_headers())
        broken = next(p for p in r.json()["plugins"] if p["name"] == "broken")
        assert "import failed" in broken["error"]

    def test_broken_plugin_did_not_kill_others(self, live_server):
        """Server came up and healthy routes work despite a broken plugin."""
        assert live_server["client"].get("/health").status_code == 200


# ── Router mounting ───────────────────────────────────────────────

class TestRouterMount:

    def test_dir_plugin_router_reachable(self, live_server):
        r = live_server["client"].get(
            "/api/plugins/stats/hello", headers=_admin_headers())
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_file_style_entry_uses_stem_as_name(self, tmp_path):
        """A bare .py entry is named after the file stem."""
        (tmp_path / "filestyle.py").write_text((PLUGINS_DIR / "filestyle.py")
                                               .read_text())
        cfg = _write_config(str(tmp_path), f"  - {tmp_path / 'filestyle.py'}")
        server = ArkestraServer(cfg, port=18007)
        client = TestClient(server.get_app())

        r = client.get("/api/plugins/filestyle/ping", headers=_admin_headers())
        assert r.status_code == 200 and r.json() == {"pong": True}

    def test_raw_plugin_path_not_exposed(self, live_server):
        """Plugin routes exist only under /api/plugins/<name>/."""
        r = live_server["client"].get("/stats/hello", headers=_admin_headers())
        assert r.status_code == 404


# ── Auth gating (inherited from /api/* middleware) ────────────────

class TestAuthGating:

    def test_requires_token(self, live_server):
        assert live_server["client"].get(
            "/api/plugins/stats/hello").status_code == 401

    def test_wrong_key_rejected(self, live_server):
        r = live_server["client"].get("/api/plugins/stats/hello",
                                      headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401

    def test_api_key_admits_plugin_routes(self, live_server):
        """Plugins sit under /api/* — the inference token must admit them."""
        r = live_server["client"].get("/api/plugins/stats/hello",
                                      headers={"Authorization": f"Bearer {API_KEY}"})
        assert r.status_code == 200


# ── register(ctx) contract: narrow surface, no raw app handle ─────

class TestRegisterHook:

    def test_register_received_context_not_app(self, live_server):
        import sys
        stats = sys.modules["arkestra_plugin_stats"]
        ctx = stats.saw_ctx
        # The context exposes the read-only model-state view…
        assert "test-model" in dict(ctx.models)
        # …and does NOT hand over a FastAPI app or middleware surface.
        from fastapi import FastAPI, APIRouter
        for value in (getattr(stats, name, None) for name in dir(ctx)):
            if not callable(value):
                assert not isinstance(value, FastAPI), \
                    "register() must never receive the raw app"

    def test_register_models_view_matches_registry(self, live_server):
        import sys
        uptime = sys.modules["arkestra_plugin_uptime"]
        server = live_server["server"]
        expected = {n: m.state.value for n, m in
                    server._arkestra.models.items()}
        assert uptime.saw_models == expected


# ── register() failure → no half-mounted router ───────────────────

class TestRegisterFailure:

    def test_setup_failure_records_error_and_skips_mount(self, tmp_path):
        cfg = _write_config(str(tmp_path), f"  - {PLUGINS_DIR / 'badsetup'}")
        server = ArkestraServer(cfg, port=18011)
        client = TestClient(server.get_app())

        r = client.get("/admin/plugins", headers=_admin_headers())
        entry = r.json()["plugins"][0]
        assert entry["state"] == "error"
        assert "setup failed" in entry["error"] and "boom" in entry["error"]

        # The router must not have been mounted by the failing plugin.
        assert client.get("/api/plugins/badsetup/hello",
                          headers=_admin_headers()).status_code == 404


# ── Malformed config: plugins key is a scalar, not a list ─────────

class TestConfigShape:

    def test_non_list_plugins_key_ignored(self, tmp_path):
        cfg = os.path.join(str(tmp_path), "config.yaml")
        Path(cfg).write_text(_CONFIG_TEMPLATE.format(
            admin_key=ADMIN_KEY, api_key=API_KEY, plugin_lines="oops").replace(
                "plugins:\noops", "plugins: oops"))
        server = ArkestraServer(cfg, port=18012)
        client = TestClient(server.get_app())

        r = client.get("/admin/plugins", headers=_admin_headers())
        assert r.json()["plugins"] == []


# ── Name validation (isolated configs — module fixture is busy) ───

class TestNameValidation:

    def test_invalid_identifier_recorded_as_error(self, tmp_path):
        bad = PLUGINS_DIR / "bad-name"
        shutil.copytree(bad if bad.is_dir() else PLUGINS_DIR / "stats",
                        bad, dirs_exist_ok=True)
        (bad / "plugin.py").write_text("router = None\n")  # valid module
        cfg = _write_config(str(tmp_path), f"  - {bad}")
        server = ArkestraServer(cfg, port=18008)
        client = TestClient(server.get_app())

        r = client.get("/admin/plugins", headers=_admin_headers())
        by_name = {p["name"]: p for p in r.json()["plugins"]}
        assert by_name["bad-name"]["state"] == "error"
        assert "identifier" in by_name["bad-name"]["error"]
        # Server still up, core routes fine
        assert client.get("/health").status_code == 200

    def test_duplicate_entry_recorded_as_error(self, tmp_path):
        """Same dir listed twice → second entry rejected, first loaded."""
        cfg = _write_config(str(tmp_path), f"\n".join(
            f"  - {PLUGINS_DIR / 'stats'}" for _ in range(2)))
        server = ArkestraServer(cfg, port=18009)
        client = TestClient(server.get_app())

        r = client.get("/admin/plugins", headers=_admin_headers())
        states = [p["state"] for p in r.json()["plugins"]]
        assert states == ["ok", "error"]
        # The duplicate error names the path that lost.
        dup = next(p for p in r.json()["plugins"] if p["state"] == "error")
        assert str(PLUGINS_DIR / "stats") in dup["error"]
        assert client.get("/api/plugins/stats/hello",
                          headers=_admin_headers()).status_code == 200

    def test_missing_path_recorded_as_error(self, tmp_path):
        cfg = _write_config(str(tmp_path), f"  - {tmp_path / 'nope'}")
        server = ArkestraServer(cfg, port=18010)
        client = TestClient(server.get_app())

        r = client.get("/admin/plugins", headers=_admin_headers())
        entry = r.json()["plugins"][0]
        assert entry["state"] == "error" and "neither" in entry["error"]
