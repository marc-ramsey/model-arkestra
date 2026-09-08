"""E2E tests for /api/* endpoint authentication.

Verifies Bearer token enforcement on all /api/* routes:
- 401 when no header or wrong token (when api_key is configured)
- 200 with correct api_key OR admin_key on any /api/* route
- New endpoints: GET /api/models, POST /api/restart/{model}

Run: pytest tests/test_e2e_api_auth.py -v --timeout=120
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Tuple

import httpx
import pytest
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


API_AUTH_PORT = 18100


# ── Config with both api-key and admin-key set via default-env ───────────────

_API_KEY_CONFIG = """\
default:
  model-start-port: 18200
  model-ports: 4

default-env:
  admin-key: super-admin-key
  api-key: test-api-key

backends:
  vulkan-process:
    runner: process
    binary_dir: /home/marc/local/llama.cpp/build-vulkan-radv/bin
    binary: llama-server
    args:
      ngl: 999
      ctx-size: 2048

models:
  test-model:
    model: bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M
    backend: vulkan-process
    args:
      temp: 0.7
"""


# ── Server helper ───────────────────────────────────────────────────────────

def _start_api_server(port: int) -> Tuple[Any, httpx.Client]:
    """Start ArkestraServer with api_key/admin_key set."""
    from model_arkestra.server import ArkestraServer

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(_API_KEY_CONFIG)
        config_path = f.name

    try:
        proxy = ArkestraServer(config_path=config_path, port=port, ready_timeout=30)
        app = proxy.get_app()
        server_obj = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        proxy._server = server_obj

        def serve():
            import asyncio
            asyncio.run(server_obj.serve())

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
                return proxy, httpx.Client(timeout=30)
            except Exception:
                time.sleep(0.3)
        raise RuntimeError(f"Server on port {port} did not become ready")
    except Exception:
        os.unlink(config_path)
        raise


def _stop_server(proxy: Any, client: httpx.Client, port: int) -> None:
    try:
        client.post(f"http://127.0.0.1:{port}/admin/shutdown", timeout=10)
    except Exception:
        pass
    time.sleep(1)
    try:
        client.close()
    except Exception:
        pass


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def api_server():
    """Single server with keys enabled for the entire test class."""
    proxy, client = _start_api_server(API_AUTH_PORT)
    yield {"server": proxy, "client": client,
           "base_url": f"http://127.0.0.1:{API_AUTH_PORT}"}
    _stop_server(proxy, client, API_AUTH_PORT)


@pytest.fixture(scope="class")
def no_key_server():
    """Server with NO keys configured — auth should be disabled."""
    cfg = _API_KEY_CONFIG.replace("  admin-key: super-admin-key\n", "") \
                         .replace("  api-key: test-api-key\n", "")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(cfg)
        config_path = f.name

    from model_arkestra.server import ArkestraServer
    port = API_AUTH_PORT + 10
    proxy = ArkestraServer(config_path=config_path, port=port, ready_timeout=30)
    app = proxy.get_app()
    server_obj = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    proxy._server = server_obj

    def serve():
        import asyncio
        asyncio.run(server_obj.serve())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    deadline = time.time() + 60
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=2)
            break
        except Exception:
            time.sleep(0.3)

    client = httpx.Client(timeout=30)
    yield {"server": proxy, "client": client,
           "base_url": f"http://127.0.0.1:{port}"}
    try:
        client.post(f"http://127.0.0.1:{port}/admin/shutdown", timeout=10)
    except Exception:
        pass
    time.sleep(1)
    client.close()


# ── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.e2e
class TestAPIAuthNoKey:
    """When keys are empty — all /api/* routes are open."""

    def test_api_models_no_auth_allowed(self, no_key_server):
        """No keys → GET /api/models returns 200 with no auth."""
        r = no_key_server["client"].get(f"{no_key_server['base_url']}/api/models")
        assert r.status_code == 200

    def test_api_start_no_auth_allowed(self, no_key_server):
        """No keys → POST /api/start returns response, not 401."""
        r = no_key_server["client"].post(
            f"{no_key_server['base_url']}/api/start/test-model", timeout=5)
        assert r.status_code == 200


@pytest.mark.e2e
class TestAPIAuthWithKey:
    """When api_key is set, /api/* requires Bearer token."""

    def test_no_header_returns_401(self, api_server):
        r = api_server["client"].get(f"{api_server['base_url']}/api/models")
        assert r.status_code == 401
        body = api_server["client"].get(f"{api_server['base_url']}/api/models").json()
        assert "Authorization" in body.get("error", "")

    def test_wrong_token_returns_401(self, api_server):
        r = api_server["client"].get(
            f"{api_server['base_url']}/api/models",
            headers={"Authorization": "Bearer wrong-key"})
        assert r.status_code == 401

    def test_api_key_works(self, api_server):
        """Correct api_key → 200 on GET /api/models."""
        r = api_server["client"].get(
            f"{api_server['base_url']}/api/models",
            headers={"Authorization": "Bearer test-api-key"})
        assert r.status_code == 200

    def test_admin_key_works_on_api(self, api_server):
        """Admin key also works on /api/* routes."""
        r = api_server["client"].get(
            f"{api_server['base_url']}/api/models",
            headers={"Authorization": "Bearer super-admin-key"})
        assert r.status_code == 200

    def test_api_start_requires_auth(self, api_server):
        r = api_server["client"].post(
            f"{api_server['base_url']}/api/start/test-model")
        assert r.status_code == 401

    def test_api_stop_requires_auth(self, api_server):
        r = api_server["client"].post(
            f"{api_server['base_url']}/api/stop/test-model")
        assert r.status_code == 401

    def test_api_restart_requires_auth(self, api_server):
        r = api_server["client"].post(
            f"{api_server['base_url']}/api/restart/test-model")
        assert r.status_code == 401


@pytest.mark.e2e
class TestAPIRoutes:
    """New /api/* endpoints return correct response shapes."""

    def test_api_models_list_structure(self, api_server):
        """GET /api/models returns cached-only models with name/model/size only."""
        r = api_server["client"].get(
            f"{api_server['base_url']}/api/models",
            headers={"Authorization": "Bearer test-api-key"})
        assert r.status_code == 200
        body = r.json()
        models = body.get("models", [])
        for m in models:
            assert set(m.keys()) <= {"name", "model", "size"}

    def test_api_restart_returns_ok(self, api_server):
        """POST /api/restart returns ok=True."""
        r = api_server["client"].post(
            f"{api_server['base_url']}/api/restart/test-model",
            headers={"Authorization": "Bearer test-api-key"})
        assert r.status_code == 200
        body = r.json()
        assert body.get("ok") is True

    def test_api_restart_unknown_model_404(self, api_server):
        r = api_server["client"].post(
            f"{api_server['base_url']}/api/restart/nonexistent-model",
            headers={"Authorization": "Bearer test-api-key"})
        assert r.status_code == 404

    def test_api_restart_wrong_key_401(self, api_server):
        r = api_server["client"].post(
            f"{api_server['base_url']}/api/restart/test-model",
            headers={"Authorization": "Bearer wrong-key"})
        assert r.status_code == 401


@pytest.mark.e2e
class TestAdminRoutesWithKey:
    """Admin routes work with both api_key and admin_key."""

    def test_admin_models_with_api_key(self, api_server):
        r = api_server["client"].get(
            f"{api_server['base_url']}/admin/models",
            headers={"Authorization": "Bearer test-api-key"})
        assert r.status_code == 200

    def test_admin_models_with_admin_key(self, api_server):
        r = api_server["client"].get(
            f"{api_server['base_url']}/admin/models",
            headers={"Authorization": "Bearer super-admin-key"})
        assert r.status_code == 200

    def test_admin_models_wrong_key_401(self, api_server):
        r = api_server["client"].get(
            f"{api_server['base_url']}/admin/models",
            headers={"Authorization": "Bearer nobody-key"})
        assert r.status_code == 401

    def test_admin_start_wrong_key_401(self, api_server):
        r = api_server["client"].post(
            f"{api_server['base_url']}/admin/start/test-model",
            headers={"Authorization": "Bearer nobody-key"})
        assert r.status_code == 401
