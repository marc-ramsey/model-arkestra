"""Live integration tests for ArkestraAdmin endpoints (via real uvicorn, no mocks).

Starts a real uvicorn server backed by a real ModelArkestra loaded from
tests/test-config.yaml.  Every admin call goes through the full stack:
FastAPI → admin routes → ModelArkestra.start / stop_all / shutdown.

No mocks — models are started via their real runners, uvicorn is a real
background process, httpx talks to it over HTTP like a real client would.
"""

from __future__ import annotations

import os
import threading
import time

import httpx
import pytest
import uvicorn

from model_arkestra.server import ArkestraServer
from tests.conftest import graceful_server_teardown


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live_admin_server():
    """Start a real ArkestraServer on a background thread for admin endpoint tests."""
    proxy = ArkestraServer(
        config_path="tests/test-config.yaml",
        port=18005,
        ready_timeout=60,
    )
    app = proxy.get_app()

    config = uvicorn.Config(app, host="127.0.0.1", port=18005, log_level="warning")
    server_obj = uvicorn.Server(config)

    # Wire it into proxy._server so shutdown route can shut down uvicorn cleanly
    proxy._server = server_obj  # type: ignore

    def serve():
        import asyncio
        asyncio.run(server_obj.serve())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    # Wait for the server to accept connections
    url = "http://127.0.0.1:18005/admin/models"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        pytest.fail("Live admin server did not become ready in 30s")

    yield proxy
    graceful_server_teardown(proxy)


BASE = "http://127.0.0.1:18005"


# ── Tests: /admin/stop-all ─────────────────────────────────────────────


class TestStopAllLive:
    """POST /admin/stop-all — live HTTP, no mocks."""

    def test_stop_all_no_models_returns_200(self, live_admin_server):
        """When no models are running, returns 200 with a message."""
        resp = httpx.post(f"{BASE}/admin/stop-all", timeout=5)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "nothing to stop" in body["message"].lower()
        assert isinstance(body["stopped"], list)
        assert len(body["stopped"]) == 0

    def test_stop_all_stops_model(self, live_admin_server):
        """When a model is running, stop-all stops it and lists it."""
        # First start a model — use short timeout config for speed
        resp = httpx.post(
            f"{BASE}/admin/start/qwen3.5-4b",
            json={"max_tokens": 16, "temp": 0.7},
            timeout=180,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["model"] == "qwen3.5-4b"

        # Now stop-all — should stop the running model
        resp = httpx.post(f"{BASE}/admin/stop-all", timeout=10)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "qwen3.5-4b" in body["stopped"]

        # Model should be stopped now (lazy restart on next request)
        models_resp = httpx.get(f"{BASE}/admin/models", timeout=5)
        for m in models_resp.json()["models"]:
            if m["id"] == "qwen3.5-4b":
                assert m["status"] == "stopped"

    def test_stop_all_twice_is_safe(self, live_admin_server):
        """Calling stop-all when already stopped returns clean response (idempotent)."""
        httpx.post(f"{BASE}/admin/stop-all", timeout=5)
        resp = httpx.post(f"{BASE}/admin/stop-all", timeout=5)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True


# ── Tests: /admin/shutdown ────────────────────────────────────────────────


class TestShutdownLive:
    """POST /admin/shutdown — verifies response before server exits."""

    def test_shutdown_sends_response_before_exit(self, live_admin_server):
        """The 200 response arrives before uvicorn stops accepting connections.

        Does NOT start a model — shutdown with no running models is still a real
        end-to-end test: the endpoint exists, auth passes, response is sent, then
        the background task tears everything down.
        """
        client = httpx.Client(timeout=None)
        resp = client.post(f"{BASE}/admin/shutdown", timeout=3)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "shutting down" in body["message"].lower()
        client.close()

        # After shutdown, server should no longer be reachable
        time.sleep(5)
        try:
            httpx.get(f"{BASE}/health", timeout=3)
            pytest.fail("Server should have shut down but is still responding")
        except (httpx.ConnectError, httpx.ConnectTimeout, OSError):
            pass  # expected — server is gone
