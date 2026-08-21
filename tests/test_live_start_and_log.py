"""Test that qwen3.5-4b starts, logs appear on /admin/log?follow=true, and we capture them.

This is a live integration test — it hits the real uvicorn server with httpx,
starts an actual model, streams its startup logs, then cleans up.

Each test is standalone: start → work → stop. VRAM image stays cached between
stop/start cycles (minimal overhead), no cross-test interference.
"""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
import uvicorn

from model_arkestra.server import ArkestraServer
from tests.conftest import graceful_server_teardown


@pytest.fixture(scope="session")
def live_server():
    """Start a real ArkestraServer on a background thread."""
    proxy = ArkestraServer(
        config_path="tests/test-config.yaml",
        port=18005,
        ready_timeout=60,
    )
    app = proxy.get_app()

    config = uvicorn.Config(app, host="127.0.0.1", port=18005, log_level="warning")
    server_obj = uvicorn.Server(config)
    proxy._server = server_obj  # type: ignore

    def serve():
        import asyncio
        asyncio.run(server_obj.serve())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    url = "http://127.0.0.1:18005/admin/models"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        pytest.fail("Live server did not become ready in 30s")

    yield {"server": proxy, "client": httpx.Client(timeout=None)}
    graceful_server_teardown({"server": proxy, "client": httpx.Client(timeout=None)})


BASE = "http://127.0.0.1:18005"


def _wait_port_free(port, timeout=30):
    """Wait until a port is no longer in use."""
    import subprocess
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["ss", "-tlnp"], capture_output=True, text=True, timeout=5,
        )
        if f":{port}" not in result.stdout:
            return True
        time.sleep(0.5)
    return False


def _start_model(client):
    """Start qwen3.5-4b and wait for it to be running."""
    resp = client.post(f"{BASE}/admin/start/qwen3.5-4b", timeout=180)
    if resp.status_code != 200:
        # Check if model is already loading/running
        r2 = client.get(f"{BASE}/admin/models", timeout=10)
        for m in r2.json()["models"]:
            if m["id"] == "qwen3.5-4b":
                if m.get("status") in ("running", "loading"):
                    resp = httpx.Response(200, request=httpx.Request("POST"))
                    break
    assert resp.status_code == 200, f"Start failed: {resp.text}"
    # Wait until model reports running
    deadline = time.time() + 90
    while time.time() < deadline:
        r = client.get(f"{BASE}/admin/models", timeout=10)
        for m in r.json()["models"]:
            if m["id"] == "qwen3.5-4b" and m.get("status") == "running":
                return
        time.sleep(0.5)
    raise AssertionError(f"Model qwen3.5-4b did not reach 'running' state")


def _stop_model(client):
    """Stop qwen3.5-4b and wait for its port to be freed."""
    client.post(f"{BASE}/admin/stop/qwen3.5-4b", timeout=60)
    # Wait for model's port to actually free up (prevents zombie processes)
    models_resp = client.get(f"{BASE}/admin/models", timeout=10)
    for m in models_resp.json()["models"]:
        if m["id"] == "qwen3.5-4b" and m.get("port") is not None:
            _wait_port_free(m["port"], timeout=30)


def _stop_model(client):
    """Stop qwen3.5-4b."""
    client.post(f"{BASE}/admin/stop/qwen3.5-4b", timeout=60)


class TestLiveStartAndLogCapture:
    """Start qwen3.5-4b, stream its logs from /admin/log?follow=true."""

    def test_start_model_returns_ok(self, live_server):
        """Model should show as running."""
        client = live_server["client"]
        _start_model(client)
        try:
            models_resp = client.get(f"{BASE}/admin/models", timeout=10)
            for m in models_resp.json()["models"]:
                if m["id"] == "qwen3.5-4b":
                    assert m["status"] == "running", (
                        f"Expected 'running' but got '{m['status']}'"
                    )
        finally:
            _stop_model(client)

    def test_log_endpoint_returns_snapshot(self, live_server):
        """Stream startup logs via /admin/log?follow=true SSE."""
        client = live_server["client"]
        _start_model(client)
        try:
            all_lines: list[str] = []

            def collect():
                try:
                    with client.stream(
                        "GET", f"{BASE}/admin/log/qwen3.5-4b?follow=true",
                        timeout=None,
                    ) as resp:
                        assert resp.status_code == 200
                        for line in resp.iter_lines():
                            line = line.strip()
                            if not line or line == "data: [DONE]":
                                continue
                            data = json.loads(line.removeprefix("data: "))
                            if data.get("type") == "snapshot" and "lines" in data:
                                all_lines.extend(data["lines"])
                            elif data.get("type") == "line" and "lines" in data:
                                all_lines.extend(data["lines"])
                except Exception as e:
                    print(f"[log collect error] {e}")

            collector_thread = threading.Thread(target=collect, daemon=True)
            collector_thread.start()
            time.sleep(1.5)
            client.post(f"{BASE}/admin/stop/qwen3.5-4b", timeout=60)
            time.sleep(1)
            collector_thread.join(timeout=3)

            assert len(all_lines) > 0, "Expected at least one log line from startup"
        finally:
            _stop_model(client)

    def test_log_endpoint_streams_with_follow(self, live_server):
        """SSE follow mode returns events and ends cleanly."""
        client = live_server["client"]
        _start_model(client)
        try:
            events: list[dict] = []

            def collect():
                try:
                    with client.stream(
                        "GET", f"{BASE}/admin/log/qwen3.5-4b?follow=true",
                    ) as resp:
                        assert resp.status_code == 200
                        for line in resp.iter_lines():
                            line = line.strip()
                            if not line or line == "data: [DONE]":
                                continue
                            data = json.loads(line.removeprefix("data: "))
                            events.append(data)
                except Exception as e:
                    print(f"[log collect error] {e}")

            collector_thread = threading.Thread(target=collect, daemon=True)
            collector_thread.start()
            time.sleep(1.5)
            client.post(f"{BASE}/admin/stop/qwen3.5-4b", timeout=60)
            time.sleep(1)
            collector_thread.join(timeout=3)

            event_types = [e.get("type") for e in events]
            assert "snapshot" in event_types, f"Expected 'snapshot' event, got {event_types}"
            assert "line" in event_types, f"Expected 'line' event, got {event_types}"
        finally:
            _stop_model(client)

    def test_log_follow_mode_does_not_crash(self, live_server):
        """GET /admin/log?follow=true returns SSE without error."""
        client = live_server["client"]
        _start_model(client)
        try:
            with client.stream(
                "GET", f"{BASE}/admin/log/qwen3.5-4b?follow=true",
            ) as resp:
                assert resp.status_code == 200
                for line in resp.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    data = json.loads(line.removeprefix("data: "))
                    assert data["type"] == "snapshot"
                    break
        finally:
            _stop_model(client)

    def test_model_stops_cleanly(self, live_server):
        """Stop the model and verify it stops."""
        client = live_server["client"]
        _start_model(client)
        try:
            models_resp = client.get(f"{BASE}/admin/models", timeout=10)
            for m in models_resp.json()["models"]:
                if m["id"] == "qwen3.5-4b":
                    assert m["status"] == "running"

            resp = client.post(f"{BASE}/admin/stop/qwen3.5-4b", timeout=60)
            assert resp.status_code == 200

            models_resp = client.get(f"{BASE}/admin/models", timeout=10)
            for m in models_resp.json()["models"]:
                if m["id"] == "qwen3.5-4b":
                    assert m["status"] == "stopped"
        finally:
            _stop_model(client)
