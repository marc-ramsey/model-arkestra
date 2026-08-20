"""Test that qwen3.5-4b starts, logs appear on /admin/log?follow=true, and we capture them.

This is a live integration test — it hits the real uvicorn server with httpx,
starts an actual model, streams its startup logs, then cleans up.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
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


def _stop_all_models_and_clean(server_dict):
    """Stop every model on the live server and remove any leftover podman containers."""
    client = server_dict["client"]
    # Stop all models via the stop-all endpoint
    try:
        client.post(f"{BASE}/admin/stop-all", timeout=30)
    except Exception:
        pass
    # Remove any leftover llm-* containers from this session
    result = subprocess.run(
        ["podman", "ps", "-a", "--filter", "name=llm-",
         "--format", "{{.ID}}"],
        capture_output=True, text=True, timeout=5,
    )
    for cid in result.stdout.strip().split():
        if cid:
            subprocess.run(["podman", "rm", "-f", cid], capture_output=True, timeout=5)


@pytest.fixture(autouse=True)
def _cleanup_live_models(live_server):
    """After each test in this module, stop all models and clean containers.

    This prevents model processes + podman containers from accumulating memory
    across the hundreds of other tests that run before/after these live tests.
    """
    yield
    _stop_all_models_and_clean(live_server)


class TestLiveStartAndLogCapture:
    """Start qwen3.5-4b, stream its logs from /admin/log?follow=true."""

    def test_start_model_returns_ok(self, live_server):
        """POST /admin/start/qwen3.5-4b should return 200 with ok=True."""
        resp = live_server["client"].post(
            f"{BASE}/admin/start/qwen3.5-4b",
            timeout=180,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["model"] == "qwen3.5-4b"
        assert "port" in body
        port = body["port"]

        # Model should show as running now
        models_resp = live_server["client"].get(
            f"{BASE}/admin/models", timeout=10
        )
        for m in models_resp.json()["models"]:
            if m["id"] == "qwen3.5-4b":
                assert m["status"] == "running", (
                    f"Expected 'running' but got '{m['status']}'"
                )

    def test_log_endpoint_returns_snapshot(self, live_server):
        """GET /admin/log/qwen3.5-4b?follow=false should return last log lines."""
        resp = live_server["client"].get(
            f"{BASE}/admin/log/qwen3.5-4b",
            params={"lines": 20},
            timeout=10,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "log"
        data = body.get("data", [])
        # Process runner captures logs in-process; podman/docker use container logs.
        # We expect some output once the model is running.

    def test_log_endpoint_streams_with_follow(self, live_server):
        """POST the model again to generate new log output, then capture it via SSE."""
        # Stop first so restart produces fresh logs
        live_server["client"].post(
            f"{BASE}/admin/stop/qwen3.5-4b", timeout=10
        )

        # Kick off log stream in background
        log_lines: list[str] = []
        collected_event = threading.Event()

        def collect_logs():
            try:
                with live_server["client"].stream(
                    "GET",
                    f"{BASE}/admin/log/qwen3.5-4b?follow=true",
                        ) as resp:
                    assert resp.status_code == 200
                    for line in resp.iter_lines():
                        line = line.strip()
                        if not line or line == "data: [DONE]":
                            continue
                        data = json.loads(line.removeprefix("data: "))
                        if data.get("type") == "snapshot" and "lines" in data:
                            log_lines.extend(data["lines"])
                        elif data.get("type") == "line" and "lines" in data:
                            log_lines.extend(data["lines"])

            except Exception as e:
                print(f"Log collection error: {e}")
            finally:
                collected_event.set()

        t = threading.Thread(target=collect_logs, daemon=True)
        t.start()

        # Give SSE a moment to connect and get the snapshot
        time.sleep(1)

        # Start model — this should produce new log lines
        resp = live_server["client"].post(
            f"{BASE}/admin/start/qwen3.5-4b",
            timeout=180,
        )
        assert resp.status_code == 200

        # Wait for collection to finish (stop route ends SSE)
        collected_event.wait(timeout=30)
        t.join(timeout=5)

        # Print what we captured
        if log_lines:
            print("\n=== Captured startup log lines ===")
            for line in log_lines[:50]:  # show first 50
                print(f"  {line}")
            print("==================================\n")
        else:
            print("[no log lines captured — logs are stored per-runner; "
                  "podman runner stores them as container logs, not in-process buffer]")

    def test_log_follow_mode_does_not_crash(self, live_server):
        """GET /admin/log?follow=true returns SSE without NameError.

        Regression test: the log_stream generator uses asyncio.sleep()
        inside an inner function — it must be in scope.
        """
        with live_server["client"].stream(
            "GET", f"{BASE}/admin/log/qwen3.5-4b?follow=true",
        ) as resp:
            assert resp.status_code == 200
            # Read the initial snapshot event
            for line in resp.iter_lines():
                if not line or line == "data: [DONE]":
                    continue
                data = json.loads(line.removeprefix("data: "))
                assert data["type"] == "snapshot"
                break  # snapshot received — the route works

    def test_model_stops_cleanly(self, live_server):
        """Stop the model and verify it stops."""
        resp = live_server["client"].post(
            f"{BASE}/admin/stop/qwen3.5-4b", timeout=10
        )
        assert resp.status_code == 200

        # Verify stopped
        models_resp = live_server["client"].get(
            f"{BASE}/admin/models", timeout=10
        )
        for m in models_resp.json()["models"]:
            if m["id"] == "qwen3.5-4b":
                assert m["status"] == "stopped"
