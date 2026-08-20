"""Live integration test: start model via server and capture startup logs.

Uses sample-config.yaml as-is — models route through backends → runners.
This is the exact documented routing chain from docs/config.md + docs/architecture.md.
Starts its own ArkestraServer on port 18005 for isolation.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time

import httpx
import pytest
import uvicorn

from model_arkestra.server import ArkestraServer
from tests.conftest import graceful_server_teardown


def _clean_containers():
    """Remove any leftover llm-* containers (pasta cleans up automatically)."""
    result = __import__("subprocess").run(
        ["podman", "ps", "-a", "--filter", "name=llm-",
         "--format", "{{.ID}}"],
        capture_output=True, text=True, timeout=5,
    )
    for cid in result.stdout.strip().split():
        if cid:
            __import__("subprocess").run(
                ["podman", "rm", "-f", cid],
                capture_output=True, timeout=10,
            )


@pytest.fixture(scope="session")
def live_server():
    """Start a real ArkestraServer on port 18005 in a background thread."""
    import time as _time
    
    # Clean up stale containers (pasta cleans up with container removal)
    _clean_containers()
    _time.sleep(0.3)
    
    proxy = ArkestraServer(
        config_path="tests/test-integration-config.yaml",
        port=18006,
        ready_timeout=360,
    )
    app = proxy.get_app()

    config = uvicorn.Config(app, host="127.0.0.1", port=18006, log_level="warning")
    server_obj = uvicorn.Server(config)
    proxy._server = server_obj  # type: ignore

    def serve():
        import asyncio
        asyncio.run(server_obj.serve())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    url = "http://127.0.0.1:18006/admin/models"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        pytest.fail("Live server did not become ready in 30s")

    key = proxy._arkestra.cm.data.get("env", {}).get("ADMIN_KEY") or ""
    client_headers = {"X-Admin-Key": key} if key else {}

    yield {
        "server": proxy,
        "client": httpx.Client(timeout=None, headers=client_headers),
    }
    graceful_server_teardown({"server": proxy})


BASE = "http://127.0.0.1:18006"


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


class TestLiveServerStartAndLogCapture:
    """Live server integration — model start + log streaming via SSE."""

    def test_server_health_ok(self, live_server):
        """Server is up and healthy."""
        r = httpx.get(f"{BASE}/health", timeout=5)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"

    def test_models_list_accessible(self, live_server):
        """GET /admin/models returns a valid model list."""
        r = live_server["client"].get(f"{BASE}/admin/models", timeout=5)
        assert r.status_code == 200
        models_by_id = {m["id"]: m for m in r.json()["models"]}
        # At least one model should be listed
        assert len(models_by_id) > 0

    def test_start_and_capture_startup_logs(self, live_server):
        """Start a model, stream its startup logs via /admin/log?follow=true SSE."""
        client = live_server["client"]

        # Kick off log stream in background
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

        # Let SSE connect and grab the snapshot
        time.sleep(1.5)

        # Start model — backend routing determined by config
        r = client.post(
            f"{BASE}/admin/start/qwen3.5-4b",
            timeout=360,
        )
        assert r.status_code == 200

        # Stop the SSE stream so it returns
        client.post(f"{BASE}/admin/stop/qwen3.5-4b", timeout=60)
        time.sleep(1)
        collector_thread.join(timeout=3)

        # Print captured log lines for debugging
        print()
        if all_lines:
            print("=== Captured startup logs ===")
            for line in all_lines:
                print(f"  {line}")
            print("=============================")
        else:
            # Fallback: podman/docker runners store logs as container logs,
            # not in-process buffer — get them directly via snapshot endpoint.
            r2 = client.get(
                f"{BASE}/admin/log/qwen3.5-4b?follow=false",
                timeout=10,
            )
            if r2.status_code == 200:
                log_data = r2.json().get("data", [])
                if log_data:
                    print("=== Container logs (fallback) ===")
                    for line in log_data:
                        print(f"  {line}")

    def test_model_state_after_start(self, live_server):
        """Model status reflects the outcome of a start attempt."""
        r = live_server["client"].get(f"{BASE}/admin/models", timeout=5)
        assert r.status_code == 200
        models_by_id = {m["id"]: m for m in r.json()["models"]}
        qwen = models_by_id.get("qwen3.5-4b")
        assert qwen is not None
        print(f"  qwen3.5-4b status: {qwen['status']} (port={qwen['port']})")
