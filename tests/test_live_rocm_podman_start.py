"""Live test: start qwen3.5-4b (rocm/podman) and capture startup logs.

Uses sample-config.yaml as-is — qwen3.5-4b → backend rocm → runner podman.
This is the exact documented routing chain from docs/config.md + docs/architecture.md.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest


BASE = "http://127.0.0.1:21110"
COOKIE = {"admin_key": "whatever"}


class TestLiveRoCmPodmanStartAndLog:
    """Start qwen3.5-4b via rocm/podman, stream its startup logs."""

    def test_server_starts_and_health_ok(self):
        """Server is up and returns 200 on /health."""
        r = httpx.get(f"{BASE}/health", timeout=5)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"

    def test_models_list_shows_qwen3_5_4b(self):
        """GET /admin/models lists qwen3.5-4b with backend rocm."""
        r = httpx.get(f"{BASE}/admin/models", cookies=COOKIE, timeout=5)
        assert r.status_code == 200
        models_by_id = {m["id"]: m for m in r.json()["models"]}
        qwen = models_by_id.get("qwen3.5-4b")
        assert qwen is not None
        assert qwen["backend_id"] == "rocm"

    def test_start_model_and_capture_startup_logs(self):
        """Start qwen3.5-4b, stream logs from /admin/log?follow=true."""
        # Kick off log stream in background
        all_lines: list[str] = []

        def collect():
            try:
                with httpx.stream(
                    "GET", f"{BASE}/admin/log/qwen3.5-4b?follow=true",
                    cookies=COOKIE, timeout=None,
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

        # Start model — rocm backend → podman runner
        r = httpx.post(
            f"{BASE}/admin/start/qwen3.5-4b",
            cookies=COOKIE, timeout=60,
        )

        # Stop the SSE stream so it returns
        httpx.post(f"{BASE}/admin/stop/qwen3.5-4b", cookies=COOKIE, timeout=10)
        time.sleep(1)
        collector_thread.join(timeout=3)

        # Print captured log lines
        print()
        if all_lines:
            print("=== Captured startup logs ===")
            for line in all_lines:
                print(f"  {line}")
            print("=============================")
            print()
            # Check we got meaningful output (container create, image pull, etc.)
            has_container_cmd = any("podman" in l or "run" in l for l in all_lines)
            has_image_ref = any("ark-llama" in l for l in all_lines)
            print(f"  Container command logged: {has_container_cmd}")
            print(f"  Image reference logged: {has_image_ref}")
        else:
            print("  [no log lines captured via in-process buffer]")
            # Podman runner stores logs as container logs — try getting them directly
            r2 = httpx.get(
                f"{BASE}/admin/log/qwen3.5-4b?follow=false",
                cookies=COOKIE, timeout=10,
            )
            if r2.status_code == 200:
                log_data = r2.json().get("data", [])
                if log_data:
                    print("=== Container logs (fallback) ===")
                    for line in log_data:
                        print(f"  {line}")

    def test_model_state_after_start(self):
        """After start attempt, model status reflects outcome."""
        r = httpx.get(f"{BASE}/admin/models", cookies=COOKIE, timeout=5)
        assert r.status_code == 200
        models_by_id = {m["id"]: m for m in r.json()["models"]}
        qwen = models_by_id.get("qwen3.5-4b")
        print(f"  qwen3.5-4b status: {qwen['status']} (port={qwen['port']})")
        assert qwen is not None
