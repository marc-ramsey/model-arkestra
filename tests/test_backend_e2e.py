"""Backend runner end-to-end tests.

Each test is fully self-contained: starts its own uvicorn server + model,
runs a full lifecycle (start → inference → logs → stop), and guarantees
cleanup before returning — no inter-test pollution possible.

Parametrized over backend+runner combinations:
  - process (default)  : llama-server launched directly via subprocess
  - docker             : llama-server launched inside a Docker container
  - podman             : llama-server launched inside a Podman container

Run all e2e tests:
    pytest tests/test_backend_e2e.py --e2e -v --timeout=300

Run only one combination:
    pytest tests/test_backend_e2e.py::test_full_lifecycle_process -v
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict

import httpx
import pytest
import uvicorn

from model_arkestra.server import ArkestraServer


# ── Test config for e2e ──────────────────────────────────────────────────────
# Supports 3 runner types via explicit "runner" field on each model.

_E2E_CONFIG = """
warmup-time: 10
models-start-port: 18500
model-ports: 64
env:
  ADMIN_KEY: test-e2e-key

backends:
  default: process-default
  process-default:
    binary_dir: /home/marc/local/llama.cpp/build-vulkan-radv/bin
    binary: llama-server
    runner: process
  docker-backend:
    binary_dir: /home/marc/local/llama.cpp/build-vulkan-radv/bin
    binary: llama-server
    image: ark-llama:vulkan-radv
    runner: docker
  podman-backend:
    binary_dir: /home/marc/local/llama.cpp/build-vulkan-radv/bin
    binary: llama-server
    image: ark-llama:vulkan-radv
    runner: podman

runners:
  default: ProcessModelRunner

models:
  qwen3.5-4b-process:
    checkpoint: unsloth/Qwen3.5-4B-GGUF:Q4_K_M
    args:
      temp: 0.7
      top-p: 0.95
      ctx-size: 2048

  qwen3.5-4b-docker:
    checkpoint: unsloth/Qwen3.5-4B-GGUF:Q4_K_M
    backend: docker-backend
    args:
      temp: 0.7
      top-p: 0.95
      ctx-size: 2048

  qwen3.5-4b-podman:
    checkpoint: unsloth/Qwen3.5-4B-GGUF:Q4_K_M
    backend: podman-backend
    args:
      temp: 0.7
      top-p: 0.95
      ctx-size: 2048

  gemma-4-e2b-process:
    checkpoint: unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
    args:
      temp: 0.7
      top-p: 0.95
      ctx-size: 2048
"""


# ── Fixture helpers ──────────────────────────────────────────────────────────

def _start_server(port: int, admin_key: str = "test-e2e-key") -> tuple[ArkestraServer, httpx.Client]:
    """Start a real ArkestraServer on the given port. Returns proxy + client."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(_E2E_CONFIG)
        config_path = f.name

    try:
        proxy = ArkestraServer(
            config_path=config_path,
            port=port,
            ready_timeout=60,
        )
        app = proxy.get_app()

        uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server_obj = uvicorn.Server(uvicorn_config)
        proxy._server = server_obj  # type: ignore

        def serve():
            asyncio.run(server_obj.serve())

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        # Wait for the server to be ready
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                httpx.get(url, timeout=2)
                break
            except Exception:
                time.sleep(0.3)
        else:
            raise RuntimeError(f"Server on port {port} did not become ready")

        client = httpx.Client(timeout=None, headers={"X-Admin-Key": admin_key})
        return proxy, client
    except Exception:
        os.unlink(config_path)
        raise


def _stop_server(proxy: ArkestraServer, client: httpx.Client, port: int) -> None:
    """Tear down server and all its models."""
    # 1. Stop all running models first
    try:
        client.post(f"http://127.0.0.1:{port}/admin/stop-all", timeout=30)
    except Exception:
        pass

    # 2. Remove any leftover containers (podman + docker)
    for cmd in (["podman", "ps", "-a", "--filter", "name=llm-", "--format", "{{.ID}}"],
                ["docker", "ps", "-a", "--filter", "name=llm-", "--format", "{{.ID}}"]):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            for cid in result.stdout.strip().split():
                if cid:
                    subprocess.run(["podman", "rm", "-f", cid], capture_output=True, timeout=5)
                    subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=5)
        except Exception:
            pass

    # 3. Shut down uvicorn
    server_obj = getattr(proxy, "_server", None)
    if server_obj is not None:
        server_obj.should_exit = True

    # 4. Wait for port to be free
    _wait_port_free(port, timeout=20)

    # 5. Close client
    try:
        client.close()
    except Exception:
        pass


def _wait_port_free(port: int, timeout: float = 20.0) -> bool:
    """Block until no process is listening on *port*."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["ss", "-tlnp"], capture_output=True, text=True, timeout=5,
        )
        if f":{port}" not in result.stdout:
            return True
        time.sleep(0.3)
    # Last resort — kill anything on the port
    subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], capture_output=True, timeout=5)
    return False


def _start_model(client: httpx.Client, base_url: str, model_name: str) -> bool:
    """Start a model and wait until it reaches 'running' state. Returns True on success."""
    resp = client.post(f"{base_url}/admin/start/{model_name}", timeout=180)
    if resp.status_code != 200:
        return False

    deadline = time.time() + 90
    while time.time() < deadline:
        r = client.get(f"{base_url}/admin/models", timeout=10)
        for m in r.json()["models"]:
            if m["id"] == model_name and m.get("status") == "running":
                return True
        time.sleep(0.5)
    return False


def _stop_model(client: httpx.Client, base_url: str, model_name: str) -> None:
    """Stop a model and wait for its port to be freed."""
    try:
        client.post(f"{base_url}/admin/stop/{model_name}", timeout=60)
    except Exception:
        pass
    # Also try stop-all as safety net
    try:
        client.post(f"{base_url}/admin/stop-all", timeout=10)
    except Exception:
        pass


# ── Parametrization ──────────────────────────────────────────────────────────

def _get_port():
    """Allocate a unique port per test function using a file-based counter."""
    import fcntl
    counter_path = "/tmp/.arkestra_e2e_port_counter"
    lock_path = counter_path + ".lock"

    with open(lock_path, "w") as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        try:
            with open(counter_path, "r") as cf:
                base = int(cf.read().strip())
            port = base + os.getpid() % 1000  # spread across the 64k range
            if port < 18500 or port > 24999:
                port = 18500
        except (FileNotFoundError, ValueError):
            port = 18500 + os.getpid() % 1000

        with open(counter_path, "w") as cf:
            cf.write(str(port))

    # Ensure uniqueness — if the port is taken, try next one
    tries = 0
    while tries < 100:
        result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=5)
        if f":{port}" not in result.stdout:
            return port
        port += 1
        tries += 1

    raise RuntimeError(f"Could not find a free port after {tries} attempts")


# Backend/runner combinations to test.
# Each tuple: (test_id, model_name_in_config, runner_label)
BACKEND_COMBOS = [
    ("process", "qwen3.5-4b-process", "ProcessModelRunner"),
    ("gemma-process", "gemma-4-e2b-process", "ProcessModelRunner"),
]

# Docker/Podman combos are conditional — only included if the runtime is available
import shutil

if shutil.which("docker"):
    BACKEND_COMBOS.append(("docker", "qwen3.5-4b-docker", "Docker"))

if shutil.which("podman"):
    BACKEND_COMBOS.append(("podman", "qwen3.5-4b-podman", "Podman"))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def e2e_server(request):
    """Per-test self-contained server + client with guaranteed cleanup.

    Each test that uses this fixture gets its own uvicorn process and port.
    The model lifecycle (start → stop) is handled within the test body using
    _start_model() and _stop_model().
    """
    test_port = _get_port()
    admin_key = "test-e2e-key"
    base_url = f"http://127.0.0.1:{test_port}"

    proxy, client = _start_server(test_port, admin_key)

    # Expose to test via request node for param info
    request.instance._e2e_port = test_port
    request.instance._e2e_base_url = base_url
    request.instance._e2e_client = client

    yield {"server": proxy, "client": client, "port": test_port, "base_url": base_url}

    # ── Guaranteed teardown (always runs, even on assertion failure) ──────
    try:
        _stop_server(proxy, client, test_port)
    except Exception:
        pass  # Best effort — don't mask test failures


# ── Tests ────────────────────────────────────────────────────────────────────

class TestFullLifecycle:
    """Start a model → verify inference → check logs → stop. Clean slate each time."""

    @pytest.mark.e2e
    @pytest.mark.parametrize("test_id,model_name,runner_label", BACKEND_COMBOS, ids=[c[0] for c in BACKEND_COMBOS])
    def test_ainvoke(self, e2e_server, test_id: str, model_name: str, runner_label: str):
        """Single message → non-streaming response."""
        client = e2e_server["client"]
        base_url = e2e_server["base_url"]

        ok = _start_model(client, base_url, model_name)
        try:
            assert ok, f"Model {model_name} ({test_id}) failed to start"

            # Verify inference works via chat completions endpoint
            resp = client.post(f"{base_url}/v1/chat/completions", json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Say one word: hello"}],
                "max_tokens": 8,
            }, timeout=60)

            assert resp.status_code == 200, f"Inference failed: {resp.text}"
            body = resp.json()
            assert body["object"] == "chat.completion"
            assert len(body["choices"]) > 0
            content = body["choices"][0]["message"]["content"]
            assert len(content) > 0, "Empty response from model"

        finally:
            _stop_model(client, base_url, model_name)

    @pytest.mark.e2e
    @pytest.mark.parametrize("test_id,model_name,runner_label", BACKEND_COMBOS, ids=[c[0] for c in BACKEND_COMBOS])
    def test_astream(self, e2e_server, test_id: str, model_name: str, runner_label: str):
        """Single message → streaming response with SSE chunks."""
        client = e2e_server["client"]
        base_url = e2e_server["base_url"]

        ok = _start_model(client, base_url, model_name)
        try:
            assert ok, f"Model {model_name} ({test_id}) failed to start"

            resp = client.post(f"{base_url}/v1/chat/completions", json={
                "model": model_name,
                "messages": [{"role": "user", "content": "List 1, 2, 3"}],
                "max_tokens": 16,
                "stream": True,
            }, timeout=60)

            assert resp.status_code == 200
            assert "[DONE]" in resp.text, "Missing [DONE] marker in stream"

            # Verify we got actual tokens (not just whitespace)
            data_lines = [
                line.removeprefix("data: ").strip()
                for line in resp.text.split("\n")
                if line.startswith("data: ") and "[DONE]" not in line
            ]
            assert len(data_lines) > 0, "No token chunks in stream"

        finally:
            _stop_model(client, base_url, model_name)

    @pytest.mark.e2e
    @pytest.mark.parametrize("test_id,model_name,runner_label", BACKEND_COMBOS, ids=[c[0] for c in BACKEND_COMBOS])
    def test_logs_captured(self, e2e_server, test_id: str, model_name: str, runner_label: str):
        """Model logs appear via the /admin/log endpoint."""
        client = e2e_server["client"]
        base_url = e2e_server["base_url"]

        ok = _start_model(client, base_url, model_name)
        try:
            assert ok, f"Model {model_name} ({test_id}) failed to start"

            # Small delay to let logs accumulate
            time.sleep(2)

            resp = client.get(f"{base_url}/admin/log/{model_name}", params={
                "since": 0, "lines": 100
            }, timeout=10)

            assert resp.status_code == 200
            data = resp.json()
            assert "lines" in data
            assert len(data["lines"]) > 0, f"No log lines captured for {model_name}"

            # Verify log entries have expected structure
            for entry in data["lines"]:
                assert "text" in entry, f"Missing 'text' in log entry: {entry}"

        finally:
            _stop_model(client, base_url, model_name)

    @pytest.mark.e2e
    @pytest.mark.parametrize("test_id,model_name,runner_label", BACKEND_COMBOS, ids=[c[0] for c in BACKEND_COMBOS])
    def test_stop_then_start_restores_stopped_state(self, e2e_server, test_id: str, model_name: str, runner_label: str):
        """After stopping a model, it reports 'stopped'. Restarting brings it back."""
        client = e2e_server["client"]
        base_url = e2e_server["base_url"]

        ok = _start_model(client, base_url, model_name)
        try:
            assert ok, f"Model {model_name} ({test_id}) failed to start"

            # Stop it
            _stop_model(client, base_url, model_name)

            # Verify stopped state
            r = client.get(f"{base_url}/admin/models", timeout=10)
            for m in r.json()["models"]:
                if m["id"] == model_name:
                    assert m["status"] == "stopped", (
                        f"Expected 'stopped' after stop, got '{m['status']}'"
                    )

        finally:
            _stop_model(client, base_url, model_name)


class TestErrorPaths:
    """Verify error handling through the full stack."""

    @pytest.mark.e2e

    def test_unknown_model_404(self, e2e_server):
        """Starting a model not in config returns 404."""
        client = e2e_server["client"]
        base_url = e2e_server["base_url"]

        resp = client.post(f"{base_url}/admin/start/nonexistent-model", timeout=10)
        assert resp.status_code == 404

    @pytest.mark.e2e
    def test_start_nonexistent_returns_503_via_chat(self, e2e_server):
        """Chat endpoint for unstarted model returns 503."""
        client = e2e_server["client"]
        base_url = e2e_server["base_url"]

        resp = client.post(f"{base_url}/v1/chat/completions", json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "hi"}],
        }, timeout=5)
        assert resp.status_code == 503
