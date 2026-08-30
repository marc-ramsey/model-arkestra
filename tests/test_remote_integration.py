"""Integration test — two real ArkestraServer instances talking over HTTP."""
from __future__ import annotations
import asyncio
import os
import sys
import tempfile
import textwrap
import time
import threading
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx
import pytest
import uvicorn


# ── Ports (18010–18019 per project convention) ────────────────────────────

WORKER_PORT = 18010
MASTER_PORT = 18011


def _start_server(port: int, config: str) -> tuple[Any, httpx.Client]:
    """Start ArkestraServer on *port*, return (proxy, httpx.Client)."""
    from model_arkestra.server import ArkestraServer

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp.write(config)
    tmp.close()

    proxy = ArkestraServer(config_path=tmp.name, port=port, ready_timeout=10)
    server_obj = uvicorn.Server(uvicorn.Config(
        proxy.get_app(), host="127.0.0.1", port=port, log_level="error"
    ))
    proxy._server = server_obj

    def serve():
        asyncio.run(server_obj.serve())

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    time.sleep(2)  # give uvicorn time to bind
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30)
    return proxy, client


def _stop(proxy: Any) -> None:
    if hasattr(proxy, "_server"):
        proxy._server.should_exit = True


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def worker_server():
    cfg = textwrap.dedent("""\
        models-start-port: 18020
        model-ports: 4
        warmup-time: 5
        backends:
          default: test-backend
        runners:
          default: ProcessModelRunner
        models:
          gemma:
            repo: hugging-face
            model: unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
    """)
    proxy, client = _start_server(WORKER_PORT, cfg)
    yield {"server": proxy, "client": client}
    _stop(proxy)


@pytest.fixture(scope="module")
def master_server(worker_server):
    cfg = textwrap.dedent(f"""\
        models-start-port: 18030
        model-ports: 4
        warmup-time: 5
        clusters:
          gpu-server:
            base-url: "http://127.0.0.1:{WORKER_PORT}"
        models:
          gpu-server/gemma:
            repo: hugging-face
            model: unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
    """)
    proxy, client = _start_server(MASTER_PORT, cfg)
    yield {"server": proxy, "client": client}
    _stop(proxy)


# ── Tests ──────────────────────────────────────────────────────────────────

class TestRemoteIntegration:

    def test_worker_health(self, worker_server):
        """Worker should respond on /health."""
        r = worker_server["client"].get("/health")
        assert r.status_code == 200

    def test_master_has_remote_model(self, master_server):
        """Master should list gpu-server/gemma in admin/models."""
        r = master_server["client"].get("/admin/models")
        assert r.status_code == 200
        ids = [m.get("id", "") for m in r.json()["models"]]
        assert "gpu-server/gemma" in ids

    def test_start_proxies_to_worker(self, worker_server, master_server):
        """POST /admin/start/<model> proxies to worker → port=0."""
        r = master_server["client"].post("/admin/start/gpu-server/gemma")
        assert r.status_code == 200
        body = r.json()
        assert body.get("ok") is True
        assert body.get("port") == 0

    def test_stop_proxies_to_worker(self, worker_server, master_server):
        """POST /admin/stop/<model> proxies to worker."""
        r = master_server["client"].post("/admin/stop/gpu-server/gemma")
        assert r.status_code == 200
        body = r.json()
        assert body.get("ok") is True

    def test_chat_proxies_to_worker(self, worker_server, master_server):
        """POST /v1/chat/completions with model=gpu-server/gemma → proxied."""
        r = master_server["client"].post(
            "/v1/chat/completions",
            json={"model": "gpu-server/gemma", "messages": [
                {"role": "user", "content": "hello"}], "stream": False},
        )
        # 503 because worker has no model loaded, but the proxy path is exercised
        assert r.status_code == 503
        detail = r.json()["detail"]
        assert "Remote inference failed" in detail or "gpu-server/gemma" in detail
