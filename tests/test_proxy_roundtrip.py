"""Round-trip integration tests for ArkestraServer (OpenAI v1 proxy).

Starts a *live* uvicorn server backed by a real ModelArkestra loaded from
test-config.yaml.  Every call goes through the full stack: FastAPI →
ArkestraServer routes → ModelArkestra.start / ainvoke / astream →
BaseModelRunner → llama-server (process/podman/docker).

No mocks of the arkestra backend — models are started via their real runners.

Usage:
    pytest tests/test_proxy_roundtrip.py -v --timeout=300
"""

from __future__ import annotations

import json as _json
import os
import threading
import time

import httpx
import pytest
import uvicorn

from model_arkestra.server import ArkestraServer


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def arkestra_server():
    """Start a real ArkestraServer on a background thread.

    Mirrors the lifetime of other integration test fixtures — started once per
    module, torn down at the end.
    """
    proxy = ArkestraServer(
        config_path="tests/test-config.yaml",
        port=18005,
        ready_timeout=60,
    )
    app = proxy.get_app()

    config = uvicorn.Config(app, host="127.0.0.1", port=18005, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=lambda: asyncio_run(server.serve()), daemon=True)
    thread.start()

    # Wait for the server to accept connections
    url = "http://127.0.0.1:18005/health"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        pytest.fail("ArkestraServer did not become ready in 30s")

    yield proxy

    # Shutdown — stop all model processes and exit uvicorn
    server.should_exit = True
    # Wait for process to drain port, then force-kill anything left
    import subprocess as _sp, time as _time
    for _ in range(10):
        result = _sp.run(["lsof", "-ti:", "18000"], capture_output=True, text=True)
        pids = [p for p in result.stdout.strip().split() if p]
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(int(pid), 9)
            except OSError:
                pass
        _time.sleep(0.3)


def asyncio_run(coro):
    """Run a coroutine — uvicorn needs its own event loop."""
    import asyncio
    asyncio.run(coro)


def _chat_url(port):
    return f"http://127.0.0.1:{port}/v1/chat/completions"


# ── Tests: non-streaming completions ───────────────────────────────────────


class TestChatCompletionsRoundTrip:
    """Real HTTP round-trip: client → uvicorn → ArkestraServer → ModelArkestra."""

    def test_basic_completion(self, arkestra_server):
        """Single user message → non-streaming response with assistant text."""
        port = arkestra_server.port

        resp = httpx.post(_chat_url(port), json={
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "Say one word."}],
        }, timeout=180)

        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert len(body["choices"][0]["message"]["content"]) > 0

    def test_streaming(self, arkestra_server):
        """Single message → streaming response with SSE chunks and [DONE]."""
        port = arkestra_server.port

        resp = httpx.post(_chat_url(port), json={
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "List numbers 1 to 3."}],
            "stream": True,
        }, timeout=180)

        assert resp.status_code == 200
        text = resp.text

        # Parse SSE chunks (exclude [DONE] and non-data lines)
        data_lines = [
            line.removeprefix("data: ").strip()
            for line in text.split("\n")
            if line.startswith("data: ") and "[DONE]" not in line
        ]
        events = [_json.loads(d) for d in data_lines]

        # Collect all token content across chunks
        tokens = "".join(
            c["delta"]["content"]
            for e in events
            for c in e.get("choices", [])
            if c.get("delta", {}).get("content")
        )
        assert len(tokens) > 0, "No tokens produced by streaming"

        # Verify SSE has the DONE marker
        assert "[DONE]" in resp.text

    def test_multi_turn(self, arkestra_server):
        """Multi-turn conversation: full history is forwarded through the stack."""
        port = arkestra_server.port

        resp = httpx.post(_chat_url(port), json={
            "model": "qwen3.5-4b",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "Four."},
                {"role": "user", "content": "Times three?"},
            ],
        }, timeout=180)

        assert resp.status_code == 200
        body = resp.json()
        content = body["choices"][0]["message"]["content"].lower()
        # Should reference the number 12 (4×3)
        assert "12" in content or "twelve" in content

    def test_request_params_forwarded(self, arkestra_server):
        """Extra params (temperature, stop tokens) are forwarded to the runner."""
        port = arkestra_server.port

        resp = httpx.post(_chat_url(port), json={
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "Say hello."}],
            "temperature": 0.1,
            "stop": ["bye"],
        }, timeout=180)

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["choices"][0]["message"]["content"]) > 0


# ── Tests: model listing / health ──────────────────────────────────────────


class TestListingHealthRoundTrip:
    """Real HTTP round-trip for non-chat endpoints."""

    def test_list_models(self, arkestra_server):
        """Running models are returned with the correct OpenAI shape."""
        port = arkestra_server.port
        resp = httpx.get(f"http://127.0.0.1:{port}/v1/models")

        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        for entry in body["data"]:
            assert "id" in entry
            assert entry["object"] == "model"
            assert "owned_by" in entry

    def test_health_endpoint(self, arkestra_server):
        """Health returns ok with a models_running count."""
        port = arkestra_server.port
        resp = httpx.get(f"http://127.0.0.1:{port}/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "models_running" in body

    def test_v1_health_endpoint(self, arkestra_server):
        """GET /v1/health returns the same data as GET /health."""
        port = arkestra_server.port
        resp = httpx.get(f"http://127.0.0.1:{port}/v1/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


# ── Tests: error handling ──────────────────────────────────────────────────


class TestErrorHandlingRoundTrip:
    """Verify error paths through the full stack."""

    def test_missing_messages_rejected(self, arkestra_server):
        """Request without messages returns 422 from FastAPI validation."""
        port = arkestra_server.port
        resp = httpx.post(_chat_url(port), json={"model": "qwen3.5-4b"})
        assert resp.status_code == 422

    def test_unstarted_model_returns_503(self, arkestra_server):
        """Starting a model not in config → 503 from ArkestraServer."""
        port = arkestra_server.port
        resp = httpx.post(_chat_url(port), json={
            "model": "does-not-exist-in-config",
            "messages": [{"role": "user", "content": "hi"}],
        }, timeout=10)
        assert resp.status_code == 503
