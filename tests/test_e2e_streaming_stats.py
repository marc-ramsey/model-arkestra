"""E2E: streaming requests record real token stats via /api/v1/stats.

Replaces the TestClient-based mock tests that could never observe post-yield
generator code (starlette's in-memory transport cancels async generators before
trailing bookkeeping runs). Uses a real uvicorn server + cached sub-1B model,
so this exercises the exact production path: verbatim SSE forwarding plus the
side-channel stats write after stream end.

Serialization: every test holds an exclusive flock for its full duration
(server start → inference → stop-all), so two concurrent pytest invocations —
or a leaked llama-server from a crashed run — can never overlap model loads on
the same GPU and OOM each other. The lock file is shared by path; if you later
harden test_backend_e2e.py the same constant applies there too.
"""

import fcntl
import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))
# Import helpers from the existing e2e suite instead of duplicating them.
from test_backend_e2e import (  # noqa: E402
    COMBOS,
    _build_e2e_config,
    _start_server,
    _stop_all_and_wait,
    _stop_server,
    _admin_headers,
)

# ── Constants ────────────────────────────────────────────────────────────────

E2E_LOCK = "/tmp/arkestra-e2e.lock"
ADMIN_PORT = 18005          # distinct from test_backend_e2e's 18003/18004
MODEL_NAME = "qwen3.5-4b"   # maps to Llama-3.2-1B-Instruct in _MODELS[0]


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def e2e_serialized():
    """Hold an exclusive cross-process lock for the whole test duration."""
    fd = os.open(E2E_LOCK, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)      # blocks until no other e2e holds it
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@pytest.fixture()
def streaming_server(e2e_serialized, request):
    """One server per test on a dedicated port; lock held across its life."""
    combo_id = request.param[0] if hasattr(request, "param") else "process-vulkan"
    backend_name = request.param[1] if hasattr(request, "param") else "vulkan-process"

    config = _build_e2e_config(combo_id, backend_name)
    proxy, client = _start_server(ADMIN_PORT, config, combo_id)

    try:
        yield {"server": proxy, "client": client,
               "base_url": f"http://127.0.0.1:{ADMIN_PORT}",
               "combo_id": combo_id}
    finally:
        # Guaranteed cleanup even on assertion failure — mirrors e2e_server.
        try:
            _stop_all_and_wait(client, f"http://127.0.0.1:{ADMIN_PORT}")
        finally:
            _stop_server(proxy, client, ADMIN_PORT)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _start_model_blocking(base_url: str, model_name: str):
    """Start a model and block until it reports loaded."""
    resp = httpx.post(f"{base_url}/admin/start/{model_name}", timeout=180)
    assert resp.status_code == 200, f"start rejected: {resp.text[:200]}"

    deadline = __import__("time").time() + 180
    while True:
        r = httpx.get(f"{base_url}/admin/models", timeout=10)
        for m in r.json()["models"]:
            if m["name"] == model_name and m.get("status", {}).get("value") == "loaded":
                return
        import time as _t
        if _t.time() > deadline:
            pytest.fail(f"model {model_name} not loaded within 180s")
        _t.sleep(0.5)


def _consume_stream(base_url: str, model_name: str):
    """POST a streaming chat request; return the full SSE body as text."""
    with httpx.stream("POST", f"{base_url}/v1/chat/completions", json={
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hi"}],
        "stream": True,
    }, timeout=300) as resp:
        assert resp.status_code == 200, f"chat rejected: {resp.read()[:200]}"
        return "".join(resp.iter_text())


# ── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.e2e
class TestStreamingStatsE2E:
    """Real uvicorn + real llama-server: stats must reflect actual token counts."""

    @pytest.mark.parametrize("streaming_server", COMBOS, indirect=True)
    def test_streaming_usage_recorded(self, streaming_server):
        base_url = streaming_server["base_url"]

        _start_model_blocking(base_url, MODEL_NAME)
        try:
            body = _consume_stream(base_url, MODEL_NAME)
            # Sanity: we got a real SSE stream with at least one content chunk.
            assert "data:" in body and "[DONE]" in body

            stats_resp = httpx.get(f"{base_url}/api/v1/stats", timeout=10)
            assert stats_resp.status_code == 200
            data = stats_resp.json()
            # A real usage chunk from llama-server lands here — not word estimates.
            assert data["stats"] is not None, "no stats recorded after stream"
            s = data["stats"]
            assert s["completion_tokens"] >= 1, f"got {s}"
            assert s["prompt_tokens"] >= 1, f"got {s}"
        finally:
            _stop_all_and_wait(httpx.Client(timeout=None), base_url)
