"""Tests for BaseModelRunner HTTP paths against a real local aiohttp server."""

from __future__ import annotations
import json
from unittest.mock import MagicMock

from aiohttp import web as aiohttp_web
from aiohttp.test_utils import TestServer
import pytest

from model_arkestra.process import ProcessModelRunner
from model_arkestra.types import RunnerState, _ModelContext


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
async def server():
    """Start a real local HTTP server with completion/stream handlers + custom path."""
    handler = FakeHandler()
    app = aiohttp_web.Application()

    # Completion and stream handlers on the same path — we'll swap via fixture for streaming tests.
    app.router.add_post("/v1/chat/completions", handler.handle_completion)
    app.router.add_post("/custom/path", handler.handle_request)

    runner = ProcessModelRunner(MagicMock())
    test_server = TestServer(app)
    await test_server.start_server()
    port = test_server.port

    ctx = _ModelContext("m", port)
    ctx.state = RunnerState.RUNNING
    runner._models["m"] = ctx

    yield runner, handler, app, test_server

    # Cleanup.
    runner._models.clear()
    await test_server.close()


@pytest.fixture(scope="function")
async def stream_server():
    """Start a real local HTTP server with the STREAM handler on /v1/chat/completions."""
    handler = FakeHandler()
    app = aiohttp_web.Application()

    # Only stream handler — used by TestAsyncStream tests.
    app.router.add_post("/v1/chat/completions", handler.handle_stream)

    runner = ProcessModelRunner(MagicMock())
    test_server = TestServer(app)
    await test_server.start_server()
    port = test_server.port

    ctx = _ModelContext("m", port)
    ctx.state = RunnerState.RUNNING
    runner._models["m"] = ctx

    yield runner, handler, app, test_server

    # Cleanup.
    runner._models.clear()
    await test_server.close()


class FakeHandler:
    """Request handlers for model_runner HTTP paths."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def handle_completion(self, req: aiohttp_web.Request) -> aiohttp_web.Response:
        body = await req.json()
        self.calls.append({"path": "/v1/chat/completions", "body": body})
        return aiohttp_web.json_response({
            "choices": [{"message": {"content": "hello from test"}}],
            "usage": {"model": "test", "prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}
        })

    async def handle_stream(self, req: aiohttp_web.Request) -> aiohttp_web.StreamResponse:
        body = await req.json()
        self.calls.append({"path": "/v1/chat/completions", "body": body})
        resp = aiohttp_web.StreamResponse(status=200, reason="OK")
        await resp.prepare(req)
        for token in ["Hello", ", ", "world"]:
            chunk = {"choices": [{"delta": {"content": token}}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def handle_request(self, req: aiohttp_web.Request) -> aiohttp_web.Response:
        body = await req.json()
        self.calls.append({"path": req.path, "body": body})
        return aiohttp_web.json_response({"echo": body.get("action", "ok")})


# ── Tests: ainvoke / _complete_async ──────────────────────────────────────

class TestAinvoke:
    async def test_basic_completion(self, server):
        runner, handler, app, _server = server
        result = await runner.ainvoke("m", "hi")
        assert result == "hello from test"
        assert len(handler.calls) == 1

    async def test_response_fields(self, server):
        runner, handler, app, _server = server
        res = await runner._complete_async("m", "hi")
        assert res["content"] == "hello from test"
        assert res["usage"]["prompt_tokens"] == 3
        assert res["usage"]["completion_tokens"] == 7

    async def test_wrong_model_raises(self, server):
        runner, handler, app, _server = server
        with pytest.raises(Exception):  # ModelNotStarted
            await runner.ainvoke("missing", "hi")


# ── Tests: async_stream ───────────────────────────────────────────────────

class TestAsyncStream:
    async def test_sse_tokens(self, stream_server):
        runner, handler, app, _server = stream_server
        chunks = []
        async for chunk in runner.astream("m", {"prompt": "hi"}):
            chunks.append(chunk)

        token_chunks = [c for c in chunks if "token" in c]
        assert len(token_chunks) == 3
        concatenated = "".join(c["token"] for c in token_chunks)
        assert concatenated == "Hello, world"

    async def test_sse_usage(self, stream_server):
        runner, handler, app, _server = stream_server
        chunks = []
        async for chunk in runner.astream("m", {"prompt": "hi"}):
            chunks.append(chunk)

        usage_chunks = [c for c in chunks if "usage" in c]
        assert len(usage_chunks) == 1
        assert "tokens_per_second" in usage_chunks[0]["usage"]


# ── Tests: request (generic POST) ─────────────────────────────────────────

class TestRequest:
    async def test_generic_post(self, server):
        runner, handler, app, _server = server
        result = await runner.request("m", "/custom/path", action="ping")
        assert result["echo"] == "ping"
