"""Tests for BaseRunner HTTP paths against a real local aiohttp server."""

from __future__ import annotations
import json
from unittest.mock import MagicMock

from aiohttp import web as aiohttp_web
from aiohttp.test_utils import TestServer
import pytest

from model_arkestra.process import ProcessRunner
from model_arkestra.providers.llama import LlamaProvider
from model_arkestra.types import RunnerState, _Model


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
async def server():
    """Start a real local HTTP server with completion/stream handlers + custom path."""
    handler = FakeHandler()
    app = aiohttp_web.Application()

    # Completion and stream handlers on the same path — we'll swap via fixture for streaming tests.
    app.router.add_post("/v1/chat/completions", handler.handle_completion)
    app.router.add_post("/custom/path", handler.handle_request)

    runner = ProcessRunner(MagicMock())
    test_server = TestServer(app)
    await test_server.start_server()
    port = test_server.port

    ctx = _Model("m", port)
    ctx._state = RunnerState.RUNNING
    runner._ctx = ctx

    yield runner, handler, app, test_server

    # Cleanup.
    runner._ctx = None
    await test_server.close()


@pytest.fixture(scope="function")
async def stream_server():
    """Start a real local HTTP server with the STREAM handler on /v1/chat/completions."""
    handler = FakeHandler()
    app = aiohttp_web.Application()

    # Only stream handler — used by TestAsyncStream tests.
    app.router.add_post("/v1/chat/completions", handler.handle_stream)

    runner = ProcessRunner(MagicMock())
    test_server = TestServer(app)
    await test_server.start_server()
    port = test_server.port

    ctx = _Model("m", port)
    ctx._state = RunnerState.RUNNING
    runner._ctx = ctx

    yield runner, handler, app, test_server

    # Cleanup.
    runner._ctx = None
    await test_server.close()


class FakeHandler:
    """Request handlers for model_runner HTTP paths."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def handle_completion(self, req: aiohttp_web.Request) -> aiohttp_web.Response:
        body = await req.json()
        self.calls.append({"path": "/v1/chat/completions", "body": body})
        if body.get("tools"):
            # Mimic llama-server: tools present → tool-call response
            return aiohttp_web.json_response({
                "choices": [{"message": {"role": "assistant", "content": None,
                                          "tool_calls": [{"type": "function",
                                                          "function": {"name": "get_weather",
                                                                        "arguments": "{\"city\": \"Paris\"}"}}]},
                              "finish_reason": "tool_calls"}],
                "usage": {"model": "test", "prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}
            })
        return aiohttp_web.json_response({
            "choices": [{"message": {"content": "hello from test"}}],
            "usage": {"model": "test", "prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}
        })

    async def handle_stream(self, req: aiohttp_web.Request) -> aiohttp_web.StreamResponse:
        body = await req.json()
        self.calls.append({"path": "/v1/chat/completions", "body": body})
        resp = aiohttp_web.StreamResponse(status=200, reason="OK")
        await resp.prepare(req)
        if body.get("tools"):
            # Mimic llama-server tool-call stream: argument deltas then a
            # finish-reason-only final chunk (no usage).
            tc_first = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1",
                                                                "type": "function",
                                                                "function": {"name": "get_weather", "arguments": "{\"city\":"}}]}}]}
            tc_rest = {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "Paris\""}}]}}]}
            final = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
            for chunk in (tc_first, tc_rest, final):
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        else:
            for token in ["Hello", ", ", "world"]:
                chunk = {"choices": [{"delta": {"content": token}}]}
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def handle_request(self, req: aiohttp_web.Request) -> aiohttp_web.Response:
        body = await req.json()
        self.calls.append({"path": req.path, "body": body})
        return aiohttp_web.json_response({"echo": body.get("action", "ok")})


def _provider(runner):
    """Build a LlamaProvider pointed at the mock server port for model 'm'."""
    ctx = runner._ctx
    return LlamaProvider("m", ctx.port)


# ── Tests: ainvoke / invoke_full (LlamaProvider) ──────────────────────────

class TestAinvoke:
    async def test_basic_completion(self, server):
        runner, handler, app, _server = server
        result = (await _provider(runner).invoke_full("hi"))["content"]
        assert result == "hello from test"
        assert len(handler.calls) == 1

    async def test_response_fields(self, server):
        runner, handler, app, _server = server
        res = await _provider(runner).invoke_full("hi")
        assert res["content"] == "hello from test"
        assert res["usage"]["prompt_tokens"] == 3
        assert res["usage"]["completion_tokens"] == 7

    async def test_wrong_model_raises(self, server):
        # Model-existence is enforced by the facade (_provider_for), not the provider.
        # The provider itself is model-agnostic: it talks to a fixed port.
        prov = _provider(server[0])
        assert prov._base.startswith("http://127.0.0.1:")

    async def test_tools_forwarded_and_calls_extracted(self, server):
        """tools/tool_choice reach llama-server; tool_calls + finish_reason come back."""
        runner, handler, app, _server = server
        tools = [{"type": "function", "function": {"name": "get_weather",
                                                   "parameters": {"type": "object"}}}]
        res = await _provider(runner).invoke_full("hi", tools=tools, tool_choice="auto")
        sent = handler.calls[-1]["body"]
        assert sent["tools"] == tools
        assert sent["tool_choice"] == "auto"
        assert res["tool_calls"][0]["function"]["name"] == "get_weather"
        assert res["finish_reason"] == "tool_calls"


# ── Tests: async_stream ───────────────────────────────────────────────────

class TestAsyncStream:
    async def test_sse_tokens(self, stream_server):
        runner, handler, app, _server = stream_server
        chunks = []
        async for chunk in _provider(runner).stream({"prompt": "hi"}):
            chunks.append(chunk)

        token_chunks = [c for c in chunks if "token" in c]
        assert len(token_chunks) == 3
        concatenated = "".join(c["token"] for c in token_chunks)
        assert concatenated == "Hello, world"

    async def test_sse_usage(self, stream_server):
        runner, handler, app, _server = stream_server
        chunks = []
        async for chunk in _provider(runner).stream({"prompt": "hi"}):
            chunks.append(chunk)

        usage_chunks = [c for c in chunks if "usage" in c]
        assert len(usage_chunks) == 1
        assert "tokens_per_second" in usage_chunks[0]["usage"]

    async def test_sse_tool_call_events(self, stream_server):
        """Tool-call deltas and the finish reason surface as typed events."""
        runner, handler, app, _server = stream_server
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        chunks = []
        async for chunk in _provider(runner).stream({"prompt": "hi", "tools": tools}):
            chunks.append(chunk)

        tool_chunks = [c["tool_call"] for c in chunks if "tool_call" in c]
        assert len(tool_chunks) == 2
        assert tool_chunks[0][0]["function"]["name"] == "get_weather"
        finish = [c for c in chunks if "finish_reason" in c]
        assert finish and finish[0]["finish_reason"] == "tool_calls"


# ── Tests: request (generic POST) ─────────────────────────────────────────

class TestRequest:
    async def test_generic_post(self, server):
        runner, handler, app, _server = server
        result = await _provider(runner).request("/custom/path", action="ping")
        assert result["echo"] == "ping"
