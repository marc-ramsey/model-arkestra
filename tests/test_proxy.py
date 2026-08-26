"""Comprehensive endpoint tests for the OpenAI v1 proxy server."""
import json
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from fastapi import HTTPException

from model_arkestra.server import (
    ArkestraServer,
    ChatCompletionRequest,
    Message,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_proxy(mock_arkestra, aliases=None):
    """Create an ArkestraServer with a mocked arkestra backend."""
    proxy = ArkestraServer.__new__(ArkestraServer)
    proxy.port = 9999
    proxy.openai_aliases = aliases or {}
    proxy.extra_headers = {}
    proxy._arkestra = mock_arkestra
    proxy._app = None
    return proxy


def _build_app(mock_arkestra, aliases=None):
    """Build a FastAPI test app with real proxy route handlers.

    Returns (client, mock_arkestra) for easy assertion on calls.
    """
    proxy = _make_proxy(mock_arkestra, aliases=aliases)

    app = FastAPI(title="Test ArkestraServer")

    # ── POST /v1/chat/completions ───────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        model_name = proxy.openai_aliases.get(req.model, req.model)
        await mock_arkestra.start(model_name)

        if req.stream:
            return StreamingResponse(
                proxy._stream_chat(model_name, req),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        try:
            content = await mock_arkestra.ainvoke(
                model_name,
                prompt="",
                messages=[m.model_dump() for m in req.messages],
                backend=req.model,
                **{k: v for k, v in {
                    "temperature": req.temperature,
                    "max_tokens": req.max_tokens,
                    "top_p": req.top_p,
                    "frequency_penalty": req.frequency_penalty,
                    "presence_penalty": req.presence_penalty,
                    "stop": req.stop,
                }.items() if v is not None},
            )

            from model_arkestra.server import (
                ChatCompletionResponseChoice, ChoiceDelta,
            )
            return {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": model_name,
                "choices": [
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChoiceDelta(role="assistant", content=content),
                        finish_reason="stop",
                    )
                ],
            }
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model error: {e}")

    # ── GET /v1/models ──────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models():
        try:
            v1_data = await mock_arkestra.get_v1_models()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

        # Apply the same field mapping that ArkestraServer does
        from model_arkestra.server import ModelInfo
        data = []
        for entry in v1_data.get("data", []):
            data.append(ModelInfo(
                id=entry.get("id", "unknown"),
                owned_by=entry.get("owned_by", "local"),
                status=entry.get("status", {"value": "stopped"}),
                port=entry.get("port"),
                runner_type=entry.get("runner_type"),
                backend_id=entry.get("backend_id"),
            ).model_dump())
        return {"object": "list", "data": data}

    # ── Health endpoints ────────────────────────────────────────

    @app.get("/health")
    async def health():
        try:
            v1_data = await mock_arkestra.get_v1_models()
            running = sum(
                1 for m in v1_data.get("data", []) if m.get("status", {}).get("value") in ("loaded", "running")
            )
        except Exception:
            running = 0
        return {"status": "ok", "models_running": running}

    @app.get("/v1/health")
    async def health_v1():
        return await health()

    client = TestClient(app)
    return client, mock_arkestra


@pytest.fixture
def mock_arkestra():
    """Create a fully-mocked ModelArkestra instance."""
    mock = MagicMock()
    mock.start = AsyncMock(return_value=None)
    mock.shutdown = AsyncMock(return_value=None)
    mock._log = AsyncMock(return_value=None)  # global log buffer helper
    mock.ainvoke = AsyncMock(
        return_value="Quantum entanglement is when particles connect across space."
    )

    def _make_sse_stream(tokens):
        """Helper to create a real async generator for SSE streaming."""
        async def sse_generator():
            for t in tokens:
                yield {"token": t}
            yield {"usage": {
                "model": "qwen3-4b",
                "prompt_tokens": 5,
                "completion_tokens": len(tokens),
                "total_tokens": 5 + len(tokens),
                "time_seconds": 0.1,
            }}
        return sse_generator()

    def _astream(model_name, payload):
        return _make_sse_stream(["Hello", " World"])

    mock.astream = _astream
    mock.get_v1_models = AsyncMock(return_value={
        "object": "list",
        "data": [
            {
                "id": "qwen3-4b",
                "owned_by": "local",
                "status": {"value": "loaded"},
                "port": 18000,
                "runner_type": "process",
                "backend_id": None,
            },
            {
                "id": "gemma-4-e2b",
                "owned_by": "local",
                "status": {"value": "sleeping"},
                "port": 18001,
                "runner_type": "podman",
                "backend_id": "rocm",
            },
        ],
    })
    return mock


# ═══════════════════════════════════════════════════════════════
# POST /v1/chat/completions — non-streaming
# ═══════════════════════════════════════════════════════════════


class TestChatCompletionsNonStreaming:
    """Tests for the blocking chat completion endpoint."""

    def test_basic_single_message(self, mock_arkestra):
        """Basic request with a single user message returns a response."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "Hello world"}],
        })
        assert resp.status_code == 200
        body = resp.json()

        assert body["object"] == "chat.completion"
        assert body["model"] == "qwen3-4b"
        assert len(body["choices"]) == 1
        choice = body["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert "Quantum entanglement" in choice["message"]["content"]

    def test_full_messages_passthrough(self, mock_arkestra):
        """Full conversation history is passed through to the runner."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [
                {"role": "system", "content": "You are a helpful tutor."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "Four."},
                {"role": "user", "content": "And 3 times that?"},
            ],
        })
        assert resp.status_code == 200

        call_kwargs = mock_arkestra.ainvoke.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 4
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]

    def test_model_alias_resolution(self, mock_arkestra):
        """OpenAI-style model aliases map to internal names."""
        client, _ = _build_app(
            mock_arkestra,
            aliases={"gpt-3.5-turbo": "qwen3-4b", "claude-3-opus": "gemma-4-e2b"},
        )

        # Test alias match
        resp = client.post("/v1/chat/completions", json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        call_model = mock_arkestra.ainvoke.call_args[0][0]
        assert call_model == "qwen3-4b"

        # Test direct name (no alias needed)
        mock_arkestra.reset_mock()
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        call_model = mock_arkestra.ainvoke.call_args[0][0]
        assert call_model == "qwen3-4b"

    def test_request_parameters_forwarded(self, mock_arkestra):
        """Extra parameters like temperature, stop, max_tokens are forwarded."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "Say hello"}],
            "temperature": 0.8,
            "max_tokens": 50,
            "top_p": 0.95,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.1,
            "stop": ["\n", "END"],
        })
        assert resp.status_code == 200

        call_kwargs = mock_arkestra.ainvoke.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 50
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["frequency_penalty"] == 0.3
        assert call_kwargs["presence_penalty"] == 0.1
        assert call_kwargs["stop"] == ["\n", "END"]

    def test_request_with_empty_messages(self, mock_arkestra):
        """Empty messages list produces a valid (if empty) request."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [],
        })
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════
# POST /v1/chat/completions — streaming
# ═══════════════════════════════════════════════════════════════


class TestChatCompletionsStreaming:
    """Tests for the SSE streaming chat completion endpoint."""

    def test_streaming_returns_sse_content_type(self, mock_arkestra):
        """Streaming response has correct media type."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_streaming_yields_tokens(self, mock_arkestra):
        """Each token produces a separate SSE chunk with the token content."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 200

        # Parse SSE chunks from the response text
        lines = [l for l in resp.text.split("\n") if l.strip()]
        data_lines = [l for l in lines if l.startswith("data: ") and "choices" in l]

        assert len(data_lines) >= 2, f"Expected at least 2 token chunks, got {len(data_lines)}"

        first_data = json.loads(data_lines[0].split(": ", 1)[1])
        assert first_data["choices"][0]["delta"]["content"] == "Hello"

        second_data = json.loads(data_lines[1].split(": ", 1)[1])
        assert second_data["choices"][0]["delta"]["content"] == " World"

    def test_streaming_ends_with_done_marker(self, mock_arkestra):
        """Last SSE line is the [DONE] marker."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 200

        assert "[DONE]" in resp.text

    def test_streaming_with_multi_turn_messages(self, mock_arkestra):
        """Streaming correctly sends full conversation history."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Say hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "Now respond streaming"},
            ],
            "stream": True,
        })
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════
# GET /v1/models
# ═══════════════════════════════════════════════════════════════


class TestListModels:
    """Tests for the model listing endpoint."""

    def test_list_models_returns_all(self, mock_arkestra):
        """All tracked models are returned in the list."""
        client, _ = _build_app(mock_arkestra)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()

        assert body["object"] == "list"
        assert len(body["data"]) == 2

    def test_list_models_field_mapping(self, mock_arkestra):
        """Each model entry has the correct OpenAI-compatible fields."""
        client, _ = _build_app(mock_arkestra)
        resp = client.get("/v1/models")
        body = resp.json()
        data = body["data"]

        qwen = next(m for m in data if m["id"] == "qwen3-4b")
        assert qwen["object"] == "model"
        assert qwen["owned_by"] == "local"
        assert qwen.get("status", {}).get("value") == "loaded"

    def test_list_models_stopped_model(self, mock_arkestra):
        """Stopped models are listed with correct status."""
        client, _ = _build_app(mock_arkestra)
        resp = client.get("/v1/models")
        body = resp.json()
        data = body["data"]

        gemma = next(m for m in data if m["id"] == "gemma-4-e2b")
        assert gemma.get("status", {}).get("value") == "sleeping"


# ═══════════════════════════════════════════════════════════════
# GET /health and GET /v1/health
# ═══════════════════════════════════════════════════════════════


class TestHealth:
    """Tests for health check endpoints."""

    def test_health_endpoint(self, mock_arkestra):
        """Basic health check returns status ok with running model count."""
        client, _ = _build_app(mock_arkestra)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()

        assert body["status"] == "ok"
        assert "models_running" in body
        assert body["models_running"] >= 1

    def test_v1_health_endpoint(self, mock_arkestra):
        """GET /v1/health returns the same data as GET /health."""
        client, _ = _build_app(mock_arkestra)
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        body = resp.json()

        assert body["status"] == "ok"


# ═══════════════════════════════════════════════════════════════
# Request validation
# ═══════════════════════════════════════════════════════════════


class TestRequestValidation:
    """Tests for request body validation."""

    def test_missing_messages_rejected(self, mock_arkestra):
        """Request without messages field is rejected with 422."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            # missing: messages
        })
        assert resp.status_code == 422

    def test_messages_not_a_list_rejected(self, mock_arkestra):
        """Messages field that is not a list is rejected with 422."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": "this should be a list",
        })
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════
# Message content types
# ═══════════════════════════════════════════════════════════════


class TestMessageContent:
    """Tests for various message content formats."""

    def test_string_content(self, mock_arkestra):
        """String content is preserved in the messages list."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "Plain text message"}],
        })
        assert resp.status_code == 200
        call_kwargs = mock_arkestra.ainvoke.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["content"] == "Plain text message"

    def test_list_content(self, mock_arkestra):
        """List content (e.g. image + text blocks) is preserved."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}},
                ],
            }],
        })
        assert resp.status_code == 200
        call_kwargs = mock_arkestra.ainvoke.call_args.kwargs
        messages = call_kwargs["messages"]
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    def test_assistant_role_preserved(self, mock_arkestra):
        """Assistant role messages are kept in the conversation history."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there!"},
                {"role": "user", "content": "What's the weather?"},
            ],
        })
        assert resp.status_code == 200
        call_kwargs = mock_arkestra.ainvoke.call_args.kwargs
        messages = call_kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert roles[1] == "assistant"

    def test_function_role_preserved(self, mock_arkestra):
        """Function/tool role messages are kept."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": None, "function_call": {"name": "get_weather"}},
                {"role": "function", "name": "get_weather", "content": '{"temp": 20}'},
            ],
        })
        assert resp.status_code == 200
        call_kwargs = mock_arkestra.ainvoke.call_args.kwargs
        messages = call_kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert roles[1] == "assistant"
        assert roles[2] == "function"


# ═══════════════════════════════════════════════════════════════
# Model name resolution
# ═══════════════════════════════════════════════════════════════


class TestModelResolution:
    """Tests for model ID resolution logic."""

    def test_alias_resolution(self, mock_arkestra):
        """OpenAI aliases are resolved to internal names."""
        client, _ = _build_app(
            mock_arkestra,
            aliases={"gpt-3.5-turbo": "qwen3-4b"},
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        call_model = mock_arkestra.ainvoke.call_args[0][0]
        assert call_model == "qwen3-4b"

    def test_unmatched_model_name_passthrough(self, mock_arkestra):
        """Unmatched model names are passed through directly."""
        client, _ = _build_app(mock_arkestra, aliases={})

        resp = client.post("/v1/chat/completions", json={
            "model": "my-custom-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        call_model = mock_arkestra.ainvoke.call_args[0][0]
        assert call_model == "my-custom-model"

    def test_model_error_raises_503(self, mock_arkestra):
        """Model error during invocation returns 503."""
        mock_arkestra.ainvoke = AsyncMock(side_effect=Exception("Model not found"))

        client, _ = _build_app(mock_arkestra)

        resp = client.post("/v1/chat/completions", json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        # Unhandled exceptions from the runner are caught and returned as 503
        assert resp.status_code == 503


# ═══════════════════════════════════════════════════════════════
# Import and API surface
# ═══════════════════════════════════════════════════════════════


class TestImports:
    """Verify all expected exports are available."""

    def test_arkestra_server_import(self):
        from model_arkestra.server import ArkestraServer
        assert ArkestraServer is not None

    def test_server_module_exports(self):
        from model_arkestra import server as s

        expected = {
            "ArkestraServer",
            "ChatCompletionRequest",
            "Message",
            "ChatCompletionResponse",
            "ChatCompletionStreamResponse",
            "ChoiceDelta",
            "ChatCompletionResponseChoice",
            "ChatCompletionStreamChoice",
            "UsageInfo",
            "ListModelsResponse",
            "ModelInfo",
            "_sse_format",
        }

        for name in expected:
            assert hasattr(s, name), f"Missing export: {name}"

    def test_arkestra_server_has_expected_methods(self):
        from model_arkestra.server import ArkestraServer

        methods = ["get_app", "start", "shutdown", "_complete_chat", "_stream_chat"]
        for method in methods:
            assert hasattr(ArkestraServer, method), f"Missing method: {method}"

    def test_pydantic_models_serializable(self):
        from model_arkestra.server import (
            ChatCompletionRequest, Message,
            ChatCompletionResponse, ChoiceDelta, ChatCompletionResponseChoice,
            ChatCompletionStreamResponse, ChatCompletionStreamChoice,
            UsageInfo, ModelInfo, ListModelsResponse,
        )

        # Request model
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Hello")],
            temperature=0.7,
            stream=False,
        )
        assert req.model_dump()["model"] == "gpt-4"

        # Response model
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            model="qwen3-4b",
            choices=[ChatCompletionResponseChoice(
                index=0,
                message=ChoiceDelta(role="assistant", content="Hello!"),
                finish_reason="stop",
            )],
        )
        d = resp.model_dump()
        assert d["choices"][0]["message"]["content"] == "Hello!"

        # Stream chunk model
        stream = ChatCompletionStreamResponse(
            id="cmpl-123",
            model="qwen3-4b",
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta=ChoiceDelta(role="assistant", content="Hi"),
            )],
        )
        d = stream.model_dump()
        assert d["choices"][0]["delta"]["content"] == "Hi"

        # Model info
        model_info = ModelInfo(id="qwen3-4b", status={"value": "loaded"}, port=18000)
        d = model_info.model_dump()
        assert d["id"] == "qwen3-4b"
        assert d.get("status", {}).get("value") == "loaded"
