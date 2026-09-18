"""Comprehensive endpoint tests for the OpenAI v1 proxy server."""
import json
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

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


def _register_v1_error_handlers(app):
    """Mirror ArkestraServer's /v1 error envelopes in the test app."""
    from fastapi import HTTPException, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    error_types = {400: "invalid_request_error", 404: "model_not_found",
                   422: "invalid_request_error"}

    def v1_resp(status_code, message):
        return JSONResponse(status_code=status_code, content={"error": {
            "message": message,
            "type": error_types.get(status_code, "server_error"),
        }})

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        if not request.url.path.startswith("/v1"):
            return JSONResponse(422, {"detail": exc.errors()})
        first = (exc.errors() or [{}])[0]
        loc = ".".join(str(x) for x in first.get("loc", []) if x not in ("body", "query"))
        msg = str(first.get("msg", "Invalid request"))
        return v1_resp(400, f"{msg} ({loc})" if loc else msg)

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException):
        if not request.url.path.startswith("/v1"):
            return JSONResponse(exc.status_code, {"detail": exc.detail})
        return v1_resp(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        if not request.url.path.startswith("/v1"):
            raise exc
        return v1_resp(500, str(exc) or "Internal server error")


def _build_app(mock_arkestra, aliases=None):
    """Build a FastAPI test app with real proxy route handlers.

    Returns (client, mock_arkestra) for easy assertion on calls.
    """
    proxy = _make_proxy(mock_arkestra, aliases=aliases)

    app = FastAPI(title="Test ArkestraServer")
    _register_v1_error_handlers(app)

    # ── POST /v1/chat/completions ───────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        model_name = proxy.openai_aliases.get(req.model, req.model)
        await mock_arkestra.start(model_name)  # unhandled exceptions → generic handler

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
                id=entry.get("name", entry.get("id", "unknown")),
                owned_by=entry.get("owned_by", "local"),
                status=entry.get("status") or {"value": "stopped"},
                context_length=entry.get("context_length", "n/a"),
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

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request):
        if request.headers.get("Content-Type", "").startswith("multipart"):
            data = await request.form()
            model_name = str(data.get("model", ""))
            audio_file = data.get("file")
            if not audio_file:
                raise HTTPException(status_code=400, detail="No file provided")
        else:
            try:
                req_body = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON body")
            if not isinstance(req_body, dict):
                raise HTTPException(status_code=400, detail="Invalid JSON body")
            model_name = str(req_body.get("model", ""))
        model_name = proxy._resolve_tagged_model(
            model_name, "asr",
            "No STT model available. Configure a model with tags: [asr]",
            legacy_key="stt-model",
        )
        return {"text": "mock transcript"}

    client = TestClient(app, raise_server_exceptions=False)
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
                "name": "qwen3-4b",
                "owned_by": "local",
                "status": {"value": "loaded"},
                "context_length": 8192,
            },
            {
                "name": "gemma-4-e2b",
                "owned_by": "local",
                "status": {"value": "sleeping"},
                "context_length": "n/a",
            },
        ],
    })
    # Non-streaming chat path uses ainvoke_full (full content+usage dict)
    mock.ainvoke_full = AsyncMock(return_value={
        "content": "Quantum entanglement is when particles connect across space.",
        "usage": {
            "model": "qwen3-4b",
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
        },
    })
    # Transcription routing: asr tag on qwen3-4b
    mock.cm = MagicMock()
    mock.cm.data = {"models": {"qwen3-4b": {"tags": ["asr"]}}}
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
        # Ollama-style status dict passes through unflattened
        assert qwen["status"] == {"value": "loaded"}
        assert qwen["context_length"] == 8192

    def test_list_models_unresolvable_context(self, mock_arkestra):
        """Models without a resolvable ctx-size report 'n/a'."""
        client, _ = _build_app(mock_arkestra)
        resp = client.get("/v1/models")
        data = resp.json()["data"]

        gemma = next(m for m in data if m["id"] == "gemma-4-e2b")
        assert gemma["status"] == {"value": "sleeping"}
        assert gemma["context_length"] == "n/a"


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
        """Missing messages → 400 with OpenAI-style error envelope."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            # missing: messages
        })
        assert resp.status_code == 400
        body = resp.json()
        assert "detail" not in body
        err = body["error"]
        assert err["type"] == "invalid_request_error"
        assert "messages" in err["message"]

    def test_messages_not_a_list_rejected(self, mock_arkestra):
        """Messages field that is not a list is rejected with 400."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": "this should be a list",
        })
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"

    def test_model_error_503_openai_shape(self, mock_arkestra):
        """Inference failures surface as 503 with the error envelope."""
        client, _ = _build_app(mock_arkestra)
        mock_arkestra.ainvoke = AsyncMock(side_effect=Exception("boom"))
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 503
        body = resp.json()
        assert "detail" not in body
        assert body["error"]["type"] == "server_error"
        assert "boom" in body["error"]["message"]

    def test_unhandled_exception_500_openai_shape(self, mock_arkestra):
        """Unexpected exceptions on /v1 routes → 500 with error envelope."""
        client, _ = _build_app(mock_arkestra)
        mock_arkestra.start = AsyncMock(side_effect=RuntimeError("kaboom"))
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 500
        body = resp.json()
        assert "detail" not in body
        assert body["error"]["type"] == "server_error"
        assert "kaboom" in body["error"]["message"]


# ═══════════════════════════════════════════════════════════════
# POST /v1/audio/transcriptions — request handling
# ═══════════════════════════════════════════════════════════════


class TestTranscriptions:
    """Tests for the ASR endpoint request validation."""

    def test_empty_body_rejected_400(self, mock_arkestra):
        """Empty (non-multipart) body → 400, not a raw 500."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post("/v1/audio/transcriptions", content=b"")
        assert resp.status_code == 400
        body = resp.json()
        assert "detail" not in body
        assert body["error"]["type"] == "invalid_request_error"

    def test_invalid_json_body_rejected_400(self, mock_arkestra):
        """Malformed JSON body → 400 with error envelope."""
        client, _ = _build_app(mock_arkestra)
        resp = client.post(
            "/v1/audio/transcriptions",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"

    def test_no_asr_model_404(self, mock_arkestra):
        """No model with 'asr' tag → 404 model_not_found."""
        client, _ = _build_app(mock_arkestra)
        mock_arkestra.cm.data = {"models": {}}
        resp = client.post(
            "/v1/audio/transcriptions",
            json={"model": "", "audio_b64": "aGk="},
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["type"] == "model_not_found"
        assert "detail" not in body


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

        # Model info — status is the Ollama-style dict, context_length int or 'n/a'
        model_info = ModelInfo(
            id="qwen3-4b",
            status={"value": "loaded"},
            context_length=8192,
        )
        d = model_info.model_dump()
        assert d["id"] == "qwen3-4b"
        assert d["status"] == {"value": "loaded"}
        assert d["context_length"] == 8192

        default_info = ModelInfo(id="x")
        assert default_info.status == {"value": "stopped"}
        assert default_info.context_length == "n/a"
