"""OpenAI v1-compatible server backed by ModelArkestra runners.

Usage as module entry point:
    python -m model_arkestra.server --config config.yaml --port 8080

Or embed into your own FastAPI app:
    from model_arkestra.server import ArkestraServer

    proxy = ArkestraServer("config.yaml", port=8080)
    app = proxy.get_app()  # returns the FastAPI instance
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import aiohttp

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
except ImportError:
    raise RuntimeError(
        "model_arkestra.server requires fastapi+uvicorn. "
        "Install with: pip install \"model-arkestra[proxy]\""
    )

try:
    import uvicorn  # optional dependency (pulled in by fastapi)
except ImportError:
    raise RuntimeError(
        "model_arkestra.server requires uvicorn. "
        'Install with: pip install "model-arkestra[proxy]"'
    )

from model_arkestra.common import resolve_config_path
from model_arkestra.http_proxy import sse_events


try:
    from pydantic import BaseModel, Field
except ImportError:
    # FastAPI pulls in pydantic; this should never happen in practice
    raise RuntimeError("pydantic is required (pulled in by fastapi)")


# ── OpenAI-compatible request/response models ───────────────────────

class Message(BaseModel):
    role: str  # "system", "user", "assistant", etc.
    content: Union[str, List[Any], None] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: bool = False


class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class Choice(BaseModel):
    index: int
    message: Optional[ChoiceDelta] = None
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChoiceDelta
    finish_reason: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-default"
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[UsageInfo] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = "cmpl-stream-default"
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"
    status: Any = "running"


class ListModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo] = []


# ── SSE helpers ──────────────────────────────────────────────────────

def _sse_format(data: Any) -> str:
    """Format data for SSE streaming."""
    if isinstance(data, dict):
        return f"data: {json.dumps(data)}\n\n"
    return f"data: {data}\n\n"


# ── Proxy ───────────────────────────────────────────────────────────

class ArkestraServer:
    """Wraps a ModelArkestra instance and exposes an OpenAI v1-compatible API.

    Args:
        config_path: Path to the YAML config file.
        port: HTTP port to listen on (default 8080).
        ready_timeout: Seconds to wait for models during startup (passed to ModelArkestra).
        openai_aliases: Dict mapping OpenAI model IDs to internal model names.
            e.g. {"gpt-4": "qwen3-4b", "claude": "llama3"}
        extra_headers: Extra response headers to inject on every response.
        admin_key: Admin panel API key — gates all /admin/* paths. Falls back
            to config.env.ADMIN_KEY if not provided.
        allow_origins: List of origins allowed for CORS (e.g. ["*"] or
            ["http://localhost:3000"]). When set, installs CORSMiddleware with
            full preflight support. Mutually exclusive with manual
            ``Access-Control-*`` headers in ``extra_headers``.

    Example embedding into an existing FastAPI app:
        proxy = ArkestraServer("config.yaml", port=8080)
        proxy.start()  # blocks until server is shut down
        # ... use your app normally ...
        await proxy.shutdown()
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        port: int = 8080,
        ready_timeout: float = 120.0,
        openai_aliases: Optional[Dict[str, str]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        admin_key: Optional[str] = None,
        broadcast_addr: Optional[str] = None,
        allow_origins: Optional[List[str]] = None,
    ):
        self.port = port
        self.openai_aliases = openai_aliases or {}
        self.extra_headers = extra_headers or {}
        self.admin_key = admin_key
        self.allow_origins = allow_origins

        from model_arkestra.arkestra import ModelArkestra
        # Resolve config path — defaults to ~/.config/arkestra/config.yaml
        resolved_path = str(resolve_config_path(config_path))
        self._arkestra = ModelArkestra(
            resolved_path,
            ready_timeout=ready_timeout,
            broadcast_addr=broadcast_addr,
        )
        self._app: Optional[FastAPI] = None
        self._server: Any = None

    # ── ONNX auxiliary model management ───────────────────────────

    def _is_onnx_model(self, model_name: str) -> bool:
        """Check if a model is an ONNX auxiliary model (embedding/whisper/tts)."""
        cfg = self._get_aux_model_cfg(model_name)
        if not cfg:
            return False
        return bool(cfg.get("type") in ("embedding", "whisper", "tts"))

    def _get_aux_model_cfg(self, model_name: str) -> Optional[Dict]:
        """Get config for a model by name."""
        if not model_name:
            return None
        cfg = self._arkestra.cm.data.get("models") or {}
        model = cfg.get(model_name)
        if isinstance(model, dict):
            return model
        # Check all keys for partial match (e.g. "default-whisper" → first key containing it)
        if model_name in cfg:
            return cfg[model_name]
        return None

    def _get_remote_base_url(self, model_name: str) -> Optional[str]:
        """Return the base-url for a remote cluster proxy, or None."""
        try:
            cluster_name, base_url, _local_id = self._arkestra.resolve_model_cluster_addr(model_name)
        except ValueError:
            return None
        return str(base_url).rstrip("/") if base_url else None

    async def _proxy_stream(self, model_name: str, payload: Dict[str, Any], base_url: str) -> AsyncIterator[str]:
        """Proxy a streaming request to a remote worker and yield SSE lines."""
        url = f"{base_url}/v1/chat/completions"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        admin_key = self.admin_key or ""
        if admin_key:
            headers["x-admin-key"] = admin_key

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=120) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    raise HTTPException(status_code=503, detail=f"Remote inference failed ({resp.status}): {detail}")

                async for event in sse_events(resp.content):
                    if "token" in event:
                        chunk = {
                            "id": "cmpl-stream-default",
                            "object": "chat.completion.chunk",
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": event["token"]}}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif "usage" in event:
                        chunk = {
                            "id": "cmpl-stream-default",
                            "object": "chat.completion.chunk",
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

    async def _proxy_complete(self, model_name: str, req: ChatCompletionRequest, base_url: str) -> Response:
        """Proxy a non-streaming request to a remote worker and return JSON."""
        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [m.model_dump() for m in req.messages],
            "stream": False,
        }
        for k, v in {"temperature": req.temperature, "max_tokens": req.max_tokens,
                      "top_p": req.top_p, "frequency_penalty": req.frequency_penalty,
                      "presence_penalty": req.presence_penalty,
                      "stop": req.stop}.items():
            if v is not None:
                payload[k] = v

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        admin_key = self.admin_key or ""
        if admin_key:
            headers["x-admin-key"] = admin_key

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload, headers=headers, timeout=120) as resp:
                    if resp.status != 200:
                        detail = await resp.text()
                        raise HTTPException(status_code=503, detail=f"Remote inference failed ({resp.status}): {detail}")
                    data = await resp.json()
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Remote server error: {e}")

    # ── FastAPI app factory ───────────────────────────────────────

    def get_app(self) -> FastAPI:
        """Return the configured FastAPI application."""
        if self._app is not None:
            return self._app

        @asynccontextmanager
        async def _lifespan(app):
            yield
            await self._arkestra.shutdown()

        app = FastAPI(
            title="ModelArkestra",
            description="OpenAI v1-compatible proxy backed by local LLM runners.",
            version="0.1.0",
            lifespan=_lifespan,
        )

        if self.allow_origins:
            # Full CORS middleware — handles preflight OPTIONS and all standard
            # access-control headers automatically.
            app.add_middleware(CORSMiddleware, allow_origins=self.allow_origins)
        elif self.extra_headers:
            @app.middleware("http")
            async def _add_headers(request, call_next):
                response = await call_next(request)
                for key, value in self.extra_headers.items():
                    response.headers[key] = value
                return response

        # ── Route: POST /v1/chat/completions ──────────────────────

        @app.post("/v1/chat/completions")
        async def chat_completions(req: ChatCompletionRequest):
            model_name = self.openai_aliases.get(req.model, req.model)
            # Only start if not already running — avoid consuming extra ports
            if not any(ctx.name == model_name for ctx in self._arkestra._get_model_contexts()):
                try:
                    await self._arkestra.start(model_name)
                except Exception as e:
                    raise HTTPException(status_code=503, detail=f"Model error: {e}")

            # Detect remote models — proxy directly to the worker
            base_url = self._get_remote_base_url(model_name)
            if base_url:
                payload = {
                    "model": model_name,
                    "messages": [m.model_dump() for m in req.messages],
                }
                for k, v in {"temperature": req.temperature, "max_tokens": req.max_tokens,
                              "top_p": req.top_p, "frequency_penalty": req.frequency_penalty,
                              "presence_penalty": req.presence_penalty,
                              "stop": req.stop}.items():
                    if v is not None:
                        payload[k] = v
                if req.stream:
                    return StreamingResponse(
                        self._proxy_stream(model_name, payload, base_url),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                    )
                else:
                    return await self._proxy_complete(model_name, req, base_url)

            if req.stream:
                return StreamingResponse(
                    self._stream_chat(model_name, req),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            else:
                response_data = await self._complete_chat(model_name, req)
                return response_data

        # ── Route: GET /v1/models ─────────────────────────────────

        @app.get("/v1/models")
        async def list_models():
            try:
                v1_data = self._arkestra.get_v1_models()
            except Exception as e:
                raise HTTPException(status_code=503, detail=str(e))

            models: List[ModelInfo] = []
            for entry in v1_data.get("data", []):
                models.append(ModelInfo(
                    id=entry.get("id", "unknown"),
                    owned_by=entry.get("owned_by", "local"),
                    status=entry.get("status", "stopped"),
                ))

            return ListModelsResponse(data=models).model_dump()

        # ── Route: GET /health (and aliased /v1/health) ───────────

        @app.get("/health")
        async def health():
            try:
                v1_data = self._arkestra.get_v1_models()
                running = sum(1 for m in v1_data.get("data", []) if m.get("status", {}).get("value") == "loaded")
            except Exception:
                running = 0

            return {"status": "ok", "uptime_seconds": time.time(), "models_running": running}

        @app.get("/v1/health")
        async def health_v1():
            return await health()

        # ── Route: POST /v1/embeddings (ONNX auxiliary models) ───────

        @app.post("/v1/embeddings")
        async def embeddings_endpoint(
            req_body: Dict[str, Any],
        ) -> Any:
            """Embedding endpoint — routes to ONNX auxiliary model.

            Expects {\'model\': \'<name>\', \'input\': \'text\'} where \<name>
            is an ONNX model configured with type: embedding.
            Falls through to chat if no matching aux model found.
            """
            model_name = req_body.get("model", "")

            # Check for remote model first — proxy directly
            base_url = self._get_remote_base_url(model_name)
            if base_url:
                payload = {"model": model_name, "input": input_text}
                headers: Dict[str, str] = {"Content-Type": "application/json"}
                admin_key = self.admin_key or ""
                if admin_key:
                    headers["x-admin-key"] = admin_key

                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{base_url}/v1/embeddings", json=payload, headers=headers, timeout=30) as resp:
                        if resp.status != 200:
                            detail = await resp.text()
                            raise HTTPException(status_code=503, detail=f"Remote embed failed ({resp.status})")
                        return Response(content=await resp.text(), media_type="application/json")

            cfg = self._get_aux_model_cfg(model_name)
            if not cfg:
                raise HTTPException(
                    status_code=404,
                    detail=(f"Model '{model_name}' not found. "
                            "Use a model configured with type: embedding."),
                )

            input_text = req_body.get("input", "")
            try:
                result = await self._arkestra.embed(model_name, input_text)
            except Exception as e:
                # Auto-start ONNX model on first access if not yet loaded
                if "not started" in str(e).lower() or "not found in config" not in str(e).lower():
                    await self._arkestra.start(model_name)
                    result = await self._arkestra.embed(model_name, input_text)
                else:
                    raise HTTPException(status_code=500, detail=str(e))
            return Response(content=json.dumps(result), media_type="application/json")

        # ── Route: POST /v1/audio/transcriptions (ONNX auxiliary models) ───

        @app.post("/v1/audio/transcriptions")
        async def transcriptions_endpoint(
            request: Request,
        ) -> Any:
            """Audio transcription endpoint — routes to ONNX Whisper model.

            Expects multipart form (file) or JSON with {\'model\': \'<name>\', 'audio_b64': ...}
            where \<name> is configured with type: whisper.
            """
            if request.headers.get("Content-Type", "").startswith("multipart"):
                data = await request.form()
                model_name = str(data.get("model", "default-whisper"))
                audio_file = data.get("file")
                if not audio_file:
                    raise HTTPException(status_code=400, detail="No file provided")
                audio_bytes = await audio_file.read()
            else:
                req_body = await request.json()
                model_name = str(req_body.get("model", "default-whisper"))
                b64_audio = req_body.get("audio_b64", req_body.get("audio", ""))
                if isinstance(b64_audio, str):
                    import base64
                    try:
                        audio_bytes = base64.b64decode(b64_audio)
                    except Exception:
                        raise HTTPException(status_code=400, detail="Invalid audio data")
                elif isinstance(b64_audio, bytes):
                    audio_bytes = b64_audio
                else:
                    audio_bytes = b""

            cfg = self._get_aux_model_cfg(model_name)
            if not cfg:
                # Try "default-whisper" key
                cfg = self._get_aux_model_cfg("default-whisper")
            if not cfg:
                raise HTTPException(
                    status_code=404,
                    detail=f"Whisper model '{model_name}' not found.")

            lang = None
            if request.headers.get("Content-Type", "").startswith("multipart"):
                lang = str(data.get("language")) if data.get("language") else None
            else:
                lang = str(req_body.get("language")) if req_body.get("language") else None

            try:
                result = await self._arkestra.transcribe(model_name, audio_bytes, lang)
            except Exception as e:
                if "not started" in str(e).lower() or "not found in config" not in str(e).lower():
                    await self._arkestra.start(model_name)
                    result = await self._arkestra.transcribe(model_name, audio_bytes, lang)
                else:
                    raise HTTPException(status_code=500, detail=str(e))
            return Response(content=json.dumps(result), media_type="application/json")

        # ── Route: POST /v1/audio/speech (ONNX auxiliary models) ───────

        @app.post("/v1/audio/speech")
        async def speech_endpoint(
            req_body: Dict[str, Any],
        ) -> Any:
            """Text-to-speech endpoint — routes to ONNX TTS model.

            Expects {\'model\': \'<name>\', \'input\': \'text\'} where \<name>
            is configured with type: tts.
            Returns WAV audio bytes.
            """
            model_name = req_body.get("model", "default-tts")
            cfg = self._get_aux_model_cfg(model_name)
            if not cfg:
                cfg = self._get_aux_model_cfg("default-tts")
            if not cfg:
                raise HTTPException(
                    status_code=404,
                    detail=f"TTS model '{model_name}' not found.")

            text_input = req_body.get("input", "")
            try:
                wav_bytes = await self._arkestra.synthesize(model_name, text_input)
            except Exception as e:
                if "not started" in str(e).lower() or "not found in config" not in str(e).lower():
                    await self._arkestra.start(model_name)
                    wav_bytes = await self._arkestra.synthesize(model_name, text_input)
                else:
                    raise HTTPException(status_code=500, detail=str(e))
            return Response(content=wav_bytes, media_type="audio/wav")

        # ── Admin subcomponent ───────────────────────────────────
        from model_arkestra.admin import ArkestraAdmin
        admin_key = self._arkestra.resolve_config("ADMIN_KEY", explicit=self.admin_key)
        self._admin = ArkestraAdmin(self, admin_key=admin_key, app=app)
        self._admin.install()

        self._app = app
        return app

    # ── Completion (non-streaming) ────────────────────────────────

    async def _complete_chat(self, model_name: str, req: ChatCompletionRequest) -> Response:
        """Blocking completion → full response string."""
        t0 = time.monotonic()
        try:
            content = await self._arkestra.ainvoke(
                model_name,
                prompt="",
                messages=[m.model_dump() for m in req.messages],
                backend=req.model,  # allow model field as backend hint if needed
                **{k: v for k, v in {
                    "temperature": req.temperature,
                    "max_tokens": req.max_tokens,
                    "top_p": req.top_p,
                    "frequency_penalty": req.frequency_penalty,
                    "presence_penalty": req.presence_penalty,
                    "stop": req.stop,
                }.items() if v is not None},
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model error: {e}")

        latency_ms = round((time.monotonic() - t0) * 1000)
        tokens = len(content.split())
        self._arkestra.log(f"[action=req model={model_name} method=POST path=/v1/chat/completions status=200 latency_ms={latency_ms} tokens={tokens}]")

        return Response(
            content=ChatCompletionResponse(
                id="chatcmpl-default",
                model=model_name,
                choices=[ChatCompletionResponseChoice(
                    index=0,
                    message=ChoiceDelta(role="assistant", content=content),
                    finish_reason="stop",
                )],
                usage=UsageInfo(total_tokens=len(content.split()) + 10),  # rough estimate
            ).model_dump_json(),
            media_type="application/json",
        )

    # ── Completion (streaming) ────────────────────────────────────

    async def _stream_chat(self, model_name: str, req: ChatCompletionRequest) -> AsyncIterator[str]:
        """Streaming completion → SSE token events."""
        t0 = time.monotonic()
        first_token_time = None
        msg_count = len(req.messages)
        tokens_seen = 0
        first_chunk_sent = False

        try:
            async for event in self._arkestra.astream(
                model_name,
                payload={
                    "messages": [m.model_dump() for m in req.messages],
                    **{k: v for k, v in {
                        "temperature": req.temperature,
                        "max_tokens": req.max_tokens,
                        "top_p": req.top_p,
                        "frequency_penalty": req.frequency_penalty,
                        "presence_penalty": req.presence_penalty,
                        "stop": req.stop,
                    }.items() if v is not None},
                },
            ):
                if "token" in event:
                    if not first_chunk_sent:
                        first_token_time = time.monotonic()
                        self._arkestra.log(f"[action=stream_start model={model_name} messages={msg_count}]")
                        first_chunk_sent = True
                    tokens_seen += 1
                    chunk = ChatCompletionStreamResponse(
                        model=model_name,
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta=ChoiceDelta(role="assistant", content=event["token"]),
                        )],
                    )
                    yield _sse_format(chunk.model_dump())

                elif "usage" in event:
                    usage = event["usage"]
                    chunk = ChatCompletionStreamResponse(
                        model=model_name,
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta=ChoiceDelta(),
                            finish_reason="stop",
                        )],
                    )
                    yield _sse_format(chunk.model_dump())
                    # Send [DONE] marker
                    latency_ms = round((time.monotonic() - t0) * 1000)
                    self._arkestra.log(f"[action=stream_end model={model_name} duration_ms={latency_ms} tokens={tokens_seen}] status=ok")
                    yield "data: [DONE]\n\n"
                    return

        except Exception as e:
            latency_ms = round((time.monotonic() - t0) * 1000)
            self._arkestra.log(f"[action=stream_end model={model_name} duration_ms={latency_ms} tokens={tokens_seen}] status=error")
            raise HTTPException(status_code=503, detail=f"Stream error: {e}")

        if not first_chunk_sent:
            latency_ms = round((time.monotonic() - t0) * 1000)
            self._arkestra.log(f"[action=stream_end model={model_name} duration_ms={latency_ms} tokens={tokens_seen}] status=no_tokens")
            # No tokens produced — send an empty final chunk
            chunk = ChatCompletionStreamResponse(
                model=model_name,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                )],
            )
            yield _sse_format(chunk.model_dump())
            yield "data: [DONE]\n\n"

    # ── Lifecycle management ──────────────────────────────────────

    async def start(self) -> None:
        """Start the proxy server. Blocks until shutdown is called.

        Sets ``self._server`` so the admin shutdown route can stop uvicorn cleanly.
        """
        self._arkestra.log(f"[action=start server port={self.port}]")

        app = self.get_app()
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="info")
        server = uvicorn.Server(config)
        self._server = server
        try:
            await server.serve()
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Uvicorn already handled shutdown (SIGINT/SIGTERM).
            pass

    async def shutdown(self) -> None:
        """Stop the proxy server and shut down all models."""
        self._arkestra.log(f"[action=shutdown server]")
        if self._server:
            await self._server.shutdown()
        await self._arkestra.shutdown()


# ── CLI entry point ────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Entry point for: python -m model_arkestra.server

    Runs a FastAPI server backed by ModelArkestra, exposing an OpenAI-compatible
    /v1/chat/completions endpoint.  All parameters have sensible defaults so you
    can start with just --config and everything else is optional.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="model-arkestra-proxy",
        description="ModelArkestra OpenAI-compatible server",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file (default: ~/.config/arkestra/config.yaml)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int, default=None,
        help="HTTP port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--host", "-H",
        default=None,
        help='Bind address — use "127.0.0.1" for localhost-only (default: 0.0.0.0)',
    )
    parser.add_argument(
        "--ready-timeout", "-t",
        type=float, default=None,
        help='Seconds to wait for models during startup (default: 120)',
    )
    parser.add_argument(
        "--alias", "-a",
        action="append", default=[],
        metavar="KEY=VALUE",
        help=(
            'OpenAI model alias mapping in KEY=VALUE form. Repeat for multiple '
            'e.g. -a gpt-4=qwen3.5-4b -a claude=gemma-4-e2b'
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Require this Bearer token on every request (basic auth bypass)",
    )
    parser.add_argument(
        "--cors",
        action="store_true", default=False,
        help="Enable full CORS via CORSMiddleware — handles preflight OPTIONS and all standard Access-Control headers.",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to TLS certificate file (PEM)",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to TLS private key file (PEM)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help='Uvicorn log level (default: info)',
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, default=1,
        help='Number of worker processes (1 for async, >1 for multiprocessing; default: 1)',
    )
    parser.add_argument(
        "--broadcast-addr",
        default=None,
        help=(
            'Address models bind to — "0.0.0.0" for all interfaces, "127.0.0.1" for '
            'localhost only. Overrides config.runners.broadcast_addr. Default: auto (0.0.0.0)'
        ),
    )

    args = parser.parse_args(argv)

    # Load config early so we can resolve defaults from it.
    import yaml

    resolved_path = str(resolve_config_path(args.config))
    try:
        with open(resolved_path) as f:
            cfg_data: dict = yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError):
        cfg_data = {}

    # ── Resolve args in order: CLI > env > config > hardwired default ───
    if args.port is None:
        env_port = os.environ.get("PORT")
        if env_port:
            try:
                args.port = int(env_port)
            except ValueError:
                pass
        if args.port is None:
            args.port = cfg_data.get("admin-port") or 8080
    if args.host is None:
        env_host = os.environ.get("HOST")
        if env_host:
            args.host = env_host
        else:
            # No dedicated config key for server host — fall back to default
            args.host = "0.0.0.0"
    if args.ready_timeout is None:
        cfg_to = cfg_data.get("warmup-time")
        if cfg_to is not None:
            try:
                args.ready_timeout = float(cfg_to)
            except (ValueError, TypeError):
                pass
        else:
            args.ready_timeout = 120.0

    # Parse aliases: --alias gpt-4=qwen3.5-4b → {"gpt-4": "qwen3.5-4b"}
    aliases: dict[str, str] = {}
    for item in args.alias:
        if "=" not in item:
            parser.error(f"Invalid alias format (expect KEY=VALUE): {item}")
        k, v = item.split("=", 1)
        aliases[k.strip()] = v.strip()

    # Build the proxy — lazy app creation happens on get_app()
    proxy = ArkestraServer(
        config_path=args.config,
        port=args.port,
        ready_timeout=args.ready_timeout,
        openai_aliases=aliases,
        allow_origins=["*"] if args.cors else None,
        broadcast_addr=args.broadcast_addr,
    )
    app = proxy.get_app()

    # ── Optional API key middleware (runs before every request) ────────
    if args.api_key:
        from fastapi import Request
        from fastapi.responses import JSONResponse

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            header = request.headers.get("Authorization", "")
            if not header.startswith("Bearer ") or header[7:] != args.api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid API key"},
                )
            return await call_next(request)

    # ── Startup banner ────────────────────────────────────────────────
    scheme = "https" if args.ssl_certfile else "http"
    print(f"ModelArkestra OpenAI Server")
    print(f"  URL       → {scheme}://{args.host}:{args.port}")
    print(f"  API docs  → {scheme}://{args.host}:{args.port}/docs")
    if aliases:
        for k, v in aliases.items():
            print(f"  Alias     {k:20s} → {v}")
    if args.api_key:
        print(f"  Auth      ● API key required")
    print()

    # ── Launch uvicorn ────────────────────────────────────────────────
    uvicorn_kwargs = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "workers": args.workers,
    }
    if args.ssl_certfile and args.ssl_keyfile:
        uvicorn_kwargs["ssl_certfile"] = args.ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = args.ssl_keyfile

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
