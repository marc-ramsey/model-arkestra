"""OpenAI v1-compatible server backed by ModelArkestra runners.

Usage as module entry point:
    python -m model_arkestra.server --config config.yaml --port 8080

Or embed into your own FastAPI app:
    from model_arkestra.server import ArkestraServer

    proxy = ArkestraServer("config.yaml", port=8080)
    app = proxy.get_app()  # returns the FastAPI instance
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union

try:
    from fastapi import FastAPI, HTTPException, Response
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
    status: str = "running"


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
        config_path: str,
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
        self._arkestra = ModelArkestra(
            config_path,
            ready_timeout=ready_timeout,
            broadcast_addr=broadcast_addr,
        )
        self._app: Optional[FastAPI] = None
        self._server: Any = None

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
                running = sum(1 for m in v1_data.get("data", []) if m.get("status") == "running")
            except Exception:
                running = 0

            return {"status": "ok", "uptime_seconds": time.time(), "models_running": running}

        @app.get("/v1/health")
        async def health_v1():
            return await health()

        # ── Admin subcomponent ───────────────────────────────────
        from model_arkestra.admin import ArkestraAdmin
        admin_key = self.admin_key or self._arkestra.cm.data.get("env", {}).get("ADMIN_KEY")
        self._admin = ArkestraAdmin(self, admin_key=admin_key, app=app)
        self._admin.install()

        self._app = app
        return app

    # ── Completion (non-streaming) ────────────────────────────────

    async def _complete_chat(self, model_name: str, req: ChatCompletionRequest) -> Response:
        """Blocking completion → full response string."""
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
                    first_chunk_sent = True
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
                    yield "data: [DONE]\n\n"
                    return

        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Stream error: {e}")

        if not first_chunk_sent:
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

        app = self.get_app()
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="info")
        server = uvicorn.Server(config)
        self._server = server
        await server.serve()

    async def shutdown(self) -> None:
        """Stop the proxy server and shut down all models."""
        if self._server:
            await self._server.shutdown()
        await self._arkestra.shutdown()


# ── CLI entry point ────────────────────────────────────────────────

def main() -> None:
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
        required=True,
        help="Path to YAML config file (required)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int, default=8080,
        help="HTTP port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--host", "-H",
        default="0.0.0.0",
        help='Bind address — use "127.0.0.1" for localhost-only (default: 0.0.0.0)',
    )
    parser.add_argument(
        "--ready-timeout", "-t",
        type=float, default=120.0,
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

    args = parser.parse_args()

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
