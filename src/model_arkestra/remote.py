from __future__ import annotations
import asyncio
import logging
import aiohttp
from model_arkestra.http_proxy import sse_events, parse_completion
from typing import Any, AsyncIterator, Dict, Optional
from model_arkestra.base import BaseRunner
from model_arkestra.types import RunnerState, _ModelContext, ModelNotStarted

logger = logging.getLogger(__name__)


class RemoteRunner(BaseRunner):
    """Proxy inference and lifecycle commands to another arkestra instance.

    No local port is allocated, no binary spawned. All HTTP calls are forwarded
    to the target worker's ``base_url`` at runtime configuration time.

    When the worker runs a raw llama-server (no admin routes), start/stop become
    no-ops — inference still works via direct proxying.
    """

    def __init__(self, config_manager: Any, restart_delay: float = 0.5,
                 ready_timeout: float = 60.0, warmup_delay: Optional[float] = None,
                 **kwargs: Any):
        super().__init__(config_manager, restart_delay=restart_delay,
                         ready_timeout=ready_timeout, warmup_delay=warmup_delay,
                         **kwargs)
        # Per-model URLs/keys live on the context (ctx._remote_base_url,
        # ctx._admin_key); these runner-level values are fallbacks only.
        self._remote_base_url: str = ""
        self._admin_key: str = ""

    def _headers(self, ctx: Optional[_ModelContext] = None) -> Dict[str, str]:
        """JSON headers for worker calls, carrying the admin key when set.

        The per-model key (from the cluster config) takes precedence over the
        runner-level fallback.
        """
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        key = (getattr(ctx, "_admin_key", "") or "") if ctx is not None else ""
        key = key or self._admin_key
        if key:
            headers["x-admin-key"] = key
        return headers

    async def start(
        self,
        model_name: str,
        port: Optional[int] = None,
        backend: Optional[str] = None,
        **inference_kwargs: Any,
    ) -> None:
        """Start remote model — no local health check, only proxy to worker."""
        ctx = next((v for k, v in self._models.items() if k == model_name), None)
        if not ctx:
            raise ModelNotStarted(model_name)

        # Restart path: STOPPED/STOPPING → reuse port & context
        if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            new_size = inference_kwargs.get("max_log_lines", self.log_buffer_size)
            await self._before_restart(ctx, new_size)
            # Mark as LOADING to bypass _dispatch() shutdown check
            ctx.state = RunnerState.LOADING
        elif ctx.state == RunnerState.RUNNING:
            # Already running — store kwargs and return
            self._inference_kwargs[model_name] = inference_kwargs
            return

        self._inference_kwargs[model_name] = inference_kwargs

        # Proxy start to worker — health check happens inside _start_model_process
        await self._start_model_process(ctx, {})

        # Mark ready either way — acked start, raw-llama-server passthrough,
        # or a worker that didn't ack in time. The proxy validates readiness
        # on the first inference call.
        ctx.state = RunnerState.RUNNING

    async def _start_model_process(
        self, ctx: _ModelContext, model_data: Dict[str, Any]
    ) -> None:
        """Proxy model start to the remote worker."""
        try:
            url = f"{ctx._remote_base_url}/v1/admin/models/{ctx.name}/start"
            body = {k: v for k, v in self._inference_kwargs.get(ctx.name, {}).items() if v is not None}

            headers = self._headers(ctx)

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=body, headers=headers, timeout=30) as resp:
                    if resp.status == 404:
                        # Worker doesn't have admin routes — likely raw llama-server.
                        if self.arkestra:
                            self.arkestra.log(f"[start] model={ctx.name} remote={ctx._remote_base_url} passthrough")
                        ctx._remote_start_ack = True  # assume model will be loaded externally
                        return
                    if resp.status != 200:
                        detail = await resp.text()
                        raise RuntimeError(f"Worker start failed ({resp.status}): {detail}")

            # Health-check the remote worker before marking ready
            health_url = f"{ctx._remote_base_url}/health"
            async with aiohttp.ClientSession() as session:
                for _ in range(10):
                    try:
                        async with session.get(health_url, timeout=5) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                status = data.get("status")
                                if status in ("ok", "loaded"):
                                    if self.arkestra:
                                        self.arkestra.log(f"[start] model={ctx.name} remote={ctx._remote_base_url}")
                                    ctx._remote_start_ack = True
                                    return
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        pass
                    await asyncio.sleep(0.5)

            # Timeout — don't fail, let the caller discover via inference proxy
            ctx._remote_start_ack = False
        except Exception as e:
            logger.warning(f"Remote start proxy failed for {ctx.name}: {e}")
            ctx._remote_start_ack = False

    async def _stop_model_process(self, ctx: _ModelContext) -> None:
        """Proxy model stop to the remote worker."""
        try:
            url = f"{ctx._remote_base_url}/v1/admin/models/{ctx.name}/stop"
            headers = self._headers(ctx)

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={}, headers=headers, timeout=30) as resp:
                    if resp.status == 404:
                        return  # Worker doesn't have admin routes
                    if resp.status != 200:
                        detail = await resp.text()
                        logger.warning(f"Remote stop failed ({resp.status}) for {ctx.name}: {detail}")
        except Exception as e:
            logger.warning(f"Remote stop proxy exception for {ctx.name}: {e}")

    async def _remote_stream_chat(
        self, ctx: _ModelContext, payload: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat completions from the remote worker."""
        url = f"{ctx._remote_base_url}/v1/chat/completions"
        headers = self._headers(ctx)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=120) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    raise RuntimeError(f"Remote inference failed ({resp.status}): {detail}")

                async for event in sse_events(resp.content):
                    if "token" in event:
                        yield {"token": event["token"]}
                    elif "usage" in event:
                        yield {"usage": event["usage"]}
                    # done marker is implicit — no final chunk needed

    async def _remote_complete_chat(
        self, ctx: _ModelContext, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Complete (non-streaming) chat completion from remote worker."""
        url = f"{ctx._remote_base_url}/v1/chat/completions"
        headers = self._headers(ctx)

        last_err: Exception | None = None
        last_status: int | None = None
        for attempt in range(6):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers, timeout=120) as resp:
                        if resp.status == 503:
                            last_status = 503
                            await asyncio.sleep(2.5)
                            continue
                        if resp.status != 200:
                            raise RuntimeError(f"Remote inference failed ({resp.status})")
                        data = await resp.json()
                return parse_completion(data)
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                last_err = exc
                if attempt == 5:
                    break
                await asyncio.sleep(2.5)
        if last_status is not None:
            raise RuntimeError(f"Remote worker still returned {last_status} after 6 attempts") from last_err
        raise RuntimeError("Remote server not reachable after 6 attempts") from last_err

    async def _stream_sse(self, model_name: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Override to proxy SSE from the remote worker."""
        await self._dispatch(model_name)
        ctx = next((v for k, v in self._models.items() if k == model_name), None)
        return self._remote_stream_chat(ctx, payload)  # type: ignore[return-value]

    async def _complete_async(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Override to proxy completion to the remote worker."""
        await self._dispatch(model_name)
        ctx = next((v for k, v in self._models.items() if k == model_name), None)

        messages = None
        if "messages" in kwargs and isinstance(kwargs["messages"], (list, tuple)):
            messages = list(kwargs.pop("messages"))
        else:
            prompt = prompt or kwargs.pop("prompt", "")
            if not prompt:
                raise ValueError("Payload must contain 'prompt' or 'messages'")
            messages = [{"role": "user", "content": prompt}]

        llama_fields = self._LLAMA_FIELDS
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
        }
        payload.update({k: v for k, v in kwargs.items() if k in llama_fields and v is not None})

        return await self._remote_complete_chat(ctx, payload)  # type: ignore[return-value]

    async def embed(self, model_name: str, text: str) -> Dict[str, Any]:
        """Proxy an embedding request to the remote worker."""
        await self._dispatch(model_name)
        ctx = next((v for k, v in self._models.items() if k == model_name), None)
        url = f"{ctx._remote_base_url}/v1/embeddings"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"model": model_name, "input": text},
                                    headers=self._headers(ctx), timeout=120) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    raise RuntimeError(f"Remote embedding failed ({resp.status}): {detail}")
                return await resp.json()
