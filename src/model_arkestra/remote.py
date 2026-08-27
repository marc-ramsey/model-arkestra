from __future__ import annotations
import asyncio
import json
import logging
import time
import aiohttp
from model_arkestra.http_proxy import sse_events, parse_completion
from typing import Any, AsyncIterator, Dict, Optional
from model_arkestra.base import BaseModelRunner
from model_arkestra.types import RunnerState, _ModelContext

logger = logging.getLogger(__name__)


class RemoteModelRunner(BaseModelRunner):
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
        self._backend_id: Optional[str] = None  # set by arkestra.py
        self._remote_base_url: str = ""
        self._admin_key: Optional[str] = ""
        # Resolve base_url and admin_key from backend config
        be_id = self._backend_id or ""
        backends = (self.cm.data.get("backends") or {})
        if isinstance(backends, dict):
            be = backends.get(str(be_id), {})
            if isinstance(be, dict):
                self._remote_base_url = str(be.get("base_url", "")).rstrip("/")
                self._admin_key = be.get("admin_key") or ""

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

        if getattr(ctx, '_remote_start_ack', False):
            ctx.state = RunnerState.RUNNING
        else:
            # No-op (raw llama-server) or started but not acked — mark as loaded
            # since the proxy will validate readiness on first inference call
            ctx.state = RunnerState.RUNNING

    async def _start_model_process(
        self, ctx: _ModelContext, model_data: Dict[str, Any]
    ) -> None:
        """Proxy model start to the remote worker."""
        try:
            url = f"{ctx._remote_base_url}/v1/admin/models/{ctx.name}/start"
            body = {k: v for k, v in self._inference_kwargs.get(ctx.name, {}).items() if v is not None}

            headers = {"Content-Type": "application/json"}
            if self._admin_key:
                headers["x-admin-key"] = self._admin_key

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=body, headers=headers, timeout=30) as resp:
                    if resp.status == 404:
                        # Worker doesn't have admin routes — likely raw llama-server.
                        print(f"[REMOTE] Model {ctx.name} on {ctx._remote_base_url} (no admin route, passing through)", flush=True)
                        if self.arkestra:
                            asyncio.create_task(self.arkestra._log(
                                f"[start] model={ctx.name} remote={ctx._remote_base_url} passthrough"))
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
                                    print(f"[REMOTE] Model {ctx.name} started on {ctx._remote_base_url}", flush=True)
                                    if self.arkestra:
                                        asyncio.create_task(self.arkestra._log(
                                            f"[start] model={ctx.name} remote={ctx._remote_base_url}"))
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
            headers: Dict[str, str] = {}
            if self._admin_key:
                headers["x-admin-key"] = self._admin_key

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
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._admin_key:
            headers["x-admin-key"] = self._admin_key

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
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._admin_key:
            headers["x-admin-key"] = self._admin_key

        last_err: Exception | None = None
        for attempt in range(6):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers, timeout=120) as resp:
                        if resp.status == 503:
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
        raise RuntimeError(f"Remote server not reachable after {attempt + 1} attempts") from last_err

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
