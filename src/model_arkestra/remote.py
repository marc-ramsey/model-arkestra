from __future__ import annotations
import asyncio
import logging
import aiohttp
from typing import Any, Dict, Optional
from model_arkestra.base import BaseRunner
from model_arkestra.types import RunnerState, _Model, ModelNotStarted

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

    def _headers(self, ctx: Optional[_Model] = None) -> Dict[str, str]:
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
        ctx = self._ctx
        if not ctx:
            raise ModelNotStarted(model_name)

        # Restart path: STOPPED/STOPPING → reuse port & context
        if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            new_size = inference_kwargs.get("max_log_lines", self.log_buffer_size)
            await self._before_restart(ctx, new_size)
            # _before_restart aborts early when already stopped; move to LOADING
            # so the ready transition below is legal.
            if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
                ctx.set_state("load")
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
        ctx.set_state("ready")

    async def _start_model_process(
        self, ctx: _Model, model_data: Dict[str, Any], model_name: str = ""
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

    async def _stop_model_process(self, ctx: _Model) -> None:
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

    # ── HTTP inference moved to providers/remote.py (RemoteProvider) ──
