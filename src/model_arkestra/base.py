from __future__ import annotations
import asyncio
import logging
import json
import time
import os
import socket
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set
import aiohttp

from model_arkestra.common import _resolve_backend, default_cache_root, resolve_model_ref
from model_arkestra.unicode_ringbuffer import UnicodeRingBuffer
from model_arkestra.types import (
    RunnerState, RunnerError, ServerReadyTimeout, 
    ModelNotStarted, MaxRestartsExceeded, ModelShutdown, _Model
)

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    LOG_BUFFER_DEFAULT = 2000  # max lines in model log ring buffer
    MODEL_START_TIMEOUT = 300  # seconds to wait for RUNNING state after start
    _DEFAULT_BACKEND = "cpu"
    _DEFAULT_RUNNER = "process"

    def __init__(self, config_manager: Any, restart_delay: float = 5.0,
                 restart_limit: int = 4, shutdown_timeout: float = 20.0,
                 ready_timeout: float = 120.0, ready_poll_ms: float = 100.0,
                 warmup_delay: Optional[float] = None, port_drain_timeout: float = 20.0,
                 broadcast_addr: str = "0.0.0.0",
                 log_buffer_size: Optional[int] = None,
                 arkestra: Any = None):
        self.cm = config_manager
        self.arkestra = arkestra
        self.restart_delay = restart_delay
        self.restart_limit = restart_limit
        self.shutdown_timeout = shutdown_timeout
        self.ready_timeout = ready_timeout
        self.ready_poll_ms = ready_poll_ms
        self.warmup_delay = warmup_delay if warmup_delay is not None else config_manager.data.get("warmup-time", 10.0)
        self.port_drain_timeout = port_drain_timeout
        # Resolve broadcast_addr: explicit param > config fallback > global default
        cfg_br = (self.cm.data.get("runners") or {}).get("broadcast_addr")
        self.broadcast_addr = broadcast_addr if broadcast_addr is not None else (cfg_br or "0.0.0.0")
        # Resolve log_buffer_size: explicit param > config key > class default
        cfg_lbs = self.cm.data.get('log-buffer-size')
        self.log_buffer_size = log_buffer_size if log_buffer_size is not None else (cfg_lbs or self.LOG_BUFFER_DEFAULT)
        self._watchers: Dict[str, asyncio.Task] = {}
        self._health_task: Optional[asyncio.Task] = None
        self._inference_kwargs: Dict[str, Dict[str, Any]] = {}
        self._models: Dict[str, _Model] = {}

    async def _ensure_port_available(self, port: int) -> None:
        """Raise RuntimeError immediately if *port* is already in use."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: socket.create_connection(("127.0.0.1", port), timeout=1)
            )
            raise RunnerError(f"Port {port} is already in use")
        except ConnectionRefusedError:
            pass  # port free — nothing to do
        except OSError as e:
            # Connection refused, unreachable, or timeout → port free
            if "[Errno 111]" not in str(e) and "[Errno 61]" not in str(e):
                pass  # non-refusal OS errors are also fine (port free)

    async def _release_port(self, port: int) -> None:
        """Subclasses may override to wait for the underlying listener to drain
        before discarding ownership."""
        pass

    @property
    def running_models(self) -> Set[str]:
        return {key for key, ctx in self._models.items() if ctx.state == RunnerState.RUNNING}

    async def __aenter__(self) -> BaseRunner:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    # ── Internal Core ──────────────────────────────────────

    async def _dispatch(self, model_name: str) -> None:
        ctx = self._models.get(model_name)
        if not ctx:
            raise ModelNotStarted(model_name)
        if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            raise ModelShutdown(f"Model '{model_name}' was stopped.")
        if ctx.state == RunnerState.ERROR:
            raise MaxRestartsExceeded(
                f"Model '{model_name}' exceeded restart limit after {ctx.restart_count} attempts"
            )

    async def _watch_process(self, model_name: str, ctx: _Model) -> None:
        """Background task to monitor process lifecycle and restart on unexpected exit."""
        if ctx.process is None:
            logger.error(f"Model {model_name}: no process to watch")
            return

        try:
            while True:
                try:
                    exit_code = await ctx.process.wait()
                except Exception as e:
                    logger.error(f"Error waiting for process {model_name}: {e}")
                    return

                if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
                    return

                logger.warning(
                    f"Model {model_name} exited unexpectedly with code {exit_code}"
                )

                try:
                    await self._handle_restart(model_name, ctx, exit_code)
                    if ctx.state == RunnerState.ERROR:
                        return
                except Exception as e:
                    logger.error(f"Model {model_name}: restart error — {e}")
                    return
        except asyncio.CancelledError:
            pass

    async def _handle_restart(self, model_name: str, ctx: _Model,
                              exit_code: int) -> None:
        if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            return

        ctx.restart_count += 1
        if ctx.restart_count >= self.restart_limit:
            logger.error(
                f"Model {model_name}: restart limit ({self.restart_limit}) "
                f"exceeded after {ctx.restart_count} attempts"
            )
            ctx.set_state("crash_limit")
            return

        logger.info(
            f"Model {model_name}: restarting in {self.restart_delay}s "
            f"(attempt {ctx.restart_count}/{self.restart_limit})"
        )
        await asyncio.sleep(self.restart_delay)

        if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            return

        if not await self._before_restart(ctx):
            return

        try:
            model_data = self.cm.get_model(model_name, env_vars={"PORT": str(ctx.port)})
            if model_data is None:
                logger.error(f"Model {model_name}: config vanished during restart")
                return
            await self._start_model_process(ctx, model_data)
        except Exception as e:
            logger.error(f"Model {model_name}: restart failed — {e}")

    async def _before_restart(self, ctx: _Model, new_size: Optional[int] = None) -> bool:
        """Prepare context for restart: transition state and manage buffer."""
        if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            return False
        ctx.set_state("load")

        # Determine desired size; crash-restart path gets current size (no change).
        if new_size is None:
            new_size = getattr(ctx, '_log_buf_lines', self.log_buffer_size)
            if not isinstance(new_size, int) or new_size == 0:
                new_size = self.log_buffer_size

        # Allocate a fresh ring buffer — no tail copy needed.
        # Guard: only recreate the ring buffer if new_size is actually an int.
        # (MagicMock-based test runners may produce non-int values otherwise.)
        if isinstance(new_size, int):
            ctx._log_ring = UnicodeRingBuffer(new_size * ctx.AVG_LINE_BYTES)
            ctx._log_buf_lines = new_size

        return True

    # ── Health watcher (shared across all models in this runner) ─────
    async def start_health_watcher(self, interval: float = 1.0) -> None:
        """Start a background task that polls /health for every active model."""
        if self._health_task is not None:
            return
        self._health_task = asyncio.create_task(
            self._health_watch_loop(interval)
        )

    async def _health_watch_loop(self, interval: float) -> None:
        """Loop that checks /health on all RUNNING models every *interval* seconds."""
        while True:
            for model_name, ctx in list(self._models.items()):
                if ctx.state != RunnerState.RUNNING or not ctx.port:
                    continue
                try:
                    async with aiohttp.ClientSession() as session:
                        resp = await session.get(
                            f"http://127.0.0.1:{ctx.port}/health", timeout=3
                        )
                        if resp.status == 200:
                            data = await resp.json()
                            status = data.get("status")
                            if status is None:
                                # Flat "ok" response — still healthy
                                continue
                            value = status.get("value", status) if isinstance(status, dict) else status
                            if value == "loading":
                                ctx.set_state("health_loading")
                            elif value == "error":
                                ctx.set_state("health_error")
                        # 5xx → server died, _watch_process will detect it later
                except Exception:
                    pass  # unreachable — will be caught by process watcher
            await asyncio.sleep(interval)

    async def shutdown_health_watcher(self) -> None:
        """Cancel the shared health watch loop."""
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

    # ── Container lifecycle hook (abstracted from process) ───────────
    async def _watch_container(self, model_name: str, ctx: _Model) -> None:
        """Monitor a detached container's lifecycle and restart on exit."""
        raise NotImplementedError  # pragma: no cover

    # ── HTTP inference moved to providers/ (LlamaProvider / RemoteProvider) ──

    async def start(
        self,
        model_name: str,
        port: Optional[int] = None,
        backend: Optional[str] = None,
        **inference_kwargs: Any,
    ) -> None:
        """Start a model process.

        *port* and *backend* are infra keys (routing/lifecycle).
        All other keyword arguments are inference params — merged with
        config args and converted to ``--flag value`` CLI flags
        inside ``build_model_args``.
        """
        # ── Context lookup / creation ────────────────────────────────
        ctx = self._models.get(model_name)

        effective_backend: Optional[str] = backend
        eff_port = port  # default — overwritten by each path
        model_data = None

        # ── Already running: health-check shortcut ───────────────────
        if ctx is not None and ctx.state == RunnerState.RUNNING:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{ctx.port}/health", timeout=3
                    ) as resp:
                        if resp.status == 200:
                            self._inference_kwargs[model_name] = inference_kwargs
                            return
            except Exception:
                pass

        # ── Restart or first start: resolve port ─────────────────────
        if ctx and ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            new_size = inference_kwargs.get("max_log_lines", self.log_buffer_size)
            await self._before_restart(ctx, new_size)
            model_data = None  # reset — will be fetched below

        if port is not None:
            eff_port = port
        elif ctx and ctx.port is not None:
            eff_port = ctx.port
        else:
            eff_port = int(
                (self.cm.data.get("default") or {}).get('model-start-port', 18000)
            )

        if not isinstance(eff_port, int) or eff_port < 1 or eff_port > 65535:
            raise ValueError(f"Invalid port: {eff_port}")

        if model_data is None:
            model_data = self.cm.get_model(model_name, env_vars={"PORT": str(eff_port)})
            if not model_data:
                raise ModelNotStarted(model_name)
            effective_backend = backend or model_data.get("backend")
            effective_backend = _resolve_backend(self.cm, model_data, model_name, effective_backend)
            if not effective_backend:
                raise RuntimeError(f"No backend resolved for model '{model_name}'")

        # Update existing context — pre-creation set backend_id, runner_type, cache.
        ctx.port = eff_port
        ctx.backend_id = effective_backend
        ctx.set_state("load")
        if self.arkestra:
            self.arkestra.log(f"[start] model={model_name} port={eff_port} backend={effective_backend}")

        await self._ensure_port_available(eff_port)

        # ── Apply transient overrides from inference_kwargs ──────────
        for key in ('args', 'model'):
            if key in inference_kwargs and inference_kwargs[key] is not None:
                model_data[key] = inference_kwargs[key]
        if inference_kwargs.get('backend') is not None:
            effective_backend = inference_kwargs['backend']
            ctx.backend_id = effective_backend

        # Store inference kwargs for _start_model_process to use
        self._inference_kwargs[model_name] = inference_kwargs

        await self._start_model_process(ctx, model_data)

        watcher_task = asyncio.create_task(self._watch_process_or_container(model_name, ctx))
        self._watchers[model_name] = watcher_task

        start_time = time.monotonic()
        ready = False
        async with aiohttp.ClientSession() as session:
            while time.monotonic() - start_time < self.ready_timeout:
                try:
                    async with session.get(f"http://127.0.0.1:{eff_port}/health", timeout=5) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            status = data.get("status")
                            if status in ("ok", "loaded"):
                                ready = True
                                break
                            elif status in ("loading model", "loading"):
                                pass
                            elif status == "error":
                                raise RunnerError(f"Server reported error status: {data}")
                            else:
                                ready = True
                                break
                        elif resp.status == 503:
                            pass
                        elif resp.status in (502, 504):
                            raise RunnerError(f"Health check returned {resp.status}: upstream or gateway failure")
                        else:
                            raise RunnerError(f"Server health check failed with status {resp.status}")
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(self.ready_poll_ms / 1000.0)

        if not ready:
            await self.stop()
            raise ServerReadyTimeout(
                f"Model '{model_name}' failed to become ready on port {eff_port} "
                f"within {self.ready_timeout}s"
            )

        if self.arkestra:
            self.arkestra.log(f"[ready] model={model_name} port={eff_port}")
        await asyncio.sleep(self.warmup_delay)

        ctx.set_state("ready")

        # Start health watcher on first RUNNING model in this runner
        await self.start_health_watcher()

    async def stop(self) -> None:
        """Stop the models on this runner.

        Dedupes by context identity: several names may alias the same shared
        context (models of one checkpoint), so each is stopped exactly once.
        """
        seen = set()
        for key in list(self._models):
            ctx = self._models[key]
            if id(ctx) in seen:
                continue
            seen.add(id(ctx))
            await self._stop_single(key)

    async def _stop_single(self, model_name: str) -> None:
        ctx = self._models.get(model_name)
        if not ctx:
            return
        # Cancel any active pull task
        if ctx.download_task and not ctx.download_task.done():
            ctx.download_task.cancel()
        # Cancel any active log-capture task (containers only — no-op for processes)
        if hasattr(ctx, 'container_id') and ctx.container_id is not None:
            if hasattr(self, '_cancel_log_task'):
                self._cancel_log_task(ctx)
        ctx.set_state("stop")
        if self.arkestra:
            self.arkestra.log(f"[stop] model={ctx.name} port={ctx.port}")
        await self._stop_model_process(ctx)
        ctx.set_state("stopped")
        if self.arkestra:
            self.arkestra.log(f"[stop] model={ctx.name} DONE")

    async def stop_all(self) -> None:
        """Stop all active model processes — skip STOPPED/UNCACHED."""
        stopping = []
        for key, ctx in list(self._models.items()):
            if ctx.state in (RunnerState.RUNNING, RunnerState.LOADING, RunnerState.DOWNLOADING):
                stopping.append(key)
        if stopping:
            self.arkestra and self.arkestra.log(f"[action=shutdown] stopping: {', '.join(stopping)}")
        for key in stopping:
            await self._stop_single(key)

    async def shutdown(self) -> None:
        """Stop all models, cancel watchers, and clear the store — full teardown."""
        await self.stop_all()
        for task in self._watchers.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.shutdown_health_watcher()
        self._models.clear()
        self._watchers.clear()

    async def _watch_process_or_container(self, model_name: str, ctx: _Model) -> None:
        if ctx.process is not None:
            await self._watch_process(model_name, ctx)
        else:
            await self._watch_container(model_name, ctx)

    # ── Abstract lifecycle hooks ───────────────────────────

    @abstractmethod
    async def _start_model_process(
        self, ctx: _Model, model_data: Dict[str, Any]
    ) -> None:
        pass

    @abstractmethod
    async def _stop_model_process(self, ctx: _Model) -> None:
        pass
