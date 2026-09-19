"""BaseRunner: lifecycle management for one checkpoint context.

Each runner instance manages exactly one _Model context (one set of weights).
The model_name parameter on methods is the *requested* model name — used for
config lookups (build_model_args, cm.get_model). The context itself is
self._ctx, set by Arkestra at wiring time.
"""
from __future__ import annotations
import asyncio
import logging
import time
import socket
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
import aiohttp

from model_arkestra.common import _resolve_backend
from model_arkestra.unicode_ringbuffer import UnicodeRingBuffer
from model_arkestra.types import (
    RunnerState, RunnerError, ServerReadyTimeout,
    ModelNotStarted, MaxRestartsExceeded, ModelShutdown, _Model
)

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    LOG_BUFFER_DEFAULT = 2000
    MODEL_START_TIMEOUT = 300
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
        cfg_br = (self.cm.data.get("runners") or {}).get("broadcast_addr")
        self.broadcast_addr = broadcast_addr if broadcast_addr is not None else (cfg_br or "0.0.0.0")
        cfg_lbs = self.cm.data.get('log-buffer-size')
        self.log_buffer_size = log_buffer_size if log_buffer_size is not None else (cfg_lbs or self.LOG_BUFFER_DEFAULT)

        # The single context this runner manages (set by Arkestra).
        self._ctx: Optional[_Model] = None
        self._watcher: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._inference_kwargs: Dict[str, Dict[str, Any]] = {}

    # ── port helpers ────────────────────────────────────────────
    async def _ensure_port_available(self, port: int) -> None:
        """Raise RunnerError if *port* is already in use."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: socket.create_connection(("127.0.0.1", port), timeout=1)
            )
            raise RunnerError(f"Port {port} is already in use")
        except ConnectionRefusedError:
            pass
        except OSError:
            pass

    async def _release_port(self, port: int) -> None:
        """Subclasses may override to wait for the listener to drain."""
        pass

    # ── context access ──────────────────────────────────────────
    @property
    def ctx(self) -> Optional[_Model]:
        return self._ctx

    @property
    def running_models(self) -> Set[str]:
        """Set of model names currently RUNNING (empty or {name})."""
        if self._ctx and self._ctx.state == RunnerState.RUNNING:
            return {self._ctx.name}
        return set()

    async def __aenter__(self) -> BaseRunner:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    # ── lifecycle: start ────────────────────────────────────────
    async def start(
        self,
        model_name: str,
        port: Optional[int] = None,
        backend: Optional[str] = None,
        **inference_kwargs: Any,
    ) -> None:
        """Start the model process for this runner's context.

        *model_name* is used for config resolution (build_model_args).
        *port* and *backend* are infra keys. Everything else is inference params.
        """
        ctx = self._ctx
        if ctx is None:
            raise ModelNotStarted(model_name)

        # ── Already running: health-check shortcut ───────────────────
        if ctx.state == RunnerState.RUNNING:
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

        # ── Resolve port ─────────────────────────────────────────────
        if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            new_size = inference_kwargs.get("max_log_lines", self.log_buffer_size)
            await self._before_restart(ctx, new_size)

        if port is not None:
            eff_port = port
        elif ctx.port is not None:
            eff_port = ctx.port
        else:
            eff_port = int(
                (self.cm.data.get("default") or {}).get('model-start-port', 18000)
            )

        if not isinstance(eff_port, int) or eff_port < 1 or eff_port > 65535:
            raise ValueError(f"Invalid port: {eff_port}")

        # ── Resolve config and backend ───────────────────────────────
        model_data = self.cm.get_model(model_name, env_vars={"PORT": str(eff_port)})
        if not model_data:
            raise ModelNotStarted(model_name)
        effective_backend = backend or model_data.get("backend")
        effective_backend = _resolve_backend(self.cm, model_data, model_name, effective_backend)
        if not effective_backend:
            raise RuntimeError(f"No backend resolved for model '{model_name}'")

        # ── Update context ───────────────────────────────────────────
        ctx.port = eff_port
        ctx.backend_id = effective_backend
        was_loading = ctx.state == RunnerState.LOADING
        ctx.set_state("load")
        if self.arkestra:
            self.arkestra.log(f"[start] model={model_name} port={eff_port} backend={effective_backend}")

        try:
            await self._ensure_port_available(eff_port)

            # ── Apply transient overrides ────────────────────────────────
            for key in ('args', 'model'):
                if key in inference_kwargs and inference_kwargs[key] is not None:
                    model_data[key] = inference_kwargs[key]
            if inference_kwargs.get('backend') is not None:
                effective_backend = inference_kwargs['backend']
                ctx.backend_id = effective_backend

            self._inference_kwargs[model_name] = inference_kwargs

            await self._start_model_process(ctx, model_data, model_name)
        except Exception as e:
            # Record so failed starts are visible in /admin/models and OWUI
            # (model_status shows error_message for ERROR states; UIs also
            # read last_error directly).
            ctx.last_error = str(e)
            if self.arkestra:
                self.arkestra.log(f"[start_fail] model={model_name} port={eff_port} error={e}", level="ERROR")
            # Roll back so a failed start never leaves the context stuck in LOADING.
            if not was_loading:
                ctx.set_state("start_fail")
            raise

        # ── Watch process ────────────────────────────────────────────
        if self._watcher is not None:
            self._watcher.cancel()
        self._watcher = asyncio.create_task(self._watch(model_name, ctx))

        # ── Wait for ready ───────────────────────────────────────────
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
        await self.start_health_watcher()

    # ── lifecycle: stop ─────────────────────────────────────────
    async def stop(self) -> None:
        """Stop the model process for this runner's context."""
        ctx = self._ctx
        if ctx is None:
            return
        if ctx.download_task and not ctx.download_task.done():
            ctx.download_task.cancel()
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

    async def stop_if_active(self) -> None:
        """Stop only if the context is in an active state."""
        ctx = self._ctx
        if ctx and ctx.state in (RunnerState.RUNNING, RunnerState.LOADING, RunnerState.DOWNLOADING):
            await self.stop()

    # ── lifecycle: shutdown ─────────────────────────────────────
    async def stop_all(self) -> None:
        """Stop if active — skip STOPPED/UNCACHED."""
        await self.stop_if_active()

    async def shutdown(self) -> None:
        """Full teardown — stop model, cancel watchers."""
        await self.stop_all()
        if self._watcher is not None:
            self._watcher.cancel()
            try:
                await self._watcher
            except asyncio.CancelledError:
                pass
            self._watcher = None
        await self.shutdown_health_watcher()

    # ── process watching ────────────────────────────────────────
    async def _watch(self, model_name: str, ctx: _Model) -> None:
        """Monitor the process/container and restart on unexpected exit."""
        if ctx.process is not None:
            await self._watch_process(model_name, ctx)
        else:
            await self._watch_container(model_name, ctx)

    async def _watch_process(self, model_name: str, ctx: _Model) -> None:
        """Background task to monitor process lifecycle and restart on unexpected exit."""
        if ctx.process is None:
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
            await self._start_model_process(ctx, model_data, model_name)
        except Exception as e:
            logger.error(f"Model {model_name}: restart failed — {e}")

    async def _before_restart(self, ctx: _Model, new_size: Optional[int] = None) -> bool:
        """Prepare context for restart: transition state and manage buffer."""
        if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            return False
        ctx.set_state("load")

        if new_size is None:
            new_size = getattr(ctx, '_log_buf_lines', self.log_buffer_size)
            if not isinstance(new_size, int) or new_size == 0:
                new_size = self.log_buffer_size

        if isinstance(new_size, int):
            ctx._log_ring = UnicodeRingBuffer(new_size * ctx.AVG_LINE_BYTES)
            ctx._log_buf_lines = new_size

        return True

    # ── health watcher ──────────────────────────────────────────
    async def start_health_watcher(self, interval: float = 1.0) -> None:
        if self._health_task is not None:
            return
        self._health_task = asyncio.create_task(
            self._health_watch_loop(interval)
        )

    async def _health_watch_loop(self, interval: float) -> None:
        while True:
            ctx = self._ctx
            if ctx and ctx.state == RunnerState.RUNNING and ctx.port:
                try:
                    async with aiohttp.ClientSession() as session:
                        resp = await session.get(
                            f"http://127.0.0.1:{ctx.port}/health", timeout=3
                        )
                        if resp.status == 200:
                            data = await resp.json()
                            status = data.get("status")
                            if status is not None:
                                value = status.get("value", status) if isinstance(status, dict) else status
                                if value == "loading":
                                    ctx.set_state("health_loading")
                                elif value == "error":
                                    ctx.set_state("health_error")
                except Exception:
                    pass
            await asyncio.sleep(interval)

    async def shutdown_health_watcher(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

    # ── dispatch guard ───────────────────────────────────────────
    async def _dispatch(self, model_name: str) -> None:
        """Raise if the context is not in a state that can serve inference."""
        ctx = self._ctx
        if not ctx:
            raise ModelNotStarted(model_name)
        if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            raise ModelShutdown(f"Model '{model_name}' was stopped.")
        if ctx.state == RunnerState.ERROR:
            raise MaxRestartsExceeded(
                f"Model '{model_name}' exceeded restart limit after {ctx.restart_count} attempts"
            )

    # ── logs ─────────────────────────────────────────────────────
    async def get_logs(self, ctx: _Model, lines: int = 100) -> List[str]:
        """Return the last N log lines from the context's ring buffer."""
        if not ctx:
            return []
        result, _oldest = ctx._get_lines_since(0, lines)
        return [t for _, t in result]

    # ── abstract hooks ──────────────────────────────────────────
    @abstractmethod
    async def _start_model_process(
        self, ctx: _Model, model_data: Dict[str, Any], model_name: str
    ) -> None:
        pass

    @abstractmethod
    async def _stop_model_process(self, ctx: _Model) -> None:
        pass

    async def _watch_container(self, model_name: str, ctx: _Model) -> None:
        raise NotImplementedError  # pragma: no cover
