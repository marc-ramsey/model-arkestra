from __future__ import annotations
import asyncio
import logging
import json
import time
import os
import socket
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional, Iterator, Set
import aiohttp
from model_arkestra.types import (
    RunnerState, RunnerError, ServerReadyTimeout, 
    ModelNotStarted, MaxRestartsExceeded, ModelShutdown, _ModelContext
)

logger = logging.getLogger(__name__)


class BaseModelRunner(ABC):
    def __init__(self, config_manager: Any, restart_delay: float = 5.0,
                 restart_limit: int = 4, shutdown_timeout: float = 20.0,
                 ready_timeout: float = 120.0, ready_poll_ms: float = 100.0,
                 warmup_delay: Optional[float] = None, port_drain_timeout: float = 20.0,
                 broadcast_addr: str = "0.0.0.0"):
        self.cm = config_manager
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
        self._watchers: Dict[str, asyncio.Task] = {}
        self._inference_kwargs: Dict[str, Dict[str, Any]] = {}
        self._models: Dict[str, _ModelContext] = {}



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

    async def __aenter__(self) -> BaseModelRunner:
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

    async def _watch_process(self, model_name: str, ctx: _ModelContext) -> None:
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

    async def _handle_restart(self, model_name: str, ctx: _ModelContext,
                              exit_code: int) -> None:
        if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            return

        ctx.restart_count += 1
        if ctx.restart_count >= self.restart_limit:
            logger.error(
                f"Model {model_name}: restart limit ({self.restart_limit}) "
                f"exceeded after {ctx.restart_count} attempts"
            )
            ctx.state = RunnerState.ERROR
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

    async def _before_restart(self, ctx: _ModelContext) -> bool:
        if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            return False
        ctx.state = RunnerState.LOADING
        return True

    async def _watch_container(self, model_name: str, ctx: _ModelContext) -> None:
        """Monitor a detached container's lifecycle and restart on exit."""
        raise NotImplementedError  # pragma: no cover

    # ── HTTP helpers ───────────────────────────────────────

    async def _stream_sse(self, model_name: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        await self._dispatch(model_name)
        ctx = next((v for k, v in self._models.items() if k == model_name), None)
        port = ctx.port
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        start_time = time.monotonic()
        tokens_so_far = []
        usage_info: Dict[str, Any] = {}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload, timeout=60) as resp:
                    if resp.status != 200:
                        raise RunnerError(f"Server error: {resp.status}")

                    while True:
                        raw = await resp.content.readline()
                        if not raw:
                            break
                        text = raw.decode("utf-8").strip()
                        if not text:
                            continue

                        event_data = text[5:].strip() if text.startswith("data:") else text

                        if event_data == "[DONE]":
                            elapsed = round(time.monotonic() - start_time, 2)
                            prompt_tok = usage_info.get("prompt_tokens", len(tokens_so_far))
                            completion_tok = usage_info.get("completion_tokens", 0)
                            if not completion_tok and tokens_so_far:
                                completion_tok = len(tokens_so_far)
                            usage_info.update({
                                "model": model_name,
                                "prompt_tokens": prompt_tok,
                                "completion_tokens": completion_tok,
                                "total_tokens": prompt_tok + completion_tok,
                                "time_seconds": elapsed,
                                "tokens_per_second": round(completion_tok / elapsed, 2) if elapsed > 0 else 0
                            })
                            yield {"usage": usage_info}
                            break

                        try:
                            chunk = json.loads(event_data)
                            choices = chunk.get("choices", [])
                            if choices and (delta := choices[0].get("delta")) and delta.get("content"):
                                tokens_so_far.append(delta["content"])
                                yield {"token": delta["content"]}
                            if "usage" in chunk:
                                usage_info.update(chunk["usage"])
                        except json.JSONDecodeError:
                            pass

            except Exception as e:
                raise RunnerError(f"Stream error: {e}")

    async def astream(self, model_name: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        payload = dict(payload)

        # Build messages array — accept full "messages" list or fall back to single prompt
        messages = None
        if "messages" in payload and isinstance(payload["messages"], (list, tuple)):
            messages = list(payload["messages"])
        else:
            prompt = payload.pop("prompt", None)
            if not prompt:
                raise ValueError("Payload must contain 'prompt' or 'messages'")
            messages = [{"role": "user", "content": prompt}]

        stream_payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": True,
        }
        stream_payload.update({k: v for k, v in payload.items() if v is not None})
        async for event in self._stream_sse(model_name, stream_payload):
            yield event

    async def _complete_async(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        await self._dispatch(model_name)
        ctx = next((v for k, v in self._models.items() if k == model_name), None)
        port = ctx.port
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        payload: Dict[str, Any] = {"model": model_name}

        # Support full messages list (for LangChain) or single prompt (legacy)
        if "messages" in kwargs and isinstance(kwargs["messages"], (list, tuple)):
            payload["messages"] = list(kwargs.pop("messages"))
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]

        payload.update({k: v for k, v in kwargs.items() if v is not None})

        async with aiohttp.ClientSession() as session:
            for attempt in range(12):
                try:
                    async with session.post(url, json=payload, timeout=60) as resp:
                        if resp.status == 503:
                            await asyncio.sleep(2.5)
                            continue
                        if resp.status in (502, 504):
                            raise RunnerError(f"Server returned {resp.status}: upstream or gateway failure")
                        if resp.status != 200:
                            raise RunnerError(f"Server error: {resp.status}")
                        data = await resp.json()
                        return {
                            "content": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                            "usage": data.get("usage", {
                                "model": model_name,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                                "time_seconds": 0
                            })
                        }
                except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                    if attempt == 11:
                        raise RunnerError(f"Server not reachable: {exc}")
                    await asyncio.sleep(2.5)
                except Exception as e:
                    raise RunnerError(f"Request failed: {e}")
            raise RunnerError("Maximum retries exceeded for completion request")

    async def ainvoke(self, model_name: str, prompt: str = "", **kwargs) -> str:
        res = await self._complete_async(model_name, prompt, **kwargs)
        return res.get("content", "")

    async def request(self, model_name: str, path: str, **kwargs) -> Any:
        await self._dispatch(model_name)
        ctx = next((v for k, v in self._models.items() if k == model_name), None)
        port = ctx.port
        url = f"http://127.0.0.1:{port}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.request("POST", url, json=kwargs, timeout=15) as resp:
                if resp.status < 400:
                    try:
                        return await resp.json()
                    except Exception:
                        return await resp.read()
                else:
                    raise RunnerError(f"Request failed with status {resp.status}")

    async def _build_cmd_line(self, args: Dict[str, Any]) -> List[str]:
        """Convert inference kwargs to CLI flags.

        Each key-value pair becomes two subprocess args:
          `--snake-case-key` `value`

        Keys are snake_case → kebab-case. Values are stringified.
        This method is appended to the base CLI from ``build_model_args``.
        """
        cli: List[str] = []
        for key, value in args.items():
            flag = f"--{key.replace('_', '-')}"  # snake_case → kebab-case
            if isinstance(value, bool):
                cli.extend([flag, str(value).lower()])
            else:
                cli.extend([flag, str(value)])
        return cli

    async def start(
        self,
        model_name: str,
        port: Optional[int] = None,
        backend: Optional[str] = None,
        **inference_kwargs: Any,
    ) -> None:
        """Start a model process.

        *port* and *backend* are infra keys (routing/lifecycle).
        All other keyword arguments are inference params — converted to
        ``--flag value`` CLI flags appended after the base args from
        ``build_model_args``.
        """
        # ── Context lookup / creation ────────────────────────────────
        ctx = self._models.get(model_name)
        if not ctx:
            ctx = next(
                (v for k, v in self._models.items() if k == model_name), None
            )

        effective_backend: Optional[str] = backend

        # ── Restart path: reuse existing port ────────────────────────
        if ctx and ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            await self._before_restart(ctx)
            eff_port = port if port is not None else ctx.port
            new_size = inference_kwargs.get("max_log_lines")
            if new_size is None:
                new_size = self.cm.data.get('log-buffer-size', 500)
            if len(ctx._log_buffer) > 0 and ctx._log_buffer.maxlen != new_size:
                from collections import deque
                ctx._log_buffer = deque(maxlen=new_size)

        # ── Already running: health-check shortcut ───────────────────
        elif ctx is not None and ctx.state == RunnerState.RUNNING:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{ctx.port}/health", timeout=3
                    ) as resp:
                        if resp.status == 200:
                            # Already healthy — store kwargs for potential restart
                            self._inference_kwargs[model_name] = inference_kwargs
                            return
            except Exception:
                pass

        # ── New model: allocate port & create context ────────────────
        else:
            eff_port = port if port is not None else int(
                self.cm.data.get('models-start-port', 18000)
            )
            if not isinstance(eff_port, int) or eff_port < 1 or eff_port > 65535:
                raise ValueError(f"Invalid port: {eff_port}")

            model_data = self.cm.get_model(model_name, env_vars={"PORT": str(eff_port)})
            if not model_data:
                raise ModelNotStarted(model_name)

            effective_backend = backend or model_data.get("backend")
            log_size = inference_kwargs.get("max_log_lines")
            if log_size is None:
                log_size = self.cm.data.get('log-buffer-size', 500)

            ctx = _ModelContext(model_name, eff_port, max_log_lines=log_size)
            ctx.backend_id = effective_backend
            ctx.broadcast_addr = self.broadcast_addr
            self._models[model_name] = ctx
            ctx.state = RunnerState.LOADING

        await self._ensure_port_available(eff_port)

        if not model_data:
            model_data = self.cm.get_model(model_name, env_vars={"PORT": str(eff_port)})

        # ── Apply transient overrides from inference_kwargs ──────────
        for key in ('args', 'checkpoint'):
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

        await asyncio.sleep(self.warmup_delay)

        ctx.state = RunnerState.RUNNING

    async def stop(self) -> None:
        """Stop the single model on this runner."""
        for key in list(self._models):
            await self._stop_single(key)

    async def _stop_single(self, model_name: str) -> None:
        ctx = self._models.get(model_name)
        if not ctx:
            return
        ctx.state = RunnerState.STOPPING
        await self._stop_model_process(ctx)
        # Release port but keep entry for restart-on-start semantics
        if hasattr(ctx, 'port'):
            await self._release_port(ctx.port)
        ctx.state = RunnerState.STOPPED

    async def stop_all(self) -> None:
        """Stop all model processes, leaving entries in STOPPED state for restart-on-start."""
        for key in list(self._models):
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
        self._models.clear()
        self._watchers.clear()

    async def _watch_process_or_container(self, model_name: str, ctx: _ModelContext) -> None:
        if ctx.process is not None:
            await self._watch_process(model_name, ctx)
        else:
            await self._watch_container(model_name, ctx)

    # ── Abstract lifecycle hooks ───────────────────────────

    @abstractmethod
    async def _start_model_process(
        self, ctx: _ModelContext, model_data: Dict[str, Any]
    ) -> None:
        pass

    @abstractmethod
    async def _stop_model_process(self, ctx: _ModelContext) -> None:
        pass
