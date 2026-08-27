from __future__ import annotations
import asyncio
import os
import signal
from typing import Any, Dict, List
from model_arkestra.base import BaseModelRunner
from model_arkestra.common import _merge_engine_defaults, _resolve_backend, _resolve_engine, build_model_args
from model_arkestra.llama_cpp import LlamaCppEngine
from model_arkestra.types import _ModelContext



class ProcessModelRunner(BaseModelRunner):

    async def get_logs(self, model_name: str, lines: int = 100) -> List[str]:
        """Return the last N log line texts for a model (backward compat)."""
        ctx = self._models.get(model_name)
        if not ctx:
            return []
        result, _oldest = ctx._get_lines_since(0, lines)
        return [t for _, t in result]

    async def _start_model_process(
        self, ctx: _ModelContext, model_data: Dict[str, Any]
    ) -> None:
        await self._ensure_port_available(ctx.port)

        # Resolve backend and merge engine defaults.
        be_id = ctx.backend_id or model_data.get("backend")
        be_id = _resolve_backend(self.cm, model_data, ctx.name, be_id)
        backend = self.cm.get_backend(be_id) if be_id else {}

        # Resolve engine and merge its defaults into the backend config.
        engine_name = (backend or {}).get("engine", "llama_cpp")
        engine_cfg = _resolve_engine(self.cm, engine_name)
        merged = _merge_engine_defaults(engine_cfg, backend or {})

        binary_dir = merged.get("binary_dir", "") or ""
        binary_name = merged.get("binary", "llama-server") or "llama-server"
        binary_path = os.path.join(binary_dir, binary_name)
        if not os.path.isfile(binary_path):
            raise RuntimeError(
                f"Binary '{binary_path}' not found for backend '{be_id}'"
            )

        # Filter inference kwargs through engine — drop anything not valid for llama.cpp.
        raw_inference = self._inference_kwargs.get(ctx.name, {})
        if engine_name == "llama_cpp":
            from model_arkestra.llama_cpp import LlamaCppEngine
            engine = LlamaCppEngine()
        else:
            engine = None  # future: resolve by name
        filtered_inference = engine.filter_infer_kwargs(raw_inference) if engine else raw_inference

        # Build args from config backend + model + inference kwargs.
        result = build_model_args(
            self.cm, ctx.name,
            env_vars={"PORT": str(ctx.port)},
            override_backend=ctx.backend_id,
            inference_kwargs=filtered_inference,
        )
        if result is None:
            raise RuntimeError(f"Model '{ctx.name}' has no backend configured")

        args_list, _cmd_str = result

        # Merge environment: process + global env + backend env_container.
        env = os.environ.copy()
        for k, v in (self.cm.get_vector("env") or {}).items():
            env[k] = str(v)
        for k, v in (backend.get("env_container") or {}).items():
            env[k] = str(v)

        ctx.process = await asyncio.create_subprocess_exec(
            binary_path, *args_list,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid
        )

        print(f"[PROCESS] Model {ctx.name} launched: pid={ctx.process.pid} binary={binary_path} port={ctx.port}", flush=True)

        # Start log capture: feed stdout/stderr lines into ctx ring buffer
        async def _read_stream(stream: asyncio.Stream, model_name: str) -> None:
            """Read one stream and append each line to the model's log buffer."""
            while True:
                try:
                    raw = await stream.readline()
                    if not raw:
                        break
                    ctx = self._models.get(model_name)
                    if ctx and len(raw) > 0:
                        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                        if line:
                            ctx._append_log_line(line)
                except asyncio.CancelledError:
                    return
                except Exception:
                    break

        log_task_stdout = asyncio.create_task(
            _read_stream(ctx.process.stdout, ctx.name)
        ) if ctx.process.stdout else None
        log_task_stderr = asyncio.create_task(
            _read_stream(ctx.process.stderr, ctx.name)
        ) if ctx.process.stderr else None
        if not hasattr(self, '_log_tasks'):
            self._log_tasks = {}
        self._log_tasks[ctx.name] = (log_task_stdout, log_task_stderr)

    async def _stop_model_process(self, ctx: _ModelContext) -> None:
        """Kill model process group using mandated strategy: SIGHUP → wait 20s → SIGKILL."""
        if ctx.process and ctx.process.returncode is None:
            pid = ctx.process.pid
            print(f"[PROCESS] Stopping model {ctx.name} (pid={pid}): sending SIGHUP", flush=True)
            try:
                os.killpg(ctx.process.pid, signal.SIGHUP)
                await asyncio.wait_for(ctx.process.wait(), timeout=20.0)
                return
            except (asyncio.TimeoutError, ProcessLookupError):
                pass
            print(f"[PROCESS] Model {ctx.name} did not exit after SIGHUP — sending SIGKILL", flush=True)
            try:
                os.killpg(ctx.process.pid, signal.SIGKILL)
                await ctx.process.wait()
            except Exception:
                pass

    async def _before_restart(self, ctx: _ModelContext, new_size=None) -> bool:
        """Reset process reference so the next ``_start_model_process`` call creates a fresh one."""
        if ctx.process is not None and ctx.process.returncode is not None:
            ctx.process = None  # replace stale process handle
        return await super()._before_restart(ctx, new_size)
