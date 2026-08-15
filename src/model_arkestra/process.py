from __future__ import annotations
import asyncio
import os
import signal
from typing import Any, Dict, List
from model_arkestra.base import BaseModelRunner
from model_arkestra.common import build_model_args
from model_arkestra.types import _ModelContext


class ProcessModelRunner(BaseModelRunner):

    async def get_logs(self, model_name: str, lines: int = 100) -> List[str]:
        """Return the last N log lines for a model."""
        ctx = self._models.get(model_name)
        if not ctx:
            return []
        buffer = list(ctx._log_buffer)
        return buffer[-lines:] if len(buffer) >= lines else buffer

    async def _start_model_process(
        self, ctx: _ModelContext, model_data: Dict[str, Any]
    ) -> None:
        await self._ensure_port_available(ctx.port)

        # Resolve backend for binary_dir, binary name, and env vars.
        be_id = ctx.backend_id or model_data.get("backend")
        if not be_id:
            be_id = self.cm.data.get("backends", {}).get("default")
        backend = self.cm.get_backend(be_id) if be_id else {}

        binary_dir = backend.get("binary_dir", "")
        binary_name = backend.get("binary", "llama-server")
        binary_path = os.path.join(binary_dir, binary_name)
        if not os.path.isfile(binary_path):
            raise RuntimeError(
                f"Binary '{binary_path}' not found for backend '{be_id}'"
            )

        # Build args from config backend + model sections.
        result = build_model_args(
            self.cm, ctx.name,
            env_vars={"PORT": str(ctx.port)},
            override_backend=ctx.backend_id,
        )
        if result is None:
            raise RuntimeError(f"Model '{ctx.name}' has no backend configured")

        args_list, _cmd_str = result

        # Append inference kwargs as CLI flags (--snake_case value)
        kwarg_flags = self._build_cmd_line(
            self._inference_kwargs.get(ctx.name, {})
        )
        args_list.extend(kwarg_flags)

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

        # Start log capture: feed stdout/stderr lines into ctx._log_buffer ring buffer
        async def _capture_logs(name: str, process: asyncio.subprocess.Process) -> None:
            """Read stdout/stderr and append each line to the model's log buffer."""
            for stream in (process.stdout, process.stderr):
                if stream is None:
                    continue
                while True:
                    try:
                        raw = await stream.readline()
                        if not raw:
                            break
                        ctx = self._models.get(name)
                        if ctx and len(raw) > 0:
                            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                            if line:
                                ctx._log_buffer.append(line)
                    except asyncio.CancelledError:
                        return
                    except Exception:
                        break

        log_task = asyncio.create_task(_capture_logs(ctx.name, ctx.process))
        if not hasattr(self, '_log_tasks'):
            self._log_tasks = {}
        self._log_tasks[ctx.name] = log_task

    async def _stop_model_process(self, ctx: _ModelContext) -> None:
        """Kill model process group using mandated strategy: SIGHUP → wait 20s → SIGKILL."""
        if ctx.process and ctx.process.returncode is None:
            pid = ctx.process.pid
            if pid:
                try:
                    os.killpg(ctx.process.pid, signal.SIGHUP)
                    await asyncio.wait_for(ctx.process.wait(), timeout=20.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        os.killpg(ctx.process.pid, signal.SIGKILL)
                        await ctx.process.wait()
                    except Exception:
                        pass

    async def _before_restart(self, ctx: _ModelContext) -> bool:
        """Reset process reference so the next ``_start_model_process`` call creates a fresh one."""
        if ctx.process is not None and ctx.process.returncode is not None:
            ctx.process = None  # replace stale process handle
        return await super()._before_restart(ctx)
