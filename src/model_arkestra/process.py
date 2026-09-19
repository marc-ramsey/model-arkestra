"""ProcessRunner: spawn and manage llama-server as a local subprocess."""
from __future__ import annotations
import asyncio
import os
import signal
from typing import Any, Dict, List

from model_arkestra.base import BaseRunner
from model_arkestra.common import build_model_args
from model_arkestra.llama_cpp import LlamaCppEngine
from model_arkestra.types import _Model


class ProcessRunner(BaseRunner):

    async def get_logs(self, ctx: _Model, lines: int = 100) -> List[str]:
        """Return the last N log lines from the context's ring buffer."""
        if not ctx:
            return []
        result, _oldest = ctx._get_lines_since(0, lines)
        return [t for _, t in result]

    async def _start_model_process(
        self, ctx: _Model, model_data: Dict[str, Any], model_name: str
    ) -> None:
        await self._ensure_port_available(ctx.port)

        # Resolve backend and locate binary. Unknown backend → empty dict,
        # which fails fast with a clear "Binary not found" error below.
        be_id = ctx.backend_id or model_data.get("backend")
        backend = {}
        if be_id:
            if self.arkestra is not None:
                backend = self.arkestra.get_backend(be_id) or {}
            else:
                backend = self.cm.get(f"backends/{be_id}") or {}
        binary_dir = (backend or {}).get("binary_dir", "") or ""
        binary_name = (backend or {}).get("binary", "llama-server") or "llama-server"
        binary_path = os.path.join(binary_dir, binary_name)
        if not os.path.isfile(binary_path):
            raise RuntimeError(
                f"Binary '{binary_path}' not found for backend '{be_id}'"
            )

        # Build merged args using the model name (config key), not ctx.name.
        merged = build_model_args(self.cm, model_name,
                                  inference_kwargs=self._inference_kwargs.get(model_name, {}))
        if merged is None:
            raise RuntimeError(f"Model '{model_name}' has no backend configured")

        engine_name = (backend or {}).get("engine", "llama-cpp")
        if engine_name == "llama-cpp":
            args_list = LlamaCppEngine.build_cli_args(merged, ctx.port)
        else:
            args_list = list(merged.values())

        # Merge environment.
        env = os.environ.copy()
        for k, v in (self.cm.data.get("env") or {}).items():
            env[k] = str(v)
        if self.arkestra:
            for k, v in self.arkestra.device_profile.items():
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

        if self.arkestra:
            self.arkestra.log(f"[launch] model={model_name} pid={ctx.process.pid} binary={binary_path} port={ctx.port}")

        # Start log capture.
        async def _read_stream(stream: asyncio.Stream) -> None:
            while True:
                try:
                    raw = await stream.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line:
                        ctx._append_log_line(line)
                except asyncio.CancelledError:
                    return
                except Exception:
                    break

        self._log_tasks = []
        if ctx.process.stdout:
            self._log_tasks.append(asyncio.create_task(_read_stream(ctx.process.stdout)))
        if ctx.process.stderr:
            self._log_tasks.append(asyncio.create_task(_read_stream(ctx.process.stderr)))

    async def _stop_model_process(self, ctx: _Model) -> None:
        """Kill model process group: SIGHUP → wait 20s → SIGKILL."""
        # Cancel log capture tasks.
        for t in getattr(self, '_log_tasks', []):
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._log_tasks = []

        if ctx.process and ctx.process.returncode is None:
            pid = ctx.process.pid
            if self.arkestra:
                self.arkestra.log(f"[stop] model={ctx.name} pid={pid} SIGHUP")
            try:
                os.killpg(ctx.process.pid, signal.SIGHUP)
                await asyncio.wait_for(ctx.process.wait(), timeout=20.0)
                return
            except (asyncio.TimeoutError, ProcessLookupError):
                pass
            if self.arkestra:
                self.arkestra.log(f"[kill] model={ctx.name} pid={pid} SIGKILL", level="WARNING")
            try:
                os.killpg(ctx.process.pid, signal.SIGKILL)
                await ctx.process.wait()
            except Exception:
                pass

    async def _before_restart(self, ctx: _Model, new_size=None) -> bool:
        """Reset process reference so the next start creates a fresh one."""
        if ctx.process is not None and ctx.process.returncode is not None:
            ctx.process = None
        return await super()._before_restart(ctx, new_size)
