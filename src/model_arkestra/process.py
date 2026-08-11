from __future__ import annotations
import asyncio
import os
import signal
from typing import Any, Dict
from model_arkestra.base import BaseModelRunner
from model_arkestra.types import _ModelContext


class ProcessModelRunner(BaseModelRunner):

    async def _start_model_process(
        self, ctx: _ModelContext, model_data: Dict[str, Any]
    ) -> None:
        await self._ensure_port_available(ctx.port)
        result = self.cm.assemble_command(
            ctx.name,
            env_vars={"PORT": str(ctx.port)},
            override_backend=ctx.backend_id,
        )
        if result is None:
            raise RuntimeError(f"Model '{ctx.name}' has no backend configured")

        args_list, _cmd_str = result
        # args_list[0] is the wrapper path; rest are the resolved arguments.
        env = os.environ.copy()
        if env_vars := self.cm.get_vector("env"):
            env.update({k: str(v) for k, v in env_vars.items()})
        ctx.process = await asyncio.create_subprocess_exec(
            *args_list,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=env, preexec_fn=os.setsid
        )

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
