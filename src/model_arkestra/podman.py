from __future__ import annotations
import asyncio
import os
import shlex
from typing import Any, Dict

from model_arkestra.container_runner import ContainerModelRunner, _resolve_backend, _build_container_cmd
from model_arkestra.common import SUBPROCESS_ENV, safe_container_name
from model_arkestra.types import _ModelContext


class PodmanModelRunner(ContainerModelRunner):
    INSIDE_PORT = 8080

    def _container_cmd(self) -> str:
        return "podman"

    async def _remove_containers(self, cids: list) -> None:
        for cid in cids:
            if cid:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "podman", "rm", "-f", cid,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                        env=SUBPROCESS_ENV,
                    )
                    await proc.wait()
                except Exception:
                    pass

    async def _start_model_process(
        self, ctx: _ModelContext, model_data: Dict[str, Any]
    ) -> None:
        await self._ensure_port_available(ctx.port)
        backend = _resolve_backend(self, ctx, model_data)
        if backend is None:
            raise RuntimeError(
                f"No backend resolved for podman model '{ctx.name}' — "
                "configure a backend with an 'image' key."
            )
        cmd_parts = _build_container_cmd(
            "podman", self, ctx.name, ctx.port,
            "0.0.0.0", PodmanModelRunner.INSIDE_PORT,
            backend,
            backend_id=ctx.backend_id,
        )
        # podman-only flags: --replace (replace existing container) + --group-add keep-groups
        cmd_parts.insert(2, "--replace")   # after "podman" "run"
        cmd_parts.insert(4, "--group-add")
        cmd_parts.insert(5, "keep-groups")

        proc = await asyncio.create_subprocess_shell(
            shlex.join(cmd_parts),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=SUBPROCESS_ENV,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_msg = stderr.decode().strip() or f"exit code {proc.returncode}"
            raise RuntimeError(
                f"podman run failed for model '{ctx.name}': {err_msg}"
            )
        ctx.container_id = stdout.decode().strip()

        # Start live log capture.
        log_task = asyncio.create_task(
            self._capture_container_logs(ctx.name, ctx.container_id)
        )
        if not hasattr(self, '_log_tasks'):
            self._log_tasks = {}
        self._log_tasks[ctx.name] = log_task
