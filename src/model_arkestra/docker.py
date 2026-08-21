from __future__ import annotations
import asyncio
import shlex
from typing import Any, Dict, List

from model_arkestra.container_runner import ContainerModelRunner, _resolve_backend, _build_container_cmd
from model_arkestra.common import SUBPROCESS_ENV, safe_container_name
from model_arkestra.types import _ModelContext


def _resolve_backend_for_docker(
    runner: Any, ctx: _ModelContext, model_data: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Resolve the effective backend for docker (priority: ctx > model_data > config)."""
    return _resolve_backend(runner, ctx, model_data)


class DockerModelRunner(ContainerModelRunner):
    INSIDE_PORT = 8080

    def _container_cmd(self) -> str:
        return "docker"

    def _resolve_image(self, image: str) -> str:
        if "/" not in image:
            image = f"localhost/{image}"
        return image

    async def _remove_containers(self, cids: list) -> None:
        for cid in cids:
            if cid:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "docker", "rm", "-f", cid,
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
        cname = safe_container_name(ctx.name, ctx.port)
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", cname,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=SUBPROCESS_ENV,
            )
            await proc.wait()
        except Exception:
            pass

        be = _resolve_backend_for_docker(self, ctx, model_data)
        if be is None:
            raise RuntimeError(
                f"No backend resolved for docker model '{ctx.name}' — "
                "configure a backend with an 'image' key."
            )
        cmd_parts = _build_container_cmd(
            "docker", self, ctx.name, ctx.port,
            self.broadcast_addr, DockerModelRunner.INSIDE_PORT,
            be,
            backend_id=ctx.backend_id,
        )

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
                f"docker run failed for model '{ctx.name}': {err_msg}"
            )
        ctx.container_id = stdout.decode().strip()

        # Start live log capture.
        log_task = asyncio.create_task(
            self._capture_container_logs(ctx.name, ctx.container_id)
        )
        if not hasattr(self, '_log_tasks'):
            self._log_tasks = {}
        self._log_tasks[ctx.name] = log_task
