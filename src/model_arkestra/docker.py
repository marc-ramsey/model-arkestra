from __future__ import annotations
import asyncio
import os
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


def _build_docker_legacy(
    runner: Any, ctx: _ModelContext, model_data: Dict[str, Any]
) -> List[str]:
    """Legacy docker run command using model_data["image"] directly (no backend config)."""
    image = str(model_data.get("image", "ark-llama:vulkan-radv"))
    cmd_str = str(model_data.get("cmd", ""))
    arg_list = shlex.split(cmd_str) if cmd_str.strip() else []
    fixed: List[str] = []
    i = 0
    while i < len(arg_list):
        if arg_list[i] == "--port" and i + 1 < len(arg_list):
            fixed.extend(["--port", str(ctx.port)])
            i += 2
        else:
            fixed.append(arg_list[i])
            i += 1
    if not fixed or "--host" not in fixed:
        fixed.extend(["--host", runner.broadcast_addr])

    parts: List[str] = [
        "docker", "run", "-d",
        "-e", f"PORT={ctx.port}",
        "--name", safe_container_name(ctx.name, ctx.port),
        "-p", f"0.0.0.0:{ctx.port}:{ctx.port}",
    ] + fixed + [image]
    return parts


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
        # Resolve the full backend dict (new architecture) or fall back to legacy.
        be = _resolve_backend_for_docker(self, ctx, model_data)
        if be is not None:
            cmd_parts = _build_container_cmd(
                "docker", self, ctx.name, ctx.port,
                self.broadcast_addr, ctx.port,  # docker: same port in and out
                be,
                backend_id=ctx.backend_id,
            )
        else:
            cmd_parts = _build_docker_legacy(self, ctx, model_data)

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
