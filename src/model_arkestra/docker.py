from __future__ import annotations
import asyncio
import shlex
from typing import Any, Dict

from model_arkestra.container_runner import ContainerModelRunner, _build_container_cmd
from model_arkestra.common import SUBPROCESS_ENV, safe_container_name
from model_arkestra.types import _ModelContext


class DockerModelRunner(ContainerModelRunner):
    INSIDE_PORT = 8080

    def _container_cmd(self) -> str:
        return "docker"

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
        cmd_parts = _build_container_cmd(
            "docker", self, ctx.name, ctx.port,
            self.broadcast_addr, DockerModelRunner.INSIDE_PORT,
            backend_config=model_data,
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

    async def _capture_container_logs(
        self, model_name: str, container_id: str
    ) -> None:
        """Stream docker logs into ctx._log_buffer ring buffer."""
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "-f", "--tail", "0", container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=SUBPROCESS_ENV,
        )

        async def _read_stream(stream):
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                ctx = self._models.get(model_name)
                if ctx and raw:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line:
                        ctx._log_buffer.append(line)

        tasks = []
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                tasks.append(asyncio.create_task(_read_stream(stream)))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            proc.kill()
            raise
