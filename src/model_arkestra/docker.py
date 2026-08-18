from __future__ import annotations
import asyncio
import os
import shlex
from typing import Any, Dict, List

from model_arkestra.container_runner import ContainerModelRunner, _resolve_backend
from model_arkestra.common import (
    SUBPROCESS_ENV, build_model_args, resolve_binary_from_backend, safe_container_name,
)
from model_arkestra.types import _ModelContext


def _resolve_backend_for_docker(
    runner: Any, ctx: _ModelContext, model_data: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Resolve the effective backend for docker (priority: ctx > model_data > config)."""
    return _resolve_backend(runner, ctx, model_data)


def _build_docker_cmd(
    runner: Any, ctx: _ModelContext, model_data: Dict[str, Any]
) -> List[str]:
    """Build a docker run command using the new backend-config architecture.

    Docker maps host-to-host (same port in and out). Falls back to legacy
    ``model_data["image"]`` when no backend resolves.
    Global env vars are merged via setdefault (global loses if key already in container_env).
    """
    backend = _resolve_backend_for_docker(runner, ctx, model_data)

    if backend is not None:
        return _build_docker_cmd_new(runner, ctx, backend)

    # Legacy fallback — image from model_data.
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


def _build_docker_cmd_new(
    runner: Any, ctx: _ModelContext, backend: Dict[str, Any]
) -> List[str]:
    """Build docker command for a resolved backend (new architecture)."""
    devices: List[str] = list(backend.get("devices", []))
    container_env: Dict[str, str] = dict(backend.get("env_container", {}))

    # Merge global env vars via setdefault (container wins)
    global_env = (runner.cm.get_vector("env") or {}) if hasattr(runner.cm, "get_vector") else {}
    for k, v in global_env.items():
        container_env.setdefault(k, str(v))

    binary_info = resolve_binary_from_backend(backend)
    binary_dir: str | None = None
    if binary_info is not None:
        binary_path, extra_devs = binary_info
        if not devices:
            devices = list(extra_devs)
        binary_dir = os.path.dirname(binary_path) or binary_dir

    image = str(backend.get("image", "ark-llama:vulkan-radv"))
    inside_port = ctx.port  # Docker: same port in and out

    parts: List[str] = ["docker", "run", "-d"]
    parts.extend(["-e", f"PORT={ctx.port}"])
    for k, v in container_env.items():
        parts.extend(["-e", f"{k}={v}"])

    ld = os.environ.get("LD_LIBRARY_PATH")
    if ld:
        parts.extend(["-e", f"LD_LIBRARY_PATH={ld}"])

    parts.extend(["--name", safe_container_name(ctx.name, ctx.port)])
    parts.extend(["-p", f"{runner.broadcast_addr}:{ctx.port}:{inside_port}"])

    for dev in devices:
        parts.extend(["--device", str(dev)])

    if binary_dir and os.path.isdir(binary_dir):
        parts.extend(["-v", f"{binary_dir}:{binary_dir}:ro"])

    # Resolve llama-server CLI args from config.
    result = build_model_args(
        runner.cm, ctx.name,
        env_vars={"PORT": str(ctx.port)},
        override_backend=backend.get("runner"),
        inference_kwargs=runner._inference_kwargs.get(ctx.name, {}),
    )
    arg_list: List[str] = list(result[0]) if result else []

    # Replace --port value with inside_port; ensure --host is present.
    fixed: List[str] = []
    i = 0
    while i < len(arg_list):
        if arg_list[i] == "--port" and i + 1 < len(arg_list):
            fixed.extend(["--port", str(inside_port)])
            i += 2
        else:
            fixed.append(arg_list[i])
            i += 1
    if not fixed or "--host" not in fixed:
        fixed.extend(["--host", runner.broadcast_addr])

    parts.extend([image] + fixed)
    return parts


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
        # Resolve the full backend dict (new architecture) or fall back to legacy.
        backend = _resolve_backend_for_docker(self, ctx, model_data)
        if backend is not None:
            cmd_parts = _build_container_cmd(
                "docker", self, ctx.name, ctx.port,
                self.broadcast_addr, ctx.port,  # docker: same port in and out
                backend,
            )
        else:
            cmd_parts = _build_docker_cmd(self, ctx, model_data)

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
