from __future__ import annotations
import asyncio
import os
import shlex
from typing import Any, Dict, List

from model_arkestra.container_runner import ContainerModelRunner, _resolve_backend
from model_arkestra.common import SUBPROCESS_ENV, build_model_args, resolve_binary_from_backend, safe_container_name
from model_arkestra.types import _ModelContext


def _resolve_backend_for_podman(
    runner: Any, ctx: _ModelContext, model_data: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Resolve the effective backend for podman (priority: ctx > model_data > config)."""
    return _resolve_backend(runner, ctx, model_data)


def _build_podman_cmd(
    runner: Any, ctx: _ModelContext, model_data: Dict[str, Any]
) -> List[str]:
    """Build a podman run command using the new backend-config architecture.

    Podman maps host_port → INSIDE_PORT (default 8080). Falls back to legacy
    ``model_data["image"]`` when no backend resolves.
    """
    inside_port = getattr(runner, "INSIDE_PORT", 8080)
    backend = _resolve_backend_for_podman(runner, ctx, model_data)

    if backend is not None:
        return _build_podman_cmd_new(runner, ctx, backend, inside_port)

    # Legacy fallback — image from model_data.
    image = str(model_data.get("image", "ark-llama:vulkan-radv"))
    container_port = int(model_data.get("container_port", inside_port))
    cmd_str = str(model_data.get("cmd", ""))
    arg_list = shlex.split(cmd_str) if cmd_str.strip() else []
    fixed: List[str] = []
    i = 0
    while i < len(arg_list):
        if arg_list[i] == "--port" and i + 1 < len(arg_list):
            fixed.extend(["--port", str(container_port)])
            i += 2
        else:
            fixed.append(arg_list[i])
            i += 1
    if not fixed or "--host" not in fixed:
        fixed.extend(["--host", runner.broadcast_addr])

    parts: List[str] = [
        "podman", "run", "-d",
        "-e", f"PORT={ctx.port}",
        "--name", safe_container_name(ctx.name, ctx.port),
        "-p", f"0.0.0.0:{ctx.port}:{container_port}",
    ] + fixed + [image]
    return parts


def _build_podman_cmd_new(
    runner: Any, ctx: _ModelContext, backend: Dict[str, Any], inside_port: int
) -> List[str]:
    """Build podman command for a resolved backend (new architecture)."""
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

    parts: List[str] = ["podman", "run", "-d"]
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
        # Resolve the full backend dict (new architecture) or fall back to legacy.
        backend = _resolve_backend_for_podman(self, ctx, model_data)
        if backend is not None:
            cmd_parts = _build_container_cmd(
                "podman", self, ctx.name, ctx.port,
                self.broadcast_addr, PodmanModelRunner.INSIDE_PORT,
                backend,
            )
        else:
            cmd_parts = _build_podman_cmd(self, ctx, model_data)
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
