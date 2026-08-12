from __future__ import annotations
import asyncio
import logging
import os
import shlex
import subprocess
import time
logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional
from model_arkestra.container_runner import ContainerModelRunner
from model_arkestra.common import (
    resolve_binary_from_backend, safe_container_name, default_image_for_backend
)
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
                        env=os.environ,
                    )
                    await proc.wait()
                except Exception:
                    pass

    async def _start_model_process(
        self, ctx: _ModelContext, model_data: Dict[str, Any]
    ) -> None:
        await self._ensure_port_available(ctx.port)
        cmd_parts = _build_podman_cmd(self, ctx, model_data)

        proc = await asyncio.create_subprocess_shell(
            shlex.join(cmd_parts),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_msg = stderr.decode().strip() or f"exit code {proc.returncode}"
            raise RuntimeError(
                f"podman run failed for model '{ctx.name}': {err_msg}"
            )
        ctx.container_id = stdout.decode().strip()


def _resolve_backend_for_podman(
    runner: "PodmanModelRunner",
    ctx: _ModelContext,
    model_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Look up the effective backend dict for this model launch."""
    backend_id = ctx.backend_id or model_data.get("backend")
    if not backend_id:
        backends = runner.cm.data.get("backends")
        if isinstance(backends, dict):
            backend_id = backends.get("default")
        if not backend_id:
            return None

    return runner.cm.get_backend(backend_id)


def _build_podman_cmd(
    runner: "PodmanModelRunner",
    ctx: _ModelContext,
    model_data: Dict[str, Any],
) -> List[str]:
    """Build a complete podman run command."""
    # ── Determine effective image, devices, env from backend ───────────
    backend = _resolve_backend_for_podman(runner, ctx, model_data)

    if backend is not None:
        # --- New backends architecture ----------------------------------
        image = str(backend.get("image", "llama-vulkan-driver:vulkan"))
        devices: List[str] = list(backend.get("devices", []))
        container_env: Dict[str, str] = dict(backend.get("env_container", {}))

        # Resolve host binary dir and optional devices from backend.
        binary_info = resolve_binary_from_backend(backend)
        binary_dir: Optional[str] = None
        binary_path: Optional[str] = None
        if binary_info is not None:
            binary_path, extra_devs = binary_info
            # Merge extra devs only when user didn't set explicit devices
            if not devices:
                devices = extra_devs
            binary_dir = os.path.dirname(binary_path)

        # Port inside the container matches the --port arg.
        parts: List[str] = ["podman", "run", "--replace", "-d"]
        parts.extend(["-e", f"PORT={ctx.port}"])
        for k, v in container_env.items():
            parts.extend(["-e", f"{k}={v}"])

        ld = os.environ.get("LD_LIBRARY_PATH")
        if ld:
            parts.extend(["-e", f"LD_LIBRARY_PATH={ld}"])

        parts.extend(["--name", safe_container_name(ctx.name, ctx.port)])
        parts.extend(["-p", f"{runner.broadcast_addr}:{ctx.port}:{runner.INSIDE_PORT}"])

        hf_cache = os.environ.get("HF_HUB_CACHE", "/home/lemonade/hub")

        for dev in devices:
            parts.extend(["--device", str(dev)])

        # Mount host binary dir when resolved.
        if binary_dir and os.path.isdir(binary_dir):
            parts.extend(["-v", f"{binary_dir}:{binary_dir}:ro"])

        # Use llama-launch.sh with the host binary as first arg.
        result = runner.cm.assemble_command(
            ctx.name,
            env_vars={"PORT": str(ctx.port)},
            override_backend=ctx.backend_id,
        )
        if result is not None:
            llama_arg_list = list(result[0]) if result is not None else []
        else:
            llama_arg_list = []

        fixed_args: List[str] = []
        i = 0
        while i < len(llama_arg_list):
            if llama_arg_list[i] == "--port" and i + 1 < len(llama_arg_list):
                fixed_args.extend(["--port", str(runner.INSIDE_PORT)])
                i += 2
            else:
                fixed_args.append(llama_arg_list[i])
                i += 1
        if not fixed_args:
            fixed_args = ["--host", runner.broadcast_addr]
        elif "--host" not in fixed_args:
            fixed_args.extend(["--host", runner.broadcast_addr])

        # launcher.sh: first arg = binary, rest = --model etc.
        if binary_path:
            parts.extend([image, binary_path] + fixed_args)
        else:
            parts.extend([image])
            parts.extend(["-c", f"/usr/local/bin/llama-launch.sh {' '.join(fixed_args)}"])
        return parts

    # ── Legacy fallback (no backends section) ──────────────────────────
    image = model_data.get("image", "llama-vulkan-driver:vulkan")
    devices = list(model_data.get("devices", []))

    parts: List[str] = ["podman", "run", "--replace", "-d"]

    for dev in devices:
        parts.extend(["--device", str(dev)])

    parts.extend(["-e", f"PORT={ctx.port}"])

    ld = os.environ.get("LD_LIBRARY_PATH")
    if ld:
        parts.extend(["-e", f"LD_LIBRARY_PATH={ld}"])

    parts.extend(["--name", safe_container_name(ctx.name, ctx.port)])

    inside_port = model_data.get(
        "container_port", PodmanModelRunner.INSIDE_PORT
    )
    parts.extend(["-p", f"{runner.broadcast_addr}:{ctx.port}:{inside_port}"])

    hf_cache = os.environ.get("HF_HUB_CACHE", "/home/lemonade/hub")

    cmd_args = model_data.get("cmd", "") or ""
    if "--host" not in cmd_args:
        cmd_args += f" --host {runner.broadcast_addr}"

    parts.extend([image, "-c", cmd_args])
    return parts
