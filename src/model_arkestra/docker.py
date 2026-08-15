from __future__ import annotations
import asyncio
import logging
import os
import re
import shlex
import subprocess
import time
logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional
from model_arkestra.container_runner import ContainerModelRunner
from model_arkestra.common import (
    build_model_args, resolve_binary_from_backend, safe_container_name, default_image_for_backend
)
from model_arkestra.types import _ModelContext


def _resolve_backend_for_docker(
    runner: "DockerModelRunner",
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


def _build_docker_cmd(
    runner: "DockerModelRunner",
    ctx: _ModelContext,
    model_data: Dict[str, Any],
) -> List[str]:
    """Build a complete docker run command."""
    # ── Determine effective image, devices, env from backend ───────────
    backend = _resolve_backend_for_docker(runner, ctx, model_data)

    if backend is not None:
        # --- New backends architecture ----------------------------------
        image = str(backend.get("image") or default_image_for_backend(ctx.backend_id))
        devices: List[str] = list(backend.get("devices", []))
        container_env: Dict[str, str] = dict(backend.get("env_container", {}))

        # Resolve host binary dir and optional devices from backend.
        binary_info = resolve_binary_from_backend(backend)
        binary_path: Optional[str] = None
        if binary_info is not None:
            binary_path, extra_devs = binary_info
            if not devices:
                devices = extra_devs

        # Merge global env vars (LLAMA_CACHE, HF_HUB_CACHE, etc.) into container env
        for k, v in (runner.cm.get_vector("env") or {}).items():
            container_env.setdefault(k, str(v))

        parts: List[str] = ["docker", "run", "-d"]

        # Device passthrough
        for dev in devices:
            parts.extend(["--device", str(dev)])

        # Extra env vars
        parts.extend(["-e", f"PORT={ctx.port}"])
        for k, v in container_env.items():
            parts.extend(["-e", f"{k}={v}"])

        # Container name
        parts.extend(["--name", safe_container_name(ctx.name, ctx.port)])

        # Port mapping — same port inside and out, bound to broadcast_addr.
        parts.extend(["-p", f"{runner.broadcast_addr}:{ctx.port}:{ctx.port}"])

        # HF cache mount (read-write)
        hf_cache = os.environ.get("HF_HUB_CACHE", "/home/lemonade/hub")
        if os.path.exists(hf_cache):
            parts.extend(["-v", f"{hf_cache}:/usr/local/hf_hub"])
            container_env.setdefault("HF_HUB_CACHE", "/usr/local/hf_hub")

        # Mount host binary dir so the resolved binary is reachable inside container.
        if binary_path:
            binary_dir = os.path.dirname(binary_path)
            if os.path.isdir(binary_dir):
                parts.extend(["-v", f"{binary_dir}:{binary_dir}:ro"])

        # Resolve the llama-server args list from backend registry.
        result = build_model_args(
            runner.cm,
            ctx.name,
            env_vars={"PORT": str(ctx.port)},
            override_backend=ctx.backend_id,
        )
        if result is not None:
            llama_arg_list = list(result[0])
        else:
            llama_arg_list = []

        # Ensure host binding (match the broadcast_addr).
        if "--host" not in llama_arg_list:
            llama_arg_list.append("--host")
            llama_arg_list.append(runner.broadcast_addr)

        parts.extend([image, "llama-server"])
        if binary_path:
            # Replace the image-bundled llama-server with the mounted host binary
            parts[-1] = binary_path
        # Append inference kwargs as CLI flags
        kwarg_flags = runner._build_cmd_line(
            runner._inference_kwargs.get(ctx.name, {})
        )
        parts.extend(llama_arg_list + kwarg_flags)

        return parts

    # ── Legacy fallback (no backends section) ──────────────────────────
    image = model_data.get("image") or "llama-strix-halo:vulkan"
    devices = list(model_data.get("devices", []))

    parts: List[str] = ["docker", "run", "-d"]

    # Device passthrough
    for dev in devices:
        parts.extend(["--device", str(dev)])

    parts.extend(["-e", f"PORT={ctx.port}"])

    parts.extend(["--name", safe_container_name(ctx.name, ctx.port)])

    # Same port inside and out, bound to broadcast_addr.
    parts.extend(["-p", f"{runner.broadcast_addr}:{ctx.port}:{ctx.port}"])

    hf_cache = os.environ.get("HF_HUB_CACHE", "/home/lemonade/hub")
    if os.path.exists(hf_cache):
        parts.extend(["-v", f"{hf_cache}:/usr/local/hf_hub"])
        parts.extend(["-e", "HF_HUB_CACHE=/usr/local/hf_hub"])

    # Legacy: model cmd already has resolved binary path.  Just pass it through.
    cmd_args = model_data.get("cmd", "") or ""
    if "--host" not in cmd_args:
        cmd_args += f" --host {runner.broadcast_addr}"

    parts.extend([image])
    parts.extend(shlex.split(cmd_args))

    return parts


class DockerModelRunner(ContainerModelRunner):

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
                        env=os.environ,
                    )
                    await proc.wait()
                except Exception:
                    pass

    async def _start_model_process(
        self, ctx: _ModelContext, model_data: Dict[str, Any]
    ) -> None:
        await self._ensure_port_available(ctx.port)
        # Remove stale container so docker run can reuse the name.
        cname = safe_container_name(ctx.name, ctx.port)
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", cname,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=os.environ,
            )
            await proc.wait()
        except Exception:
            pass
        cmd_parts = _build_docker_cmd(self, ctx, model_data)

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
                f"docker run failed for model '{ctx.name}': {err_msg}"
            )
        ctx.container_id = stdout.decode().strip()
