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
from model_arkestra.base import BaseModelRunner
from model_arkestra.types import _ModelContext

INSPECT_RE = re.compile(r"^(exited|dead|paused|removing)\s*$", re.IGNORECASE)


def _resolve_backend_for_podman(
    runner: "PodmanModelRunner",
    ctx: _ModelContext,
    model_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Look up the effective backend dict for this model launch.

    Resolution order:
    1. *ctx.backend_id* (set by runtime ``backend=`` on ``start()``)
    2. Model's ``backend`` YAML key
    3. Registry default from ``backends.default``

    Returns **None** when the backends architecture is not in use, in which
    case the old ``model_data["image"]`` / fallback behaviour takes over.
    """
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
    """Build a complete podman run command.

    Uses the backend registry when available (new architecture).  Falls back
    to legacy heuristics / model-data keys when no backends are defined.
    """
    # ── Determine effective image, devices, env from backend ───────────
    backend = _resolve_backend_for_podman(runner, ctx, model_data)

    if backend is not None:
        # --- New backends architecture ----------------------------------
        wrapper_path = backend.get("wrapper")
        image = str(backend.get("image", "llama-strix-halo:vulkan"))
        devices: List[str] = list(backend.get("devices", []))
        container_env: Dict[str, str] = dict(backend.get("env_container", {}))

        # Port inside the container matches the --port arg.
        # Resolve and mount the wrapper script into the container.
        if wrapper_path and os.path.isfile(wrapper_path):
            # Determine the image to use from the backend definition.
            host_dir = os.path.dirname(wrapper_path)
            parts: List[str] = []

            parts.extend(["podman", "run", "--rm", "-d"])

            # Device passthrough
            for dev in devices:
                parts.extend(["--device", str(dev)])

            # Extra env vars
            parts.extend(["-e", f"PORT={ctx.port}"])
            for k, v in container_env.items():
                parts.extend(["-e", f"{k}={v}"])

            # Container name
            safe_name = ctx.name.replace("_", "-").replace(".", "-")
            parts.extend(["--name", f"llm-{safe_name}-{ctx.port}"])

            # Port mapping — host port (ctx.port) to internal.
            parts.extend(["-p", f"{ctx.port}:{runner.INSIDE_PORT}"])

            # HF cache mount (read-write)
            hf_cache = os.environ.get("HF_HUB_CACHE", "/home/lemonade/hub")
            if os.path.exists(hf_cache):
                parts.extend(["-v", f"{hf_cache}:/home/lemonade/hub"])

            # Resolve the llama-server args list from backend registry.
            result = runner.cm.assemble_command(
                ctx.name,
                env_vars={"PORT": str(ctx.port)},
                override_backend=ctx.backend_id,
            )
            if result is not None:
                # First element is wrapper path — skip it for the container.
                llama_arg_list = result[0][1:] if len(result[0]) > 1 else []
            else:
                llama_arg_list = []

            # Replace host PORT with INSIDE_PORT inside container.
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
                fixed_args = ["--host", "0.0.0.0"]
            elif "--host" not in fixed_args:
                fixed_args.append("--host")
                fixed_args.append("0.0.0.0")

            parts.extend(["--entrypoint", "/bin/bash"])
            parts.extend([image])
            parts.extend(["-c", f"/usr/local/bin/llama-vulkan.sh {' '.join(fixed_args)}"])

            return parts

        # No wrapper script — build minimal podman run with image + env.
        parts = ["podman", "run", "--rm", "-d"]
        for dev in devices:
            parts.extend(["--device", str(dev)])
        parts.extend(["-e", f"PORT={ctx.port}"])
        for k, v in container_env.items():
            parts.extend(["-e", f"{k}={v}"])

        safe_name = ctx.name.replace("_", "-").replace(".", "-")
        parts.extend(["--name", f"llm-{safe_name}-{ctx.port}"])

        # Port mapping — host port (ctx.port) to internal.
        parts.extend(["-p", f"{ctx.port}:{runner.INSIDE_PORT}"])

        hf_cache = os.environ.get("HF_HUB_CACHE", "/home/lemonade/hub")
        if os.path.exists(hf_cache):
            parts.extend(["-v", f"{hf_cache}:/home/lemonade/hub"])

        # Use the container's built-in llama-vulkan.sh wrapper.
        result = runner.cm.assemble_command(
            ctx.name,
            env_vars={"PORT": str(ctx.port)},
            override_backend=ctx.backend_id,
        )
        if result is not None:
            llama_arg_list = result[0][1:] if len(result[0]) > 1 else []
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
            fixed_args = ["--host", "0.0.0.0"]
        elif "--host" not in fixed_args:
            fixed_args.extend(["--host", "0.0.0.0"])

        parts.extend(["--entrypoint", "/bin/bash"])
        parts.extend([image])
        parts.extend(["-c", f"/usr/local/bin/llama-vulkan.sh {' '.join(fixed_args)}"])
        return parts

    # ── Legacy fallback (no backends section) ──────────────────────────
    # Respect per-model overrides for image/device/env.
    image = model_data.get("image", "llama-strix-halo:vulkan")
    devices = list(model_data.get("devices", []))

    parts: List[str] = ["podman", "run", "--rm", "-d"]

    for dev in devices:
        parts.extend(["--device", str(dev)])

    parts.extend(["-e", f"PORT={ctx.port}"])

    safe_name = ctx.name.replace("_", "-").replace(".", "-")
    parts.extend(["--name", f"llm-{safe_name}-{ctx.port}"])

    # Determine effective inside port — model override, or class default.
    inside_port = model_data.get(
        "container_port", runner.INSIDE_PORT
    )
    parts.extend(["-p", f"{ctx.port}:{inside_port}"])

    hf_cache = os.environ.get("HF_HUB_CACHE", "/home/lemonade/hub")
    if os.path.exists(hf_cache):
        parts.extend(["-v", f"{hf_cache}:/home/lemonade/hub"])

    # Legacy: model cmd already has resolved binary path.  Just pass it through.
    cmd_args = model_data.get("cmd", "") or ""
    if "--host" not in cmd_args:
        cmd_args += " --host 0.0.0.0"

    parts.extend([image, "-c", cmd_args])
    return parts


class PodmanModelRunner(BaseModelRunner):
    INSIDE_PORT = 9090

    async def _release_port(self, port: int) -> None:
        """Wait up to ``port_drain_timeout`` seconds for a stopped container's listener to release the port."""
        deadline = time.monotonic() + self.port_drain_timeout
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["lsof", "-ti:", str(port)],
                capture_output=True, text=True,
            )
            if not result.stdout.strip():
                return
            await asyncio.sleep(0.2)

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

    async def _stop_model_process(self, ctx: _ModelContext) -> None:
        """Stop a Podman container gracefully, falling back to force-kill."""
        cid = getattr(ctx, "container_id", None)
        if not cid:
            return
        try:
            stop = await asyncio.create_subprocess_exec(
                "podman", "stop", "--time", str(self.port_drain_timeout), cid,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ,
            )
            await stop.wait()
        except Exception:
            pass
        ctx.container_id = None

    async def _watch_container(self, model_name: str, ctx: _ModelContext) -> None:
        """Poll Podman for container status and restart on unexpected exit."""
        cid = getattr(ctx, "container_id", None)
        if not cid:
            return

        while True:
            await asyncio.sleep(2.0)  # poll interval

            try:
                proc = await asyncio.create_subprocess_exec(
                    "podman", "inspect", cid, "--format", "{{.State.Status}}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=os.environ,
                )
                stdout, _ = await proc.communicate()

                # Container was removed (e.g. by --rm during stop) — nothing to do
                if proc.returncode != 0:
                    return

                status = stdout.decode().strip().lower()

                if INSPECT_RE.match(status):
                    await self._handle_restart(model_name, ctx, exit_code=1)
                    # After restart the old container is gone or replaced.
                    return

            except Exception as e:
                logger.warning(
                    f"Error inspecting podman container {cid}: {e}"
                )
                continue


    async def _before_restart(self, ctx: _ModelContext) -> bool:
        """Ensure stale container reference is cleared before restarting."""
        ctx.container_id = None
        return await super()._before_restart(ctx)

    async def shutdown(self) -> None:
        """Full teardown — stop models, force-remove containers, clear state."""
        cids = [getattr(ctx, "container_id", None) for ctx in list(self._models.values())]
        await super().shutdown()
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
