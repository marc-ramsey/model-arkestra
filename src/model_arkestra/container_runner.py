"""Intermediate base class for container-based runners (Podman / Docker).

Contains shared logic for container lifecycle, health watching, restart behavior,
port release, and command building — eliminating near-duplicate code between
docker.py and podman.py.
"""
from __future__ import annotations
import asyncio
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from model_arkestra.base import BaseModelRunner
from model_arkestra.common import INSPECT_RE, SUBPROCESS_ENV, build_model_args, resolve_binary_from_backend, safe_container_name
from model_arkestra.types import RunnerState, _ModelContext


def _default_image(runner: Any) -> str:
    """Resolve default image from runner config."""
    data = runner.cm.data if hasattr(runner, "cm") and hasattr(runner.cm, "data") else {}
    images = data.get("images") or {}
    default = data.get("default-image")
    for candidate in ["vulkan-radv", "rocm"]:
        if candidate in images:
            return images[candidate].get("image", default or "ark-llama:vulkan-radv")
    return default or "ark-llama:vulkan-radv"


def _resolve_backend(runner: Any, model_data: Dict[str, Any]) -> Any | None:
    """Look up the effective backend dict for this model launch."""
    backend_id = (getattr(runner, "_current_context", None)
                  and getattr(runner._current_context, "backend_id", None)) \
                 or model_data.get("backend")
    if not backend_id:
        backends = runner.cm.data.get("backends")
        if isinstance(backends, dict):
            backend_id = backends.get("default")
        if not backend_id:
            return None
    return runner.cm.get_backend(backend_id)


def _build_container_cmd(
    cmd: str,
    runner: Any,
    model_name: str,
    port: int,
    broadcast_addr: str,
    inside_port: int,
    backend_config: Dict[str, Any],
) -> List[str]:
    """Build a container run command — shared by docker.py and podman.py.

    `backend_config` is the model_data from start() containing at minimum
    {"backend": <id>, ...}.  The backend ID is resolved here to get the full
    backend dict (image, devices, env etc.).
    """
    # Resolve the full backend dict from the backend_id in model_data.
    backend_id = backend_config.get("backend")
    if not backend_id:
        backends = runner.cm.data.get("backends", {})
        if isinstance(backends, dict):
            backend_id = backends.get("default")

    full_backend: Dict[str, Any] = {}
    if backend_id:
        resolved = runner.cm.get_backend(backend_id) or {}
        full_backend = dict(resolved)

    image = str(full_backend.get("image") or _default_image(runner))
    devices: List[str] = list(full_backend.get("devices", []))
    container_env: Dict[str, str] = dict(full_backend.get("env_container", {}))
    backend_runner_id: str | None = full_backend.get("runner") or ""

    binary_info = resolve_binary_from_backend(full_backend)
    binary_dir: str | None = None
    if binary_info is not None:
        binary_path, extra_devs = binary_info
        if not devices:
            devices = list(extra_devs)
        binary_dir = os.path.dirname(binary_path) or binary_dir

    parts: List[str] = [cmd, "run", "-d"]
    parts.extend(["-e", f"PORT={port}"])
    for k, v in container_env.items():
        parts.extend(["-e", f"{k}={v}"])

    ld = os.environ.get("LD_LIBRARY_PATH")
    if ld:
        parts.extend(["-e", f"LD_LIBRARY_PATH={ld}"])

    parts.extend(["--name", safe_container_name(model_name, port)])
    parts.extend(["-p", f"{broadcast_addr}:{port}:{inside_port}"])

    # Devices
    for dev in devices:
        parts.extend(["--device", str(dev)])

    # Mount host binary dir
    if binary_dir and os.path.isdir(binary_dir):
        parts.extend(["-v", f"{binary_dir}:{binary_dir}:ro"])

    # Resolve llama-server args
    result = build_model_args(
        runner.cm, model_name,
        env_vars={"PORT": str(port)},
        override_backend=backend_id,
        inference_kwargs=runner._inference_kwargs.get(model_name, {}),
    )
    arg_list: List[str] = list(result[0]) if result else []

    # Fix --port → inside_port; ensure --host is present.
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
        fixed.extend(["--host", broadcast_addr])

    # ENTRYPOINT (llama-launch.sh) has the binary baked in — pass CLI args only.
    parts.extend([image] + fixed)
    return parts


class ContainerModelRunner(BaseModelRunner, ABC):
    """Abstract base for container-based model runners."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    async def get_logs(self, model_name: str, lines: int = 100) -> List[str]:
        """Return the last N log lines for a model via container runtime."""
        ctx = self._models.get(model_name)
        if not ctx:
            return []
        cid = getattr(ctx, "container_id", None)
        name = getattr(ctx, "name", model_name)
        cmd_parts = [self._container_cmd(), "logs"]
        if lines:
            cmd_parts.extend(["--tail", str(lines)])
        if not cid:
            # Container may still be starting — try by name
            cmd_parts.append(name)
        else:
            cmd_parts.append(cid)
        proc = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=SUBPROCESS_ENV,
        )
        stdout, _stderr = await proc.communicate()
        if proc.returncode != 0:
            return []
        text = stdout.decode("utf-8", errors="replace")
        return [line for line in text.splitlines() if line]

    async def _release_port(self, port: int) -> None:
        """Wait up to ``port_drain_timeout`` seconds for a stopped container's
        listener to release the port.  Uses non-blocking subprocess calls so the
        event loop can be cancelled if shutdown proceeds.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self.port_drain_timeout
        while loop.time() < deadline:
            proc = await asyncio.create_subprocess_exec(
                "lsof", f"-ti:{port}",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            if not stdout.strip():
                return
            await asyncio.sleep(0.2)

    async def _before_restart(self, ctx: _ModelContext) -> bool:
        """Cancel active log capture and clear stale container reference."""
        self._cancel_log_task(ctx)
        ctx.container_id = None
        return await super()._before_restart(ctx)

    async def shutdown(self) -> None:
        """Full teardown — stop models, force-remove containers, clear state."""
        cids = [getattr(ctx, "container_id", None) for ctx in list(self._models.values())]
        await super().shutdown()
        await self._remove_containers(cids)

    @abstractmethod
    def _container_cmd(self) -> str:
        """Return the container runtime command (e.g. 'docker', 'podman')."""

    @abstractmethod
    async def _remove_containers(self, cids: list) -> None:
        """Force-remove a list of stale container IDs."""

    async def _watch_container(self, model_name: str, ctx: _ModelContext) -> None:
        """Poll container status and restart on unexpected exit."""
        cid = getattr(ctx, "container_id", None)
        if not cid:
            return

        while True:
            await asyncio.sleep(2.0)  # poll interval

            try:
                proc = await asyncio.create_subprocess_exec(
                    self._container_cmd(), "inspect", cid, "--format", "{{.State.Status}}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=SUBPROCESS_ENV,
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
                from model_arkestra import logger
                logger.warning(
                    f"Error inspecting {self._container_cmd()} container {cid}: {e}"
                )
                continue

    # ── Log capture helpers (Docker SDK follow-stream) ───────────

    def _cancel_log_task(self, ctx: _ModelContext) -> None:
        """Cancel the asyncio log-capture task for a model (if any)."""
        if not hasattr(self, '_log_tasks'):
            return
        task = self._log_tasks.pop(ctx.name, None)
        if task and not task.done():
            task.cancel()

    async def _stop_model_process(self, ctx: _ModelContext) -> None:
        """Stop a container gracefully (cancel log stream first), falling back to force-kill."""
        # Cancel log capture before stopping the container
        self._cancel_log_task(ctx)
        cid = getattr(ctx, "container_id", None)
        if not cid:
            return
        try:
            stop = await asyncio.create_subprocess_exec(
                self._container_cmd(), "stop", "--time", str(self.port_drain_timeout), cid,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=SUBPROCESS_ENV,
            )
            await stop.wait()
        except Exception:
            pass
        ctx.container_id = None
