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
from pathlib import Path
from typing import Any, Dict, List

from model_arkestra.base import BaseModelRunner
from model_arkestra.common import (
    INSPECT_RE, SUBPROCESS_ENV, build_model_args, default_cache_root,
    resolve_binary_from_backend, safe_container_name,
)
from model_arkestra.types import RunnerState, _ModelContext


def _resolve_backend(
    runner: Any, ctx_or_none: Any | None, model_data: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Resolve the effective backend dict for a model launch.

    Priority: ``ctx.backend_id`` → ``model_data["backend"]`` → config backends.default.
    Returns the full backend dict (via ``runner.cm.get_backend()``) or **None** if
    nothing resolves.
    """
    backend_id = None

    # 1. Explicit context override (e.g. restart with a different backend)
    if ctx_or_none is not None:
        backend_id = getattr(ctx_or_none, "backend_id", None)

    # 2. Per-model backend from config
    if not backend_id:
        backend_id = model_data.get("backend")

    # 3. Global default
    if not backend_id:
        backends = runner.cm.data.get("backends")
        if isinstance(backends, dict):
            backend_id = backends.get("default")

    return runner.cm.get_backend(backend_id) if backend_id else None


def _build_container_cmd(
    cmd: str,
    runner: Any,
    model_name: str,
    port: int,
    broadcast_addr: str,
    inside_port: int,
    backend_config: Dict[str, Any],
    backend_id: Optional[str] = None,
) -> List[str]:
    """Build a container run command for the backend-config architecture.

    ``backend_config`` is the full backend dict (from ``cm.get_backend()``).
    """
    devices: List[str] = list(backend_config.get("devices", []))
    container_env: Dict[str, str] = dict(backend_config.get("env_container", {}))

    # Resolve binary directory (optional mount)
    binary_info = resolve_binary_from_backend(backend_config)
    binary_dir: str | None = None
    if binary_info is not None:
        binary_path, extra_devs = binary_info
        if not devices:
            devices = list(extra_devs)
        binary_dir = os.path.dirname(binary_path) or binary_dir

    # Resolve host HF cache directory for volume mount
    cache_path: str | None = (runner.cm.data.get("env") or {}).get("HF_HUB_CACHE")
    if not cache_path:
        cache_path = os.environ.get("HF_HUB_CACHE")
    if not cache_path:
        cache_path = str(default_cache_root())
    cache_path = Path(cache_path).expanduser()

    image = str(backend_config.get("image"))
    image = runner._resolve_image(image)

    parts: List[str] = [cmd, "run", "-d"]
    parts.extend(["-e", f"PORT={port}"])
    for k, v in container_env.items():
        parts.extend(["-e", f"{k}={v}"])

    ld = os.environ.get("LD_LIBRARY_PATH")
    if ld:
        parts.extend(["-e", f"LD_LIBRARY_PATH={ld}"])

    parts.extend(["--name", safe_container_name(model_name, port)])
    parts.extend(["-p", f"{broadcast_addr}:{port}:{inside_port}"])

    for dev in devices:
        parts.extend(["--device", str(dev)])

    if binary_dir and os.path.isdir(binary_dir):
        parts.extend(["-v", f"{binary_dir}:/llm-server/bin:ro"])

    # Mount host HF cache at /data/hf and ensure env var points there
    _inside_cache = "/data/hf"
    if cache_path and os.path.isdir(str(cache_path)):
        parts.extend(["-v", f"{cache_path}:{_inside_cache}:rw"])
        parts.extend(["-e", f"HF_HUB_CACHE={_inside_cache}"])
    result = build_model_args(
        runner.cm, model_name,
        env_vars={"PORT": str(port)},
        override_backend=backend_id,
        inference_kwargs=runner._inference_kwargs.get(model_name, {}),
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
        fixed.extend(["--host", "0.0.0.0"])

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
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return []
        text = stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")
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

    async def _before_restart(self, ctx: _ModelContext, new_size=None) -> bool:
        """Cancel active log capture and clear stale container reference."""
        self._cancel_log_task(ctx)
        ctx.container_id = None
        return await super()._before_restart(ctx, new_size)

    async def shutdown(self) -> None:
        """Full teardown — stop models, force-remove containers, clear state."""
        cids = [getattr(ctx, "container_id", None) for ctx in list(self._models.values())]
        await super().shutdown()
        await self._remove_containers(cids)

    @abstractmethod
    def _container_cmd(self) -> str:
        """Return the container runtime command (e.g. 'docker', 'podman')."""

    def _resolve_image(self, image: str) -> str:
        return image

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

    # ── Log capture helpers (container log streaming) ───────────

    async def _capture_container_logs(
        self, model_name: str, container_id: str
    ) -> None:
        """Stream container logs into ctx._log_buffer ring buffer."""
        proc = await asyncio.create_subprocess_exec(
            self._container_cmd(), "logs", "-f", "--tail", "0", container_id,
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
                        ctx._append_log_line(line)

        tasks = []
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                tasks.append(asyncio.create_task(_read_stream(stream)))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            proc.kill()
            raise

    def _cancel_log_task(self, ctx: _ModelContext) -> None:
        """Cancel the asyncio log-capture task(s) for a model (if any).

        Handles both the legacy single-task format and the new tuple-of-two-tasks
        format introduced when stdout/stderr are now captured concurrently.
        """
        if not hasattr(self, '_log_tasks'):
            return
        entry = self._log_tasks.pop(ctx.name, None)
        if isinstance(entry, tuple):
            for task in entry:
                if task and not task.done():
                    task.cancel()
        elif entry and not entry.done():
            entry.cancel()

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
