"""Intermediate base class for container-based runners (Podman / Docker).

Contains shared logic for container lifecycle, health watching, restart behavior,
and port release — eliminating near-duplicate code between docker.py and podman.py.
"""
from __future__ import annotations
import asyncio
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from model_arkestra.base import BaseModelRunner
from model_arkestra.common import INSPECT_RE
from model_arkestra.types import RunnerState, _ModelContext


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
            env=os.environ,
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
                env=os.environ,
            )
            await stop.wait()
        except Exception:
            pass
        ctx.container_id = None
