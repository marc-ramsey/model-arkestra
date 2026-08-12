"""Intermediate base class for container-based runners (Podman / Docker).

Contains shared logic for container lifecycle, health watching, restart behavior,
and port release — eliminating near-duplicate code between docker.py and podman.py.
"""
from __future__ import annotations
import asyncio
import os
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from model_arkestra.base import BaseModelRunner
from model_arkestra.common import INSPECT_RE
from model_arkestra.types import RunnerState, _ModelContext


class ContainerModelRunner(BaseModelRunner, ABC):
    """Abstract base for container-based model runners."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

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

    async def _before_restart(self, ctx: _ModelContext) -> bool:
        """Ensure stale container reference is cleared before restarting."""
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

    async def _stop_model_process(self, ctx: _ModelContext) -> None:
        """Stop a container gracefully, falling back to force-kill."""
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
