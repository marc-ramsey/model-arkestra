from __future__ import annotations
import asyncio
import os
import shlex
from pathlib import Path
from typing import Any, Dict, Optional

from model_arkestra.binary_downloader import BinaryDownloader, BinaryDownloaderError
from model_arkestra.container_runner import ContainerModelRunner, _resolve_backend, _build_container_cmd
from model_arkestra.common import SUBPROCESS_ENV, safe_container_name
from model_arkestra.types import _ModelContext


class PodmanModelRunner(ContainerModelRunner):
    INSIDE_PORT = 8080

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Cache dir for OCI image pulls (reuses binary downloader cache)
        self._image_cache_dir = Path(
            kwargs.get("image_cache_dir", "~/.local/share/model-arkestra/bin-cache")
        ).expanduser()
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)

    def _container_cmd(self) -> str:
        return "podman"

    async def _resolve_image_from_source(
        self, image: str, source_ref: Optional[str]
    ) -> str:
        """Resolve image via BinaryDownloader if backend references an OCI-image source.
        Falls back to raw image string if no source_ref.
        """
        if not source_ref or not isinstance(source_ref, str):
            return image

        sources = self.cm.data.get("sources") or {}
        source_cfg = sources.get(source_ref)
        if not source_cfg:
            return image

        # Only process oci-image type sources
        if source_cfg.get("type") != "oci-image":
            return image

        downloader = BinaryDownloader(
            cache_dir=self._image_cache_dir,
            backend_id=source_ref,
            source_cfg=source_cfg,
        )
        try:
            # Pull the image (async, may take time on first run)
            resolved = await downloader.resolve(version="latest")
            return str(resolved)
        except BinaryDownloaderError as e:
            # Fall back to raw image — container runtime will handle pull
            self.logger.warning(
                f"OCI source resolution failed for {source_ref}: {e}. "
                f"Using raw image reference: {image}"
            )
            return image

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
        backend = _resolve_backend(self, ctx, model_data)
        if backend is None:
            raise RuntimeError(
                f"No backend resolved for podman model '{ctx.name}' — "
                "configure a backend with an 'image' key."
            )

        # Resolve image from source (pulls via downloader if oci-image source)
        raw_image = str(backend.get("image", ""))
        source_ref = backend.get("source_ref")

        if source_ref:
            sources = self.cm.data.get("sources") or {}
            source_cfg = sources.get(source_ref)
            if source_cfg and source_cfg.get("type") == "oci-image":
                # Image comes entirely from the OCI image source
                raw_image = ""

        image = await self._resolve_image_from_source(raw_image, source_ref)
        if not image:
            raise RuntimeError(
                f"No container image resolved for podman backend '{ctx.backend_id}'. "
                f"Configure an 'image' key or an oci-image 'source_ref' in the backend."
            )
        if "/" not in image:
            image = f"localhost/{image}"

        # Inject resolved image into backend for _build_container_cmd
        backend["image"] = image

        cmd_parts = _build_container_cmd(
            "podman", self, ctx.name, ctx.port,
            "0.0.0.0", PodmanModelRunner.INSIDE_PORT,
            backend,
            backend_id=ctx.backend_id,
        )
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
