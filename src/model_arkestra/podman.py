from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from model_arkestra.binary_downloader import BinaryDownloader, BinaryDownloaderError
from model_arkestra.container_runner import ContainerRunner
from model_arkestra.common import SUBPROCESS_ENV, safe_container_name
from model_arkestra.types import _ModelContext


class PodmanRunner(ContainerRunner):
    INSIDE_PORT = 8080

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._image_cache_dir = Path(
            kwargs.get("image_cache_dir", "~/.local/share/model-arkestra/bin-cache")
        ).expanduser()
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)

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

    def _extra_run_args(self) -> List[str]:
        return ["--replace", "--group-add", "keep-groups"]
