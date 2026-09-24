from __future__ import annotations
from pathlib import Path
from typing import Any, List

from model_arkestra.container_runner import ContainerRunner


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

    def _extra_run_args(self) -> List[str]:
        return ["--replace", "--group-add", "keep-groups"]
