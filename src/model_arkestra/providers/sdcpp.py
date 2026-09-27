"""SdcppProvider — stable-diffusion.cpp sd-server, text-to-image.

Bound to one port (one model). Serves the OpenAI-compatible images API:
POST /v1/images/generations. The engine is sd-server, not llama-server —
same GGUF weights, different binary and flag surface.
"""
from __future__ import annotations

import asyncio
import base64
from typing import Any, Dict, Optional

import aiohttp

from model_arkestra.providers.base import Provider
from model_arkestra.types import RunnerError


class SdcppProvider(Provider):
    capabilities = frozenset({"image-gen"})

    def __init__(self, model_name: str, port: int):
        self.model_name = model_name
        self.port = port
        self._base = f"http://127.0.0.1:{port}"

    async def probe(self) -> bool:
        # sd-server has no /health; /v1/models is the OpenAI liveness probe.
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._base}/v1/models", timeout=5) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def generate_image(self, prompt: str, *,
                             size: Optional[str] = None,
                             n: int = 1,
                             output_format: str = "png") -> bytes:
        """Generate one image; returns the decoded image bytes.

        *size* is sd-server's WxH form (e.g. "1328x1328").
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "n": n,
            "output_format": output_format,
        }
        if size:
            payload["size"] = size

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base}/v1/images/generations",
                json=payload, timeout=600,
            ) as resp:
                if resp.status != 200:
                    detail = (await resp.text())[:500]
                    raise RunnerError(f"Image generation error: {resp.status}: {detail}")
                data = await resp.json()

        items = data.get("data") or []
        if not items or "b64_json" not in items[0]:
            raise RunnerError("Image generation returned no image data")
        return base64.b64decode(items[0]["b64_json"])

