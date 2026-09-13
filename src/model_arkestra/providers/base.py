"""Provider ABC — the inference surface of a single model.

Every provider exposes the full modality set; methods the model can't do raise
``NotSupported``. This keeps call sites uniform (``model.provider.embed(...)``)
while capability gating stays explicit and cheap.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict


class NotSupported(RuntimeError):
    """Raised when a provider is asked for a modality the model doesn't offer."""


class Provider:
    """Base inference provider. Subclasses implement the modalities they serve."""

    #: Modalities this provider actually serves (subset of the full set).
    capabilities: frozenset = frozenset()

    async def probe(self) -> bool:
        """Return True if the model is reachable/ready. Default: not probed."""
        return False

    # ── chat / stream (the unified front) ────────────────────────
    async def invoke_full(self, prompt: str = "", **kwargs: Any) -> Dict[str, Any]:
        """Run one completion, returning ``{"content", "usage"}``."""
        raise NotSupported("chat")

    async def stream(self, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Yield SSE events (token / usage) for a streaming chat."""
        raise NotSupported("stream")
        yield  # pragma: no cover  (makes this an async generator)

    # ── specialized modalities ───────────────────────────────────
    async def embed(self, text: str) -> Dict[str, Any]:
        raise NotSupported("embed")

    async def transcribe(self, audio_bytes: bytes, language: str = "") -> Dict[str, Any]:
        raise NotSupported("asr")

    async def synthesize(self, text: str, voice: str = "", speed: float = 1.0) -> bytes:
        raise NotSupported("tts")

    # ── streaming variants (ONNX only) ───────────────────────────
    async def stream_asr(self, audio_bytes: bytes) -> Dict[str, Any]:
        raise NotSupported("stream-asr")

    async def stream_tts(self, text: str) -> bytes:
        raise NotSupported("stream-tts")

    # ── generic passthrough (llama/remote only) ──────────────────
    async def request(self, path: str, **kwargs: Any) -> Any:
        raise NotSupported("request")
