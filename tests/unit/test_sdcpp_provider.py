"""Tests for SdcppProvider — image-gen HTTP provider and engine-keyed registry."""
from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────

def _fake_response(status: int, payload: dict):
    """Async context-manager mimicking an aiohttp client response."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=payload)
    resp.text = AsyncMock(return_value=json.dumps(payload))
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.fixture()
def sdcpp_provider():
    from model_arkestra.providers.sdcpp import SdcppProvider
    return SdcppProvider("qwen-image", 18019)


# ── generate_image ─────────────────────────────────────────────────────────

class TestGenerateImage:

    @pytest.mark.asyncio
    async def test_returns_decoded_bytes(self, sdcpp_provider):
        png = b"\x89PNG\r\n\x1a\nfake"
        b64 = base64.b64encode(png).decode()
        resp = _fake_response(200, {"created": 1, "data": [{"b64_json": b64}],
                                    "output_format": "png"})
        with patch("model_arkestra.providers.sdcpp.aiohttp.ClientSession") as sess:
            sess.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
                post=MagicMock(return_value=resp)))
            out = await sdcpp_provider.generate_image("a cat")
        assert out == png

    @pytest.mark.asyncio
    async def test_payload_shape(self, sdcpp_provider):
        resp = _fake_response(200, {"data": [{"b64_json": base64.b64encode(b"x").decode()}]})
        with patch("model_arkestra.providers.sdcpp.aiohttp.ClientSession") as sess:
            inner = MagicMock(post=MagicMock(return_value=resp))
            sess.return_value.__aenter__ = AsyncMock(return_value=inner)
            await sdcpp_provider.generate_image("a cat", size="1328x1328", n=2)
        kwargs = inner.post.call_args.kwargs
        body = kwargs["json"]
        assert body["prompt"] == "a cat"
        assert body["size"] == "1328x1328"
        assert body["n"] == 2
        url = inner.post.call_args.args[0]
        assert url.endswith("/v1/images/generations")

    @pytest.mark.asyncio
    async def test_no_size_omits_key(self, sdcpp_provider):
        resp = _fake_response(200, {"data": [{"b64_json": base64.b64encode(b"x").decode()}]})
        with patch("model_arkestra.providers.sdcpp.aiohttp.ClientSession") as sess:
            inner = MagicMock(post=MagicMock(return_value=resp))
            sess.return_value.__aenter__ = AsyncMock(return_value=inner)
            await sdcpp_provider.generate_image("a cat")
        assert "size" not in inner.post.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_error_status_raises(self, sdcpp_provider):
        from model_arkestra.types import RunnerError
        resp = _fake_response(500, {"error": "boom"})
        with patch("model_arkestra.providers.sdcpp.aiohttp.ClientSession") as sess:
            sess.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
                post=MagicMock(return_value=resp)))
            with pytest.raises(RunnerError):
                await sdcpp_provider.generate_image("a cat")


# ── probe ──────────────────────────────────────────────────────────────────

class TestProbe:

    @pytest.mark.asyncio
    async def test_probe_uses_v1_models(self, sdcpp_provider):
        resp = _fake_response(200, {"data": []})
        with patch("model_arkestra.providers.sdcpp.aiohttp.ClientSession") as sess:
            inner = MagicMock(get=MagicMock(return_value=resp))
            sess.return_value.__aenter__ = AsyncMock(return_value=inner)
            assert await sdcpp_provider.probe() is True
        url = inner.get.call_args.args[0]
        assert url.endswith("/v1/models")

    @pytest.mark.asyncio
    async def test_probe_unreachable(self, sdcpp_provider):
        import aiohttp
        with patch("model_arkestra.providers.sdcpp.aiohttp.ClientSession") as sess:
            inner = MagicMock(get=MagicMock(
                side_effect=aiohttp.ClientConnectionError("refused")))
            sess.return_value.__aenter__ = AsyncMock(return_value=inner)
            assert await sdcpp_provider.probe() is False


# ── registry: providers keyed by engine ────────────────────────────────────

class TestProviderRegistry:

    def test_sdcpp_engine_maps_to_sdcpp_provider(self):
        from model_arkestra.providers import PROVIDERS
        from model_arkestra.providers.sdcpp import SdcppProvider
        assert PROVIDERS["sdcpp"] is SdcppProvider

    def test_llama_engine_maps_to_llama_provider(self):
        from model_arkestra.providers import PROVIDERS
        from model_arkestra.providers.llama import LlamaProvider
        assert PROVIDERS["llama-cpp"] is LlamaProvider


# ── engine registry (arg builders keyed by engine name) ────────────────────

class TestEngineRegistry:

    def test_sdcpp_registered(self):
        from model_arkestra.engines import ENGINES
        from model_arkestra.sdcpp import SdcppEngine
        assert ENGINES["sdcpp"] is SdcppEngine

    def test_llama_registered(self):
        from model_arkestra.engines import ENGINES
        from model_arkestra.llama_cpp import LlamaCppEngine
        assert ENGINES["llama-cpp"] is LlamaCppEngine

    def test_unknown_engine_raises(self):
        from model_arkestra.engines import build_args, EngineError
        with pytest.raises(EngineError):
            build_args("nope", {"model": "x"}, 1234)
