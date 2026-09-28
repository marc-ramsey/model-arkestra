"""Regression: _run_with_autostart must pass model_name to the callee.

The tag-routed endpoints (embed/asr/tts/image-gen) call
``_run_with_autostart(model_name, self._arkestra.embed, input_text)``;
a bound method still needs model_name first, so dropping it produces
``missing 1 required positional argument`` 500s at runtime.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from model_arkestra.types import ModelNotStarted
from model_arkestra.server import ArkestraServer


def _make_server():
    srv = ArkestraServer.__new__(ArkestraServer)
    arkestra = MagicMock()
    arkestra.model_obj.return_value = MagicMock(state="RUNNING")
    arkestra.start = AsyncMock()
    arkestra.embed = AsyncMock(return_value={"embeddings": [[0.1]]})
    srv._arkestra = arkestra
    return srv, arkestra


@pytest.mark.asyncio
async def test_model_name_forwarded():
    srv, arkestra = _make_server()
    result = await srv._run_with_autostart("embed-model", arkestra.embed, "hello")
    assert result == {"embeddings": [[0.1]]}
    # model_name must be the first positional arg of the call.
    args, _ = arkestra.embed.call_args
    assert args[0] == "embed-model"
    assert args[1] == "hello"


@pytest.mark.asyncio
async def test_model_name_forwarded_after_autostart():
    srv, arkestra = _make_server()
    arkestra.embed = AsyncMock(
        side_effect=[ModelNotStarted("stopped"), {"embeddings": [[0.2]]}]
    )
    result = await srv._run_with_autostart("embed-model", arkestra.embed, "hello")
    assert result == {"embeddings": [[0.2]]}
    arkestra.start.assert_awaited_once_with("embed-model")
    args, _ = arkestra.embed.call_args
    assert args[0] == "embed-model"
    assert args[1] == "hello"
