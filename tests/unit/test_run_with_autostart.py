"""Regression: _run_with_autostart must pass model_name to the callee.

The tag-routed endpoints (embed/asr/tts/image-gen) call
``_run_with_autostart(model_name, self._arkestra.embed, input_text)``;
a bound method still needs model_name first, so dropping it produces
``missing 1 required positional argument`` 500s at runtime.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from model_arkestra.types import ModelNotStarted, RunnerState
from model_arkestra.server import ArkestraServer


def _make_server():
    srv = ArkestraServer.__new__(ArkestraServer)
    arkestra = MagicMock()
    arkestra.model_obj.return_value = MagicMock(state=RunnerState.RUNNING)
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


# ── _ensure_running: the pre-check that makes tag-routed endpoints autostart ──

def _ctx(state):
    ctx = MagicMock()
    ctx.state = state
    return ctx


@pytest.mark.asyncio
async def test_ensure_running_noop_when_running():
    srv, arkestra = _make_server()
    arkestra.model_obj.return_value = _ctx(RunnerState.RUNNING)
    await srv._ensure_running("embed-model")
    arkestra.start.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_running_starts_stopped_model():
    """The nomic-embed bug: a STOPPED model must be started, not 500."""
    srv, arkestra = _make_server()
    arkestra.model_obj.return_value = _ctx(RunnerState.STOPPED)
    arkestra.get_model.return_value = {}
    arkestra.cm.data = {"default": {}}
    # _wait_for_ready polls model_obj; make it report RUNNING after start.
    arkestra.model_obj.side_effect = [_ctx(RunnerState.STOPPED),
                                      _ctx(RunnerState.RUNNING)]
    await srv._ensure_running("embed-model")
    arkestra.start.assert_awaited_once_with("embed-model")


@pytest.mark.asyncio
async def test_ensure_running_rejects_uncached():
    from fastapi import HTTPException
    srv, arkestra = _make_server()
    arkestra.model_obj.return_value = _ctx(RunnerState.UNCACHED)
    with pytest.raises(HTTPException) as exc:
        await srv._ensure_running("embed-model")
    assert exc.value.status_code == 503
    arkestra.start.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_running_starts_unknown_model():
    """ctx is None (never registered) → still start it."""
    srv, arkestra = _make_server()
    arkestra.model_obj.side_effect = [None, _ctx(RunnerState.RUNNING)]
    arkestra.get_model.return_value = {}
    arkestra.cm.data = {"default": {}}
    await srv._ensure_running("embed-model")
    arkestra.start.assert_awaited_once_with("embed-model")
