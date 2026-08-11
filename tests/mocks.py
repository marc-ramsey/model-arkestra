"""Shared fixtures and mocks for model_runner tests."""

from __future__ import annotations
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

import pytest

from model_arkestra.types import RunnerState, _ModelContext


def mock_cm(model_data: dict | None = None):
    """Create a ConfigManager mock that returns *model_data* from get_model()."""
    cm = MagicMock()
    cm.get_model.return_value = model_data or {}
    cm.get_vector.return_value = None  # no env section by default
    return cm


def _make_model(name: str, port: int, state: RunnerState, has_process: bool = False):
    """Build a ready-to-use _ModelContext with the given state."""
    ctx = _ModelContext(name, port)
    ctx.state = state
    if has_process:
        fake_proc = MagicMock()
        fake_proc.pid = 12345
        fake_proc.returncode = None
        ctx.process = fake_proc
    return ctx


class _MockSSEStream:
    """Async context-manager wrapper mimicking aiohttp.ResponseContentReader."""

    def __init__(self, lines: list[str]):
        self._lines = iter(lines)
        self._at_end = False

    async def readline(self) -> str:
        try:
            return next(self._lines) + "\n"
        except StopIteration:
            self._at_end = True
            return ""


class _MockSSEResp:
    """Async context-manager mimicking aiohttp.ClientResponse for streaming."""

    def __init__(self, *, status=200, lines: list[str] | None = None):
        self.status = status
        self._lines = lines or ["data: {\"choices\":[{\"delta\":{}}]}", "data: [DONE]"]
        self._content_iter = None

    async def __aenter__(self):
        self._content_iter = _MockSSEStream(self._lines)
        return self

    async def __aexit__(self, *a):
        pass

    @property
    def content(self):
        return self._content_iter


class _MockResp:
    """Async context-manager mimicking aiohttp.ClientResponse for non-streaming."""

    def __init__(self, *, status=200, json_data: dict | None = None):
        self.status = status
        self._json_data = json_data or {"choices": [{"message": {"content": "hello"}}], "usage": {}}
        self._json_called = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def json(self) -> dict:
        self._json_called = True
        return self._json_data


class _MockSession:
    """Fake aiohttp.ClientSession that can be swapped in via patch."""

    def __init__(self):
        self.post_calls = []  # track how many times post was called

    def post(self, *a, **kw):
        call_num = len(self.post_calls)
        self.post_calls.append({"args": a, "kwargs": kw})
        return self._post_responses.get(call_num, _MockResp())

    @property
    def _post_responses(self) -> dict[int, _MockResp]:
        """Override this property in subclasses or set it on instances."""
        # Default: first call succeeds with content+usage.
        return {0: _MockResp(status=200, json_data={
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"model": "m", "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })}


    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


@pytest.fixture()
def running_ctx():
    """A model context in RUNNING state with a fake process handle."""
    return _make_model("running-model", 18000, RunnerState.RUNNING, has_process=True)


@pytest.fixture()
def stopped_ctx():
    """A model context in STOPPED state (never started)."""
    return _make_model("stopped-model", 18001, RunnerState.STOPPED)


@pytest.fixture()
def stopping_ctx():
    """A model context being stopped."""
    return _make_model("stopping-model", 18002, RunnerState.STOPPING)
