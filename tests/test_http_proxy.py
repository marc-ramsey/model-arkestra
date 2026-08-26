"""Tests for model_arkestra.http_proxy — shared SSE parser, completion helper, and status mapper."""
from __future__ import annotations

import asyncio
import json

import pytest

from model_arkestra.http_proxy import (
    _SSEParser,
    sse_events,
    parse_completion,
    model_status,
    model_status_for_ctx,
)
from model_arkestra.types import RunnerState


# ═══════════════════════════════════════════════════════════════
# _SSEParser.parse_line — single line unit tests
# ═══════════════════════════════════════════════════════════════


class TestSSEParser:
    """Unit tests for the stateless SSE parser."""

    def test_token_content(self):
        chunk = {"choices": [{"delta": {"content": "Hello"}}]}
        result = _SSEParser.parse_line(f"data: {json.dumps(chunk)}")
        assert result == {"token": "Hello"}

    def test_reasoning_content_not_yielded(self):
        """Delta without 'content' returns None — reasoning delta is separate."""
        chunk = {"choices": [{"delta": {"reasoning": "Thinking..."}}]}
        result = _SSEParser.parse_line(f"data: {json.dumps(chunk)}")
        assert result is None

    def test_usage_event(self):
        chunk = {"usage": {"prompt_tokens": 5, "completion_tokens": 10}}
        result = _SSEParser.parse_line(f"data: {json.dumps(chunk)}")
        assert result == {"usage": {"prompt_tokens": 5, "completion_tokens": 10}}

    def test_done_marker(self):
        result = _SSEParser.parse_line("data: [DONE]")
        assert result == {"done": True}

    def test_empty_data_returns_none(self):
        assert _SSEParser.parse_line("") is None

    def test_non_data_prefix_returns_none(self):
        assert _SSEParser.parse_line(": something") is None
        assert _SSEParser.parse_line("comment: foo") is None

    def test_invalid_json_returns_none(self):
        result = _SSEParser.parse_line("data: {broken")
        assert result is None

    def test_empty_choices_returns_none(self):
        chunk = {"choices": []}
        result = _SSEParser.parse_line(f"data: {json.dumps(chunk)}")
        assert result is None


# ═══════════════════════════════════════════════════════════════
# sse_events — async generator tests (no network)
# ═══════════════════════════════════════════════════════════════


class TestSSEEvents:
    """Tests for the SSE event stream parser."""

    @staticmethod
    def _make_lines(lines: list[str]):
        """Wrap raw strings into a list of async-iterated items."""
        async def gen():
            for l in lines:
                yield l
        return gen()

    @pytest.mark.asyncio
    async def test_yields_tokens_and_done(self):
        raw = self._make_lines([
            f"data: {json.dumps({'choices': [{'delta': {'content': 'A'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': 'B'}}]})}",
            "data: [DONE]",
        ])
        events = [e async for e in sse_events(raw)]
        assert len(events) == 3
        assert events[0] == {"token": "A"}
        assert events[1] == {"token": "B"}
        assert events[2] == {"done": True}

    @pytest.mark.asyncio
    async def test_empty_lines_skipped(self):
        raw = self._make_lines(["", "\n", f"data: {json.dumps({'choices': [{'delta': {'content': 'x'}}]})}"])
        events = [e async for e in sse_events(raw)]
        assert len(events) == 1
        assert events[0] == {"token": "x"}

    @pytest.mark.asyncio
    async def test_usage_event_in_stream(self):
        raw = self._make_lines([
            f"data: {json.dumps({'choices': [{'delta': {'content': 'Hi'}}]})}",
            f"data: {json.dumps({'usage': {'prompt_tokens': 2}})}",
            "data: [DONE]",
        ])
        events = [e async for e in sse_events(raw)]
        assert {"token": "Hi"} in events
        assert {"usage": {"prompt_tokens": 2}} in events

    @pytest.mark.asyncio
    async def test_no_events_returns_empty(self):
        raw = self._make_lines(["", "", ""])
        events = [e async for e in sse_events(raw)]
        assert events == []


# ═══════════════════════════════════════════════════════════════
# parse_completion — non-streaming response extraction
# ═══════════════════════════════════════════════════════════════


class TestParseCompletion:
    """Tests for extracting {content, usage} from a chat.completion response."""

    def test_standard_response(self):
        data = {
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }
        result = parse_completion(data)
        assert result["content"] == "Hello world"
        assert result["usage"]["prompt_tokens"] == 3

    def test_reasoning_content_fallback(self):
        data = {
            "choices": [{"message": {"role": "assistant", "reasoning_content": "Let me think..."}}],
        }
        result = parse_completion(data)
        assert result["content"] == "Let me think..."

    def test_empty_choices(self):
        result = parse_completion({"choices": []})
        assert result["content"] == ""
        assert isinstance(result["usage"], dict)

    def test_reasoning_and_content_prefers_content(self):
        """When both content and reasoning_content exist, content wins."""
        data = {"choices": [{"message": {"content": "Final answer", "reasoning_content": "Thinking..."}}]}
        result = parse_completion(data)
        assert result["content"] == "Final answer"


# ═══════════════════════════════════════════════════════════════
# model_status — RunnerState → WebUI status mapping
# ═══════════════════════════════════════════════════════════════


class TestModelStatus:
    """Tests for the state-to-WebUI-status mapper."""

    def test_running_becomes_loaded(self):
        assert model_status(RunnerState.RUNNING) == {"value": "loaded"}

    def test_loading_stays_loading(self):
        assert model_status(RunnerState.LOADING) == {"value": "loading"}

    def test_stopped_becomes_sleeping(self):
        assert model_status(RunnerState.STOPPED) == {"value": "sleeping"}

    def test_stopping_becomes_sleeping(self):
        assert model_status(RunnerState.STOPPING) == {"value": "sleeping"}

    def test_uncached_becomes_unloaded(self):
        assert model_status(RunnerState.UNCACHED) == {"value": "unloaded"}

    def test_error_with_message(self):
        result = model_status(RunnerState.ERROR, "oom killed")
        assert result == {"value": "error", "error_message": "oom killed"}

    def test_error_without_message(self):
        result = model_status(RunnerState.ERROR)
        assert result == {"value": "error", "error_message": "unknown error"}


# ═══════════════════════════════════════════════════════════════
# model_status_for_ctx — context-aware wrapper
# ═══════════════════════════════════════════════════════════════


class TestModelStatusForCtx:
    """Tests for the None-safe context wrapper."""

    def test_none_context_is_stopped(self):
        assert model_status_for_ctx(None) == {"value": "stopped"}

    def test_empty_mock_context_runs(self):
        ctx = type("Ctx", (), {"state": RunnerState.RUNNING, "last_error": None})()
        result = model_status_for_ctx(ctx)
        assert result == {"value": "loaded"}
