"""Tests for StreamStats and parse_completion in http_proxy."""
from __future__ import annotations

import json

import pytest

from model_arkestra.http_proxy import (
    StreamStats,
    parse_completion,
    model_status,
    model_status_for_ctx,
)
from model_arkestra.types import RunnerState


def _sse(*chunks: dict) -> bytes:
    """Build raw SSE bytes from a list of OpenAI chunk dicts."""
    parts = []
    for c in chunks:
        parts.append(f"data: {json.dumps(c)}\n\n")
    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode()


# ── StreamStats ───────────────────────────────────────────────────────────

class TestStreamStats:
    def test_counts_tokens(self):
        stats = StreamStats()
        raw = _sse(
            {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        )
        stats.feed(raw)
        assert stats.tokens == 2

    def test_extracts_usage(self):
        stats = StreamStats()
        raw = _sse(
            {"choices": [{"delta": {"content": "Hi"}}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}},
        )
        stats.feed(raw)
        assert stats.usage["prompt_tokens"] == 5
        assert stats.usage["completion_tokens"] == 1

    def test_handles_fragmented_chunks(self):
        """Feed one byte at a time — buffer must reassemble correctly."""
        stats = StreamStats()
        raw = _sse(
            {"choices": [{"delta": {"content": "A"}}]},
            {"choices": [{"delta": {"content": "B"}}]},
        )
        for i in range(len(raw)):
            stats.feed(raw[i:i+1])
        assert stats.tokens == 2

    def test_ignores_non_data_lines(self):
        stats = StreamStats()
        raw = b"event: message\ndata: {\"choices\": [{\"delta\": {\"content\": \"x\"}}]}\n\n"
        stats.feed(raw)
        # The data line is still parsed (SSE spec: event: is a separate field)
        assert stats.tokens == 1

    def test_ignores_malformed_json(self):
        stats = StreamStats()
        raw = b"data: {broken\n\ndata: [DONE]\n\n"
        stats.feed(raw)
        assert stats.tokens == 0

    def test_reset(self):
        stats = StreamStats()
        stats.feed(_sse({"choices": [{"delta": {"content": "x"}}]}))
        assert stats.tokens == 1
        stats.reset()
        assert stats.tokens == 0
        assert stats.usage == {}

    def test_empty_content_not_counted(self):
        """Role-only first chunk (content="") must not count as a token."""
        stats = StreamStats()
        raw = _sse(
            {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"delta": {"content": "real"}}]},
        )
        stats.feed(raw)
        assert stats.tokens == 1

    def test_tool_call_delta_not_counted_as_token(self):
        stats = StreamStats()
        raw = _sse(
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "f"}}]}}]},
        )
        stats.feed(raw)
        assert stats.tokens == 0


# ── parse_completion ──────────────────────────────────────────────────────

class TestParseCompletion:
    def test_basic(self):
        data = {
            "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
        result = parse_completion(data)
        assert result["content"] == "hello"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 3

    def test_tool_calls(self):
        tc = [{"type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]
        data = {
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": tc}, "finish_reason": "tool_calls"}],
            "usage": {},
        }
        result = parse_completion(data)
        assert result["tool_calls"] == tc
        assert result["finish_reason"] == "tool_calls"

    def test_empty_choices(self):
        data = {"choices": [], "usage": {}}
        result = parse_completion(data)
        assert result["content"] == ""
        assert result["finish_reason"] == "stop"


# ── model_status ──────────────────────────────────────────────────────────

class TestModelStatus:
    def test_running(self):
        assert model_status(RunnerState.RUNNING) == {"value": "loaded"}

    def test_stopped(self):
        assert model_status(RunnerState.STOPPED) == {"value": "stopped"}

    def test_error(self):
        result = model_status(RunnerState.ERROR, "boom")
        assert result["value"] == "error"
        assert result["error_message"] == "boom"

    def test_ctx_none(self):
        assert model_status_for_ctx(None) == {"value": "uncached"}
