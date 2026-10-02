"""Shared HTTP helpers for SSE streaming and chat completion proxying.

Used by LlamaProvider, RemoteProvider, and ArkestraServer.
"""
from __future__ import annotations
import json
from typing import Any, Dict
from model_arkestra.types import RunnerState


def _extract_content(msg: Dict[str, Any]) -> str:
    """Extract text content from an OpenAI-style message dict."""
    return msg.get("content") or msg.get("reasoning_content") or ""


# ── Stream stats (side-channel) ───────────────────────────────────────────

class StreamStats:
    """Side-channel stats for a streaming SSE response.

    Feed raw SSE bytes; get token/usage counts without gating the data path.
    The proxy forwards bytes verbatim — this class only observes.

    Usage::

        stats = StreamStats()
        async for chunk in resp.content.iter_any():
            stats.feed(chunk)
            yield chunk  # data path — never blocked by stats
        print(stats.tokens, stats.usage)
    """

    def __init__(self):
        self.tokens: int = 0
        self.usage: Dict[str, Any] = {}
        # Bytes, not str: avoids re-concatenating a growing string per chunk
        # (O(n^2) across the stream). Only complete lines are decoded.
        self._buf: bytes = b""

    def feed(self, chunk: bytes) -> None:
        """Process a raw byte chunk. Updates tokens/usage counters."""
        self._buf += chunk
        while b"\n" in self._buf:
            line_bytes, self._buf = self._buf.split(b"\n", 1)
            # Decode each complete line once — the buffer holds at most one
            # partial line between feeds, so growth stays O(line), not O(stream).
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue
            choices = obj.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    self.tokens += 1
            usage = obj.get("usage")
            if usage:
                self.usage.update(usage)

    def reset(self) -> None:
        self.tokens = 0
        self.usage = {}
        self._buf = b""


# ── Chat completion extraction ───────────────────────────────────────────

def model_status(state: RunnerState, error_message: str | None = None) -> Dict[str, str]:
    """Map a RunnerState enum value to the structured dict that Open WebUI expects.

    WebUI auto-loads models whose status is ``{"value": "loaded"}``.
    All other states map literally for display fidelity.
    """
    state_map = {
        RunnerState.LOADING:  {"value": "loading"},
        RunnerState.RUNNING:  {"value": "loaded"},
        RunnerState.STOPPED:  {"value": "stopped"},
        RunnerState.STOPPING: {"value": "stopping"},
        RunnerState.UNCACHED: {"value": "uncached"},
        RunnerState.DOWNLOADING: {"value": "downloading"},
    }
    entry = state_map.get(state, {})
    if state == RunnerState.ERROR:
        return {"value": "error", "error_message": error_message or "unknown error"}
    return entry


def model_status_for_ctx(ctx) -> Dict[str, str]:
    """Helper for call sites that hold a _Model or None."""
    if ctx is None:
        return {"value": "uncached"}
    return model_status(ctx.state, ctx.last_error)


def parse_completion(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract {content, tool_calls, finish_reason, usage} from an OpenAI
    non-streaming response dict."""
    choices = data.get("choices", [])
    msg = choices[0].get("message", {}) if choices else {}
    content = _extract_content(msg) or ""
    return {
        "content": content,
        "tool_calls": msg.get("tool_calls"),
        "finish_reason": (choices[0].get("finish_reason") if choices else None) or "stop",
        "usage": data.get("usage", {
            "model": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "time_seconds": 0,
        }),
    }
