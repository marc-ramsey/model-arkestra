"""Shared HTTP helpers for SSE streaming and chat completion proxying.

Used by BaseRunner (local llama-server), RemoteModelRunner, and ArkestraServer.
"""
from __future__ import annotations
import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional
from model_arkestra.types import RunnerState


def _extract_content(msg: Dict[str, Any]) -> str:
    """Extract text content from an OpenAI-style message dict."""
    return msg.get("content") or msg.get("reasoning_content") or ""


# ── SSE parsing ───────────────────────────────────────────────────────────

class _SSEParser:
    """Stateless SSE parser that yields typed events.

    Usage::

        async for event in sse_events(raw_stream):
            if "token" in event:
                yield event["token"]
            elif "usage" in event:
                emit_usage(event["usage"])
            elif event.get("done"):
                break
    """

    @staticmethod
    def parse_line(line: str) -> Optional[Dict[str, Any]]:
        """Parse a single SSE data line. Returns None for non-data lines."""
        if not line.startswith("data:"):
            return None
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            return {"done": True}

        try:
            chunk = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            return None

        choices = chunk.get("choices", [])
        delta = choices[0].get("delta", {}) if choices else {}
        content = delta.get("content")
        if content:
            return {"token": content}
        usage = chunk.get("usage")
        if usage:
            return {"usage": usage}
        return None


async def sse_events(raw_lines) -> AsyncIterator[Dict[str, Any]]:
    """Convert raw HTTP lines into typed SSE events.

    Yields dicts with keys: ``"token"``, ``"usage"``, or ``"done"``.
    """
    async for raw in raw_lines:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = text.strip()
        if not line:
            continue
        event = _SSEParser.parse_line(line)
        if event is not None:
            yield event


# ── Chat completion extraction ───────────────────────────────────────────

def model_status(state: RunnerState, error_message: str | None = None) -> Dict[str, str]:
    """Map a RunnerState enum value to the structured dict that Open WebUI expects.

    WebUI auto-loads models whose status is ``{"value": "loaded"}``.
    All other states map literally for display fidelity.
    """
    state_map = {
        RunnerState.LOADING:  {"value": "loading"},
        RunnerState.RUNNING:  {"value": "loaded"},
        RunnerState.STOPPED:  {"value": "sleeping"},
        RunnerState.STOPPING: {"value": "sleeping"},
        RunnerState.UNCACHED: {"value": "unloaded"},
        RunnerState.DOWNLOADING: {"value": "downloading"},
    }
    entry = state_map.get(state, {})
    if state == RunnerState.ERROR:
        return {"value": "error", "error_message": error_message or "unknown error"}
    return entry


def model_status_for_ctx(ctx) -> Dict[str, str]:
    """Helper for call sites that hold a _ModelContext or None."""
    return {"value": "stopped"} if ctx is None else model_status(ctx.state, ctx.last_error)


def parse_completion(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract {content, usage} from an OpenAI non-streaming response dict."""
    choices = data.get("choices", [])
    msg = choices[0].get("message", {}) if choices else {}
    content = _extract_content(msg) or ""
    return {
        "content": content,
        "usage": data.get("usage", {
            "model": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "time_seconds": 0,
        }),
    }


# ── HTTP POST helpers ─────────────────────────────────────────────────────

async def post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str] | None = None,
                    timeout: float = 120.0) -> tuple[int, bytes]:
    """Post JSON and return (status_code, response_bytes).

    Caller decodes / parses the body as needed.
    """
    import aiohttp

    hdrs = headers or {"Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=hdrs, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            body = await resp.read()
            return resp.status, body


async def post_json_retried(url: str, payload: Dict[str, Any], headers: Dict[str, str] | None = None,
                            max_retries: int = 6, retry_delay: float = 2.5) -> tuple[int, bytes]:
    """Like ``post_json`` but retries on connection errors and timeouts."""
    import aiohttp

    hdrs = headers or {"Content-Type": "application/json"}
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await post_json(url, payload, hdrs)
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError, asyncio.CancelledError) as exc:
            last_err = exc
            if attempt == max_retries - 1:
                break
            await asyncio.sleep(retry_delay)
    raise RuntimeError(f"Remote server not reachable after {max_retries} retries") from last_err
