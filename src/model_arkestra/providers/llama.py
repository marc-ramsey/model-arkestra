"""LlamaProvider — local llama.cpp model, reached via its HTTP API.

Bound to one port (one model). Owns the HTTP chat/stream/embed/request logic.
Used by process and container backends. The "HTTP" part is a mechanism detail;
the provider's identity is that it serves a llama.cpp engine.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, Optional

import aiohttp

from model_arkestra.providers.base import Provider
from model_arkestra.http_proxy import parse_completion
from model_arkestra.types import RunnerError


# Request-time sampling params that may be forwarded to llama-server.
# Anything else is dropped — never forwarded (prevents bogus POST fields).
LLAMA_FIELDS = frozenset({
    "temperature", "top_p", "top_k", "repetition_penalty",
    "frequency_penalty", "presence_penalty", "stop", "seed",
    "mirostat", "mirostat_tau", "mirostat_eta", "grammar",
    "max_tokens", "min_tokens", "logit_bias",
    # OpenAI-style function calling — forwarded verbatim to llama-server.
    "tools", "tool_choice",
    # Ask llama-server to emit a final usage chunk (required for real
    # token counts; without it clients fall back to word estimates).
    "stream_options",
})

_RETRY_SLEEP = 2.5
# Wall-clock budget for the whole call, retries included. Counting attempts is
# not a bound: a server that accepts TCP but never answers burns one full
# per-attempt timeout each go-round (12 × 60s ≈ 12 min of a frozen request).
_RETRY_DEADLINE = 90.0
# Single POST budget. Bounded below _RETRY_DEADLINE so at least one retry fits.
_CHAT_TIMEOUT = 60.0
# Per-request timeout for chat/stream calls. sock_read bounds the gap between
# chunks — it is the only liveness guard. total is unbounded: long-context
# prefills and slow per-token rates on 27B-class models legitimately exceed
# any fixed total (a tighter one used to abort healthy streams mid-flight,
# surfacing as client "Connection error" + hang).
#
# A stream that is *silent* for minutes is not a dead peer: agentic workloads
# stall it on the client side while tools execute, and single tokens can take
# very long once KV cache runs deep. 120s fired exactly there (mid-reasoning,
# ~2 min in) killing streams that were fine. Bounded at 600s so a truly wedged
# llama-server still surfaces an error instead of hanging forever; dead peers
# are caught immediately by TCP reset, not this timeout.
# Floor is deliberately generous: a silent stream mid-flight usually means the
# client (agentic tool execution) or one very slow token, not a dead peer.
_DEFAULT_STREAM_SOCK_READ = 300.0


def stream_timeout(sock_read: Optional[float] = None) -> aiohttp.ClientTimeout:
    """Build the stream ClientTimeout, honoring a config override for sock_read."""
    if sock_read is None:
        sock_read = _DEFAULT_STREAM_SOCK_READ
    return aiohttp.ClientTimeout(total=None, sock_read=sock_read)


class LlamaProvider(Provider):
    capabilities = frozenset({"chat", "stream", "embed"})

    def __init__(self, model_name: str, port: int,
                 stream_sock_timeout: Optional[float] = None):
        self.model_name = model_name
        self.port = port
        self._base = f"http://127.0.0.1:{port}"
        self._stream_timeout = stream_timeout(stream_sock_timeout)

    # ── probe ────────────────────────────────────────────────────
    async def probe(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._base}/health", timeout=5) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    # ── chat / stream ────────────────────────────────────────────
    async def invoke_full(self, prompt: str = "", **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model_name}
        if "messages" in kwargs and isinstance(kwargs["messages"], (list, tuple)):
            payload["messages"] = list(kwargs.pop("messages"))
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]
        payload.update({k: v for k, v in kwargs.items() if k in LLAMA_FIELDS and v is not None})

        deadline = time.monotonic() + _RETRY_DEADLINE
        last_err: Exception | None = None
        while True:
            retryable = False
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self._base}/v1/chat/completions", json=payload,
                        timeout=_CHAT_TIMEOUT,
                    ) as resp:
                        if resp.status == 503:
                            # Server is up, model busy/unloaded — worth retrying,
                            # but keep the real cause for the final report.
                            last_err = RunnerError("Server returned 503 (model busy)")
                            retryable = True
                        elif resp.status in (502, 504):
                            raise RunnerError(f"Server returned {resp.status}: upstream or gateway failure")
                        elif resp.status != 200:
                            raise RunnerError(f"Server error: {resp.status}")
                        else:
                            data = await resp.json()
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                last_err = exc
                retryable = True
            except RunnerError:
                raise
            except Exception as e:
                raise RunnerError(f"Request failed: {e}") from None

            if not retryable:
                return parse_completion(data)

            if deadline - time.monotonic() <= _RETRY_SLEEP:
                break
            await asyncio.sleep(_RETRY_SLEEP)

        # asyncio.TimeoutError stringifies to '' — name the cause explicitly.
        why = str(last_err) or type(last_err).__name__
        raise RunnerError(f"{why} — giving up after {int(_RETRY_DEADLINE)}s of retries") from last_err

    async def stream(self, payload: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Yield raw SSE bytes from llama-server. No parsing — the caller
        (server._stream_chat) forwards them verbatim and uses StreamStats
        for side-channel counters."""
        p = dict(payload)
        if "messages" in p and isinstance(p["messages"], (list, tuple)):
            messages = list(p["messages"])
        else:
            prompt = p.pop("prompt", None)
            if not prompt:
                raise ValueError("Payload must contain 'prompt' or 'messages'")
            messages = [{"role": "user", "content": prompt}]

        stream_payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
        }
        stream_payload.update({k: v for k, v in p.items() if k in LLAMA_FIELDS and v is not None})

        url = f"{self._base}/v1/chat/completions"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=stream_payload, timeout=self._stream_timeout) as resp:
                    if resp.status != 200:
                        detail = (await resp.text())[:500]
                        raise RunnerError(f"Server error: {resp.status}: {detail}")
                    async for chunk in resp.content.iter_any():
                        yield chunk
            except Exception as e:
                raise RunnerError(f"Stream error: {e}")

    # ── embeddings (llama models that serve /v1/embeddings) ──────
    async def embed(self, text: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base}/v1/embeddings",
                json={"model": self.model_name, "input": text}, timeout=30,
            ) as resp:
                if resp.status != 200:
                    raise RunnerError(f"Embedding error: {resp.status}")
                return await resp.json()

    # ── generic passthrough ──────────────────────────────────────
    async def request(self, path: str, **kwargs: Any) -> Any:
        url = f"{self._base}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.request("POST", url, json=kwargs, timeout=15) as resp:
                if resp.status < 400:
                    try:
                        return await resp.json()
                    except Exception:
                        return await resp.read()
                raise RunnerError(f"Request failed with status {resp.status}")
