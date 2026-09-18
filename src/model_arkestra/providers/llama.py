"""LlamaProvider — local llama.cpp model, reached via its HTTP API.

Bound to one port (one model). Owns the HTTP chat/stream/embed/request logic.
Used by process and container backends. The "HTTP" part is a mechanism detail;
the provider's identity is that it serves a llama.cpp engine.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict

import aiohttp

from model_arkestra.providers.base import Provider, NotSupported
from model_arkestra.http_proxy import sse_events, parse_completion
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
})

_RETRIES = 12
_RETRY_SLEEP = 2.5


class LlamaProvider(Provider):
    capabilities = frozenset({"chat", "stream", "embed"})

    def __init__(self, model_name: str, port: int):
        self.model_name = model_name
        self.port = port
        self._base = f"http://127.0.0.1:{port}"

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

        last_err: Exception | None = None
        for attempt in range(_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self._base}/v1/chat/completions", json=payload, timeout=60
                    ) as resp:
                        if resp.status == 503:
                            await asyncio.sleep(_RETRY_SLEEP)
                            continue
                        if resp.status in (502, 504):
                            raise RunnerError(f"Server returned {resp.status}: upstream or gateway failure")
                        if resp.status != 200:
                            raise RunnerError(f"Server error: {resp.status}")
                        data = await resp.json()
                return parse_completion(data)
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                last_err = exc
                if attempt == _RETRIES - 1:
                    break
                await asyncio.sleep(_RETRY_SLEEP)
            except Exception as e:
                raise RunnerError(f"Request failed: {e}") from None
        raise RunnerError(f"Server not reachable after {_RETRIES} attempts") from last_err

    async def stream(self, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        p = dict(payload)
        messages = None
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
        start_time = time.monotonic()
        tokens_so_far: list[str] = []
        usage_info: Dict[str, Any] = {}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=stream_payload, timeout=60) as resp:
                    if resp.status != 200:
                        raise RunnerError(f"Server error: {resp.status}")
                    async for event in sse_events(resp.content):
                        if "token" in event:
                            tokens_so_far.append(event["token"])
                            yield {"token": event["token"]}
                        elif "reasoning" in event:
                            yield {"reasoning": event["reasoning"]}
                        elif "tool_call" in event:
                            yield {"tool_call": event["tool_call"]}
                        elif "finish_reason" in event:
                            yield {"finish_reason": event["finish_reason"]}
                        elif "usage" in event:
                            usage_info.update(event["usage"])
                        else:
                            elapsed = round(time.monotonic() - start_time, 2)
                            prompt_tok = usage_info.get("prompt_tokens", len(tokens_so_far))
                            completion_tok = usage_info.get("completion_tokens") or len(tokens_so_far)
                            usage_info.update({
                                "model": self.model_name,
                                "prompt_tokens": prompt_tok,
                                "completion_tokens": completion_tok,
                                "total_tokens": prompt_tok + completion_tok,
                                "time_seconds": elapsed,
                                "tokens_per_second": round(completion_tok / elapsed, 2) if elapsed > 0 else 0,
                            })
                            yield {"usage": usage_info}
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
