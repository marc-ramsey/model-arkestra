"""RemoteProvider — proxy inference to a cluster worker.

Engine-agnostic: the worker may run llama or ONNX behind its own OpenAI API.
The provider holds the worker base URL and admin key, and forwards chat/stream/
embed requests with the ``x-admin-key`` header when set.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict

import aiohttp

from model_arkestra.providers.base import Provider
from model_arkestra.providers.llama import LLAMA_FIELDS
from model_arkestra.http_proxy import sse_events, parse_completion


_RETRIES = 6
_RETRY_SLEEP = 2.5


class RemoteProvider(Provider):
    capabilities = frozenset({"chat", "stream", "embed"})

    def __init__(self, model_name: str, base_url: str, admin_key: str = ""):
        self.model_name = model_name
        self._base = base_url.rstrip("/")
        self._admin_key = admin_key or ""

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._admin_key:
            headers["x-admin-key"] = self._admin_key
        return headers

    async def probe(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._base}/health", timeout=5) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    # ── chat / stream ────────────────────────────────────────────
    async def invoke_full(self, prompt: str = "", **kwargs: Any) -> Dict[str, Any]:
        messages = None
        if "messages" in kwargs and isinstance(kwargs["messages"], (list, tuple)):
            messages = list(kwargs.pop("messages"))
        else:
            prompt = prompt or kwargs.pop("prompt", "")
            if not prompt:
                raise ValueError("Payload must contain 'prompt' or 'messages'")
            messages = [{"role": "user", "content": prompt}]

        payload: Dict[str, Any] = {"model": self.model_name, "messages": messages, "stream": False}
        payload.update({k: v for k, v in kwargs.items() if k in LLAMA_FIELDS and v is not None})

        url = f"{self._base}/v1/chat/completions"
        last_err: Exception | None = None
        last_status: int | None = None
        for attempt in range(_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=self._headers(), timeout=120) as resp:
                        if resp.status == 503:
                            last_status = 503
                            await asyncio.sleep(_RETRY_SLEEP)
                            continue
                        if resp.status != 200:
                            raise RuntimeError(f"Remote inference failed ({resp.status})")
                        data = await resp.json()
                return parse_completion(data)
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                last_err = exc
                if attempt == _RETRIES - 1:
                    break
                await asyncio.sleep(_RETRY_SLEEP)
        if last_status is not None:
            raise RuntimeError(f"Remote worker still returned {last_status} after {_RETRIES} attempts") from last_err
        raise RuntimeError(f"Remote server not reachable after {_RETRIES} attempts") from last_err

    async def stream(self, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        url = f"{self._base}/v1/chat/completions"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self._headers(), timeout=120) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    raise RuntimeError(f"Remote inference failed ({resp.status}): {detail}")
                async for event in sse_events(resp.content):
                    if "token" in event:
                        yield {"token": event["token"]}
                    elif "reasoning" in event:
                        yield {"reasoning": event["reasoning"]}
                    elif "tool_call" in event:
                        yield {"tool_call": event["tool_call"]}
                    elif "finish_reason" in event:
                        yield {"finish_reason": event["finish_reason"]}
                    elif "usage" in event:
                        yield {"usage": event["usage"]}

    # ── embeddings ───────────────────────────────────────────────
    async def embed(self, text: str) -> Dict[str, Any]:
        url = f"{self._base}/v1/embeddings"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json={"model": self.model_name, "input": text},
                headers=self._headers(), timeout=120,
            ) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    raise RuntimeError(f"Remote embedding failed ({resp.status}): {detail}")
                return await resp.json()

    # ── generic passthrough (proxied) ────────────────────────────
    async def request(self, path: str, **kwargs: Any) -> Any:
        url = f"{self._base}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=kwargs, headers=self._headers(), timeout=30) as resp:
                if resp.status < 400:
                    try:
                        return await resp.json()
                    except Exception:
                        return await resp.read()
                raise RuntimeError(f"Remote request failed with status {resp.status}")
