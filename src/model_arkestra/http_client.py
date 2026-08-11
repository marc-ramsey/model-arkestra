"""Lightweight HTTP client wrapper around aiohttp for model_runner.

Encapsulates session management and request/response patterns used by
BaseModelRunner methods (ainvoke, astream, request, health checks).
Subclassing or patching this class is the supported way to test runner
HTTP paths without dealing with aiohttp's async context manager internals.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import aiohttp


class _RawResponseWrapper:
    """Async context manager that closes both response and session on exit."""

    def __init__(self, session: aiohttp.ClientSession,
                 resp: aiohttp.ClientResponse) -> None:
        self._session = session
        self._resp = resp

    async def __aenter__(self) -> aiohttp.ClientResponse:
        return self._resp

    async def __aexit__(self, *exc) -> None:
        await self._resp.close()
        await self._session.close()


class ModelHttpClient:
    """Thin wrapper around ``aiohttp.ClientSession`` for model_runner use cases."""

    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout

    async def get_json(self, url: str) -> Dict[str, Any]:
        """GET a URL and return parsed JSON body."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.timeout) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def post_json(self, url: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
        """POST to a URL and return parsed JSON body."""
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=json_body, timeout=self.timeout) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def post_raw(self, url: str, json_body: Optional[Dict[str, Any]] = None):
        """POST and return an async context manager for the raw response.

        Usage:
            async with client.post_raw(url) as resp:
                data = await resp.json()
        """
        session = aiohttp.ClientSession()
        try:
            resp = await session.post(url, json=json_body, timeout=self.timeout)
            return _RawResponseWrapper(session, resp)
        except Exception:
            await session.close()
            raise

    async def stream_sse(self, url: str, json_body: Dict[str, Any]):
        """Iterate SSE ``data:`` lines from a POST endpoint.

        Yields raw string lines (without the ``data:`` prefix).
        Raises ``RunnerError`` on non-200 status.
        """
        from model_arkestra.types import RunnerError

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=json_body, timeout=60) as resp:
                if resp.status != 200:
                    raise RunnerError(f"Server error: {resp.status}")
                while True:
                    raw = await resp.content.readline()
                    if not raw:
                        break
                    text = raw.decode("utf-8").strip()
                    if not text:
                        continue
                    event_data = text[5:].strip() if text.startswith("data:") else text
                    yield event_data

    def close(self) -> None:
        """No-op — sessions are session-scoped in aiohttp."""
        pass
