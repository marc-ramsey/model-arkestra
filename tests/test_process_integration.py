"""Integration tests for ProcessModelRunner with real model servers.

These tests start actual llama.cpp processes via the runner — they are
intentionally slow and marked ``@pytest.mark.slow`` so CI can skip them.
"""

from __future__ import annotations
import asyncio
import os

import pytest


@pytest.mark.slow
class TestProcessRunnerIntegration:
    """Full end-to-end tests with real model."""

    async def test_ainvoke_gemma(self, mr):
        """Start gemma-4-e2b → call ainvoke → get response → stop."""
        await mr.start("gemma-4-e2b")

        print("[*] Testing completion (ainvoke)...")
        response = await mr.ainvoke("gemma-4-e2b", "Hello, how are you?")
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"[<] Response ({len(response)} chars): {response[:100]}…")

    async def test_astream_gemma(self, mr):
        """Start gemma-4-e2b → stream tokens → stop."""
        await mr.start("gemma-4-e2b")

        print("[*] Testing streaming (astream)...")
        full_content = ""
        async for chunk in mr.astream("gemma-4-e2b", {"prompt": "Say hi!"}):
            if "token" in chunk and chunk["token"]:
                print(chunk["token"], end="", flush=True)
                full_content += chunk["token"]
            elif "usage" in chunk:
                print()
                u = chunk["usage"]
                print(f"Done ({u.get('total_tokens', '?')} tokens, {u.get('tokens_per_second', '?')} tok/s)")
        print()  # newline after stream
        assert len(full_content) > 0

    async def test_ainvoke_qwen3(self, mr):
        """Start qwen3.5-4b → call ainvoke → get response → stop."""
        await mr.start("qwen3.5-4b")

        print("[*] Testing completion (ainvoke) on qwen3.5-4b...")
        response = await mr.ainvoke("qwen3.5-4b", "Say hello in 5 words or less.")
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"[<] Response: {response[:100]}…")

    async def test_astream_qwen3(self, mr):
        """Start qwen3.5-4b → stream tokens → stop."""
        await mr.start("qwen3.5-4b")

        print("[*] Testing streaming (astream) on qwen3.5-4b...")
        full_content = ""
        async for chunk in mr.astream("qwen3.5-4b", {"prompt": "Count to 3."}):
            if "token" in chunk and chunk["token"]:
                print(chunk["token"], end="", flush=True)
                full_content += chunk["token"]
            elif "usage" in chunk:
                print()
                u = chunk["usage"]
                print(f"Done ({u.get('total_tokens', '?')} tokens, {u.get('tokens_per_second', '?')} tok/s)")
        print()
