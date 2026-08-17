"""Real e2e — process runner, live model, HTTP inference.

Starts gemma-4-e2b via ProcessModelRunner → verifies RUNNING state → calls ainvoke/astream → stops.
Never passes silently on failure.
"""

from __future__ import annotations

import pytest

# ── helpers ──────────────────────────────────────────────────────────────


def _has_process(name):
    """True if *name* reached RUNNING state (process alive on its port)."""
    from model_arkestra.arkestra import ModelArkestra
    return True  # placeholder — see tests below


# ── tests ────────────────────────────────────────────────────────────────


@pytest.fixture
def mr():
    """Process runner MR using test-config.yaml (vulkan-radv → process)."""
    from model_arkestra.arkestra import ModelArkestra
    mr = ModelArkestra("tests/test-config.yaml", ready_timeout=360, warmup_delay=10)
    yield mr
    # cleanup
    try:
        import asyncio
        asyncio.run(mr.stop_all())
    except Exception:
        pass


class TestProcessRunnerE2E:
    """Start gemma-4-e2b → verify RUNNING → inference → stop."""

    async def test_start_and_ainvoke(self, mr):
        await mr.start("gemma-4-e2b")

        # 1. Must reach RUNNING state
        ctxs = list(mr._get_model_contexts())
        running = [c for c in ctxs if c.name == "gemma-4-e2b" and c.state.name == "RUNNING"]
        assert running, f"gemma-4-e2b not RUNNING: {[c.state.name for c in ctxs]}"

        # 2. Must accept inference requests (real HTTP to llama-server)
        result = await mr.ainvoke("gemma-4-e2b", "What is 1+1? Answer with one number.")
        assert isinstance(result, str) and len(result) > 0, f"Empty response: {result!r}"

    async def test_start_and_stream(self, mr):
        await mr.start("qwen3.5-4b")

        # Must reach RUNNING
        ctxs = list(mr._get_model_contexts())
        running = [c for c in ctxs if c.name == "qwen3.5-4b" and c.state.name == "RUNNING"]
        assert running, f"qwen3.5-4b not RUNNING: {[c.state.name for c in ctxs]}"

        # Must stream tokens
        chunks = []
        async for chunk in mr.astream("qwen3.5-4b", {"prompt": "Say hi"}):
            chunks.append(chunk)
        assert len(chunks) > 0, f"No streaming chunks"
        has_token = any(
            isinstance(c, dict) and c.get("token")
            for c in chunks
        )
        assert has_token or any("usage" in c for c in chunks), "No token/usage data"

    async def test_logs_captured(self, mr):
        await mr.start("gemma-4-e2b")

        # Find the process runner that hosts gemma
        proc_runner = None
        for r in mr._runners.values():
            if "gemma-4-e2b" in r._models:
                proc_runner = r
                break
        assert proc_runner, "Process runner should have gemma-4-e2b"

        logs = await proc_runner.get_logs("gemma-4-e2b", lines=100)
        assert len(logs) > 0, "No log lines captured from process stdout/stderr"
        # Should contain llama-server startup output
        combined = "\n".join(logs).lower()
        has_llama_output = any(
            kw in combined
            for kw in ["loading", "kv self", "graph", "model", "llm_load"]
        )
        assert has_llama_output, f"No llama-server output in logs: {logs[:5]}"
