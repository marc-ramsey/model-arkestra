"""ProcessRunner stop-path tests.

Regression: if the stop task is cancelled while waiting out the SIGHUP
grace period (server shutdown / Ctrl+C race), the process group must
still receive SIGKILL — otherwise llama-server survives holding the
port and VRAM until someone kills it by hand.
"""
from __future__ import annotations
import asyncio
import os
import signal
from unittest.mock import MagicMock

import pytest

from model_arkestra.models.model import _Model
from model_arkestra.process import ProcessRunner


@pytest.mark.asyncio
async def test_stop_kills_process_group_even_when_cancelled():
    # sleep ignores SIGHUP (trap ""), mimicking llama-server mid-generation
    proc = await asyncio.create_subprocess_exec(
        "sh", "-c", 'trap "" HUP; sleep 300',
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    ctx = _Model("test-stop-cancel")
    ctx.process = proc

    runner = ProcessRunner(MagicMock())
    runner.shutdown_timeout = 10.0
    runner.arkestra = None

    task = asyncio.create_task(runner._stop_model_process(ctx))
    await asyncio.sleep(0.3)  # let it enter the SIGHUP grace wait
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # SIGKILL must have landed — poll for reaping
    for _ in range(50):
        if proc.returncode is not None:
            break
        await asyncio.sleep(0.1)
    assert proc.returncode is not None, (
        "process survived a cancelled stop — SIGHUP grace wait "
        "swallowed CancelledError without escalating to SIGKILL"
    )
