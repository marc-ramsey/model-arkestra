"""Test fixtures for model-arkestra.

Central fixture system — every integration test module should rely on these:

* **``mr``** (module-scoped) — shared ``ModelArkestra`` instance. Config loaded
  once, runner class maps built once, port allocator shared across all tests in
  the module.

* **``_cleanup_after_test``** (function-scoped autouse) — calls ``mr.stop_all()``
  after every test so each method sees a clean slate (empty models dict, fresh
  port counter, no lingering watchers).

* **``_cleanup_ports``** (module-scoped autouse) — safety net that kills any
  lingering processes on the configured port range (*before* and *after* each
  module).  The range is read from ``test-config.yaml``:
  ``models-start-port`` through ``models-start-port + model-ports``.

Helper functions (**``shutdown_runner``**, ``_kill_port``, ``_kill_runner``) are
exported for modules that need low-level cleanup inside their own fixtures.
"""

from __future__ import annotations
import os
import signal
import subprocess
from typing import Any, Dict, Optional

import pytest
import yaml

from model_arkestra.arkestra import ModelArkestra

# ── Port range (computed once from the test config at module load) ───────────

_test_cfg_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test-config.yaml"
)
with open(_test_cfg_path) as _f:
    _cfg: dict = yaml.safe_load(_f)

_START_PORT: int = int(_cfg.get("models-start-port", 18000))
_NUM_PORTS:   int = int(_cfg.get("model-ports", 32))
_CLEANUP_PORTS = tuple(range(_START_PORT, _START_PORT + _NUM_PORTS))


# ── Port / process helpers (reusable by test modules) ────────────────────────


def _kill_port(port: int) -> None:
    """Kill any process listening on *port* using ``lsof``."""
    result = subprocess.run(
        ["lsof", "-ti:", str(port)], capture_output=True, text=True
    )
    for pid in result.stdout.strip().split():
        if pid:
            try:
                os.kill(int(pid), 9)
            except OSError:
                pass


def _kill_runner(runner: Any) -> None:
    """Kill all models on a ``BaseModelRunner`` using synchronous OS signals.

    This is the ONLY reliable way to shut down runners from sync teardown code
    (e.g. module-scoped fixtures where asyncio may not be available).
    """
    if not (models := getattr(runner, "_models", None)):
        return
    for key, ctx in list(models.items()):
        proc = getattr(ctx, "process", None)
        if proc is not None and proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGHUP)
            except (ProcessLookupError, OSError):
                pass
        cid = getattr(ctx, "container_id", None)
        if cid:
            for cmd in (["podman", "kill", cid], ["docker", "kill", cid]):
                try:
                    subprocess.run(cmd, capture_output=True, timeout=5)
                except Exception:
                    pass


def shutdown_runner(runner: Any) -> None:
    """Synchronously kill all models across every runner in a ``ModelArkestra``."""
    if not hasattr(runner, "_runners"):
        return
    for r in runner._runners.values():
        _kill_runner(r)


# ── Shared infrastructure (one ModelArkestra per module) ────────────────────


@pytest.fixture(scope="module")
def mr() -> ModelArkestra:
    """Shared ``ModelArkestra`` instance for all tests in a module.

    The same object is yielded to every test method and class within the module.
    Port allocation and runner class maps are shared — but each test should start
    its own models and let :fixture:`_cleanup_after_test` handle teardown.

    Teardown kills any lingering processes so nothing leaks to other modules.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test-config.yaml"
    )
    runner = ModelArkestra(config_path, ready_timeout=30, warmup_delay=10)
    yield runner
    # Sync teardown — asyncio may not be available at module-scope cleanup time.
    shutdown_runner(runner)


# ── Per-test isolation (ensures clean state between every test method) ──────


@pytest.fixture(autouse=True)
async def _cleanup_after_test(mr: ModelArkestra) -> None:
    """Guarantee that models are stopped after each test, regardless of outcome.

    Calls ``mr.stop_all()`` which resets ``_models``, clears watchers, and resets
    the port counter — giving every test method a fresh slate.  A ``RuntimeError``
    guard handles the rare case where the event loop is already closed during
    teardown.
    """
    yield
    if mr._runners:  # only bother if something was actually started
        try:
            await mr.stop_all()
        except RuntimeError:
            pass  # event loop may be closed — nothing we can do


# ── Port-level safety net (kills lingering listeners before/after a module) ─


@pytest.fixture(autouse=True, scope="module")
def _cleanup_ports() -> None:
    """Kill any process on the configured port range before and after each module."""
    for port in _CLEANUP_PORTS:
        _kill_port(port)
    yield
    for port in _CLEANUP_PORTS:
        _kill_port(port)
