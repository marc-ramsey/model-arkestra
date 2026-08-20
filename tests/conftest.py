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
import asyncio
import os
import signal
import subprocess
import time
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

# Also cover proxy test ports (e.g. 20100)
_EXTRA_PORTS = tuple(range(20090, 20110))

# ── Port / process helpers (reusable by test modules) ────────────────────────


def graceful_server_teardown(fixture_dict_or_proxy) -> None:
    """Gracefully shut down a live server fixture and release all test ports.

    Releases ports **18000–18009** — the full range live fixtures may use
    (server port + child model ports).

    Usage::

        yield {"server": proxy, "client": client}
        graceful_server_teardown(live_server)

    Or::

        yield proxy
        graceful_server_teardown(proxy)

    Waits up to **20 seconds** at 0.5 s intervals for ports to become free.
    Only after the full wait is exhausted is *kill -9* used as a last resort.
    """
    # Accept either the fixture dict or the raw proxy object
    if isinstance(fixture_dict_or_proxy, dict):
        proxy = fixture_dict_or_proxy["server"]
    else:
        proxy = fixture_dict_or_proxy

    server_obj = getattr(proxy, "_server", None)
    arkestra = getattr(proxy, "_arkestra", None)

    # 1. Stop all running models gracefully (if available)
    if arkestra is not None and hasattr(arkestra, "shutdown"):
        try:
            asyncio.run(arkestra.shutdown())
        except Exception:
            pass

    # 2. Signal uvicorn to shut down
    if server_obj is not None:
        server_obj.should_exit = True

    ports = tuple(range(18000, 18010))

    # 3. Wait for all ports release: 0.5 s × 40 iterations = 20 seconds max
    poll_interval = 0.5
    max_polls = 40  # 20 seconds total

    for _ in range(max_polls):
        all_free = True
        for p in ports:
            result = subprocess.run(
                ["lsof", f"-ti:{p}"], capture_output=True, text=True
            )
            if result.stdout.strip():
                all_free = False
                break
        if all_free:
            return  # all ports free — done
        time.sleep(poll_interval)

    # Port(s) still occupied after 20 s → force kill as last resort
    for p in ports:
        result = subprocess.run(
            ["lsof", f"-ti:{p}"], capture_output=True, text=True
        )
        for pid in result.stdout.strip().split():
            if pid:
                try:
                    os.kill(int(pid), 9)
                except OSError:
                    pass


def _kill_port(port: int) -> None:
    """Kill any process listening on *port* using ``lsof``."""
    result = subprocess.run(
        ["lsof", f"-ti:{port}"], capture_output=True, text=True
    )
    for pid in result.stdout.strip().split():
        if pid:
            try:
                os.kill(int(pid), 9)
            except OSError:
                pass


def _wait_for_port_free(port: int, timeout: float = 10.0) -> bool:
    """Block until no process is listening on *port*, or *timeout* seconds elapse."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        result = subprocess.run(
            ["lsof", f"-ti:{port}"], capture_output=True, text=True
        )
        pids = [p for p in result.stdout.strip().split() if p]
        if not pids:
            return True
        # Still alive — SIGKILL anyone who won't die
        for pid in pids:
            try:
                os.kill(int(pid), 9)
            except OSError:
                pass
        time.sleep(0.2)
    return False


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
    runner = ModelArkestra(config_path, ready_timeout=360, warmup_delay=10)
    yield runner
    # Sync teardown — asyncio may not be available at module-scope cleanup time.
    shutdown_runner(runner)


# ── Per-test isolation (ensures clean state between every test method) ──────


@pytest.fixture(autouse=True)
async def _cleanup_after_test(mr: ModelArkestra) -> None:
    """Guarantee that models are stopped after each test, regardless of outcome.

    Calls ``mr.shutdown()`` which resets ``_models``, clears watchers, and resets
    the port counter — giving every test method a fresh slate.  A ``RuntimeError``
    guard handles the rare case where the event loop is already closed during
    teardown.
    """
    yield
    if mr._runners:  # only bother if something was actually started
        try:
            await mr.shutdown()  # clears _models, _runners, resets port counter
            # shutdown() stops models but does not remove containers — clean up any leftovers.
            for cid in (subprocess.run(
                ["podman", "ps", "-a", "--filter", "name=llm-",
                 "--format", "{{.ID}}"],
                capture_output=True, text=True,
            ).stdout.strip().split() + subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=llm-",
                 "--format", "{{.ID}}"],
                capture_output=True, text=True,
            ).stdout.strip().split()):
                if cid:
                    subprocess.run(["podman", "rm", "-f", cid], capture_output=True, timeout=5)
                    subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=5)
            # Wait until every port is actually free — prevents next test from hitting
            # a stale listener that survived shutdown/kill (rootless pasta networking,
            # slow process teardown, or leftover containers from other modules).
            for port in list(_CLEANUP_PORTS) + list(_EXTRA_PORTS):
                _wait_for_port_free(port, timeout=5.0)
        except RuntimeError:
            pass  # event loop may be closed — nothing we can do


# ── Port-level safety net (kills lingering listeners before/after a module) ─


@pytest.fixture(autouse=True, scope="module")
def _cleanup_ports() -> None:
    """Kill any process on the configured port range before and after each module."""
    for port in _CLEANUP_PORTS:
        _kill_port(port)
    for port in _EXTRA_PORTS:
        _kill_port(port)
    # Wait so killed listeners release their file descriptors.
    time.sleep(0.5)
    yield
    # Kill again after module — some processes survive the first pass.
    time.sleep(0.2)
    for port in _CLEANUP_PORTS:
        _kill_port(port)
    for port in _EXTRA_PORTS:
        _kill_port(port)


# ── Podman test resource tracker (guaranteed teardown via fixture) ───────────


class _PodmanCleanupTracker:
    """Tracks podman containers, watcher tasks, and ports created during a test.

    All tracked resources are cleaned up in fixture teardown — even if the test
    fails an assertion or crashes.  Each test class that needs it instantiates
    one tracker and calls ``track_*`` methods as resources are created.
    """

    def __init__(self) -> None:
        self._containers: list[str] = []
        self._tasks: list[asyncio.Task] = []  # type: ignore[type-arg]
        self._ports: set[int] = set()

    def track_container(self, cid: str) -> None:
        """Mark a podman container for removal in teardown."""
        self._containers.append(cid)

    def track_task(self, task: asyncio.Task) -> None:
        """Mark an async watcher task for cancellation in teardown."""
        self._tasks.append(task)

    def track_port(self, port: int) -> None:
        """Mark a port for killing in teardown."""
        self._ports.add(port)


@pytest.fixture()
async def podman_cleanup() -> _PodmanCleanupTracker:
    """Fixture that tracks podman containers/tasks/ports with guaranteed teardown.

    Import and use inside test methods — it is not autouse so only tests that
    need it pay the cost:

        async def test_something(self, podman_cleanup):
            cleanup = podman_cleanup
            cid = await _start_container(...)
            cleanup.track_container(cid)

    Teardown cancels watcher tasks (awaited), removes containers, and kills ports
    — regardless of whether the test passed or raised an assertion error.
    """
    tracker = _PodmanCleanupTracker()
    yield tracker

    # ── Teardown (always runs, even on assertion failure) ───────────────

    # 1. Cancel watcher tasks first — they may spawn new containers we need
    for task in tracker._tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # 2. Remove tracked containers
    for cid in tracker._containers:
        subprocess.run(["podman", "rm", "-f", cid], capture_output=True, timeout=10)

    # 3. Kill tracked ports
    for port in tracker._ports:
        _kill_port(port)
    # Brief wait so listeners release their file descriptors.
    time.sleep(0.2)

