"""Integration tests for Podman restart logic using **real** podman containers."""

from __future__ import annotations
import asyncio
import os
import subprocess
import uuid
from unittest.mock import MagicMock

import pytest

from model_arkestra.podman import PodmanModelRunner
from model_arkestra.types import RunnerState, _ModelContext


# ── Helpers ────────────────────────────────────────────────────────────────────

def _podman(*args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["podman"] + list(args),
        capture_output=True, text=True, timeout=30
    )
    return result


def _make_runner():
    mock_cm = MagicMock()
    mock_cm.get_model.return_value = {"image": "alpine:latest", "container_port": 80}
    return PodmanModelRunner(
        config_manager=mock_cm,
        restart_delay=0.05,
        restart_limit=3,
        ready_timeout=60,
        ready_poll_ms=100,
    )


def _kill_port(port: int) -> None:
    """Kill process on *port* and remove any containers mapped to it."""
    # Kill via lsof (most common)
    result = subprocess.run(["lsof", "-ti:", str(port)], capture_output=True, text=True)
    for pid in result.stdout.strip().split():
        if pid:
            try:
                os.kill(int(pid), 9)
            except OSError:
                pass
    # Also kill via ss — catches rootless podman pasta processes that lsof misses
    import re as _re
    result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if f":{port}" in line and "pasta" in line:
            m = _re.search(r"pid=(\d+)", line)
            if m:
                try:
                    os.kill(int(m.group(1)), 9)
                except OSError:
                    pass
    # Remove any containers mapped to this port
    result = subprocess.run(
        ["podman", "ps", "-a", "--format", "{{.ID}}"],
        capture_output=True, text=True,
    )
    for cid in result.stdout.strip().split():
        if not cid:
            continue
        portmap = subprocess.run(
            ["podman", "inspect", "--format", "{{json .NetworkSettings.Ports}}", cid],
            capture_output=True, text=True,
        ).stdout.strip()
        if f'"{port}/tcp"' in portmap or str(port) in portmap:
            subprocess.run(["podman", "rm", "-f", cid], capture_output=True, timeout=10)


# ── Module fixture: ensure alpine image is available ─────────────────────────

@pytest.fixture(scope="module")
def alpine_image():
    """Pull alpine:latest once per test module (slow, shared)."""
    subprocess.run(["podman", "pull", "alpine:latest"], capture_output=True, timeout=120)
    return "alpine:latest"


# ── Test 1: Container starts and can be externally killed ─────────────────────

class TestContainerStartAndDetectExit:
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_container_starts_and_can_be_killed(self, alpine_image):
        runner = _make_runner()
        port = 18000

        ctx = _ModelContext("alpine-sleeper", port)
        runner._models["alpine-sleeper"] = ctx
        ctx.state = RunnerState.RUNNING

        proc = await asyncio.create_subprocess_shell(
            f"podman run --rm -d -p {port}:80 alpine:latest sh -c 'while true; do sleep 1; done'",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        cid = stdout.decode().strip()

        if not cid:
            pytest.fail(f"Failed to start container: {stderr.decode()}")

        ctx.container_id = cid
        print(f"[+] Started container: {cid[:12]}")

        insp = _podman("inspect", "--format", "{{.State.Status}}", cid)
        assert insp.stdout.strip() == "running"

        _podman("kill", cid)
        await asyncio.sleep(0.5)

        result = _podman("inspect", "--format", "{{.State.Status}}", cid)
        assert result.returncode != 0
        print("[-] Container killed and verified as stopped")


# ── Test 2: _watch_container detects exit after external kill ────────────────

class TestWatchContainerDetectsExit:
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_watch_detects_exit(self, alpine_image, podman_cleanup):
        cleanup = podman_cleanup
        runner = _make_runner()
        port = 18001
        name = f"alpine-watcher-{uuid.uuid4().hex[:8]}"

        ctx = _ModelContext("alpine-exit", port)
        runner._models["alpine-exit"] = ctx
        ctx.state = RunnerState.RUNNING

        proc = await asyncio.create_subprocess_shell(
            f"podman run -d --name {name} -p {port}:80 alpine:latest sh -c 'while true; do sleep 1; done'",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        cid = stdout.decode().strip()

        if not cid:
            pytest.fail(f"Failed to start container: {proc.stderr}")

        ctx.container_id = cid
        print(f"[+] Started container: {cid[:12]}")
        cleanup.track_container(cid)

        task = asyncio.create_task(runner._watch_container("alpine-exit", ctx))
        await asyncio.sleep(0.5)
        _podman("kill", cid)

        # Wait for: 2s poll + restart delay ≈ 2.1s
        await asyncio.sleep(3.0)

        assert ctx.restart_count >= 1, \
            f"Expected restart detected, got count={ctx.restart_count}"
        print(f"[+] Watcher detected exit → {ctx.restart_count} restart(s)")

        cleanup.track_task(task)
        cleanup.track_port(port)
        await runner.stop()


# ── Test 3: stop() prevents restart after crash ───────────────────────────────

class TestStopPreventsRestart:
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_stop_prevents_restart(self, alpine_image, podman_cleanup):
        cleanup = podman_cleanup
        runner = _make_runner()
        port = 18002
        name = f"alpine-stop-{uuid.uuid4().hex[:8]}"

        ctx = _ModelContext("alpine-stop-test", port)
        runner._models["alpine-stop-test"] = ctx
        ctx.state = RunnerState.RUNNING

        proc = await asyncio.create_subprocess_shell(
            f"podman run -d --name {name} -p {port}:80 alpine:latest sh -c 'while true; do sleep 1; done'",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        cid = stdout.decode().strip()

        if not cid:
            pytest.fail(f"Failed to start container: {proc.stderr}")

        ctx.container_id = cid
        print(f"[+] Started container: {cid[:12]}")
        cleanup.track_container(cid)

        task = asyncio.create_task(runner._watch_container("alpine-stop-test", ctx))
        await asyncio.sleep(0.3)
        _podman("kill", cid)

        # MUST be awaited — stop() is async
        await runner.stop()
        await asyncio.sleep(3.0)  # wait for next poll cycle (2s)

        cleanup.track_task(task)
        cleanup.track_port(port)

        assert ctx.state == RunnerState.STOPPED, \
            f"Expected STOPPED, got {ctx.state}"
        assert ctx.restart_count == 0, \
            f"stop() should block restart, but count={ctx.restart_count}"
        print("[+] stop() correctly prevented restart after crash")


# ── Test 4: Full runner lifecycle via start()/stop_all() ─────────────────────

class TestFullLifecycle:
    """Verify the full lifecycle API with a container that responds to /health."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_start_and_stop(self, alpine_image):
        # Use a port outside the 18000-18003 range to avoid conflicts with
        # podman rootless pasta networking (which lsof can't see).
        port = 28003
        _kill_port(port)

        runner = PodmanModelRunner(
            MagicMock(get_model=lambda n, **kw: {"image": "alpine:latest", "container_port": 80}),
            restart_delay=0.5,
        )

        # Patch _start_model_process to launch nginx with JSON /health endpoint
        async def patched_start(self_inner, ctx_inner, model_data):
            conf = (
                'server { listen 80; location /health '
                '{ default_type application/json; return 200 '
                '\'{"status":"ok"}\' ; } }'
            )
            tmp_conf = f"/tmp/nginx-health-{uuid.uuid4().hex}.conf"
            with open(tmp_conf, "w") as f:
                f.write(conf)

            cmd = (
                f'podman run --rm -d '
                f'-v {tmp_conf}:/etc/nginx/conf.d/default.conf:ro '
                f'-p {ctx_inner.port}:80 docker.io/library/nginx:alpine'
            )
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(stderr.decode().strip())
            ctx_inner.container_id = stdout.decode().strip()

        orig = PodmanModelRunner._start_model_process
        PodmanModelRunner._start_model_process = patched_start.__get__(runner, PodmanModelRunner)

        try:
            async with runner as r:
                await r.start("test-model", port=port)

                ctx = next(iter(r._models.values()))
                assert ctx.state == RunnerState.RUNNING
                cid = getattr(ctx, "container_id", None)
                assert cid
                print(f"[+] Model running in container: {cid[:12]}")

            await asyncio.sleep(1.0)
        finally:
            PodmanModelRunner._start_model_process = orig

        # Clean up the container (may have been removed by --rm already)
        ctx = runner._models.get("test-model")
        if ctx and getattr(ctx, "container_id", None):
            _podman("rm", "-f", ctx.container_id)
        _kill_port(port)

        print("[+] Full lifecycle — start → run → stop_all — clean")
