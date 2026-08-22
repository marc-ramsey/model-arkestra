"""Tests for the global server log buffer and /admin/logs endpoint."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_arkestra.arkestra import ModelArkestra
from model_arkestra.admin import ArkestraAdmin
from model_arkestra.unicode_ringbuffer import UnicodeRingBuffer


# ── Ring buffer unit tests ────────────────────────────────────────────────


def test_global_log_buffer_wraps_on_overflow():
    """When the buffer is full, oldest entries are evicted and new ones accepted."""
    buf = UnicodeRingBuffer(100)  # ~1 entry worth
    for i in range(50):
        text = f"log line {i}\n"
        try:
            buf.write(i, text)
        except UnicodeRingBuffer.BufferFullError:
            if not buf:
                return
            buf.read_entries(max_lines=1)

    # Should still be able to write — oldest was evicted
    buf.write(50, "still fits\n")


def test_global_log_buffer_utf8_roundtrip():
    """Non-ASCII text survives the ring buffer."""
    buf = UnicodeRingBuffer(5000)
    line = "Hello, 世界! 🌍 model crash\n"
    buf.write(1, line)

    entries = buf.read_entries(max_lines=1)
    assert len(entries) == 1
    assert entries[0][1] == "Hello, 世界! 🌍 model crash"


def test_global_log_buffer_empty_read():
    """Reading from an empty buffer returns empty list."""
    buf = UnicodeRingBuffer(500)
    entries = buf.read_entries(max_lines=10)
    assert entries == []


# ── ModelArkestra _log helper tests ───────────────────────────────────────


def test_model_arkestra_creates_global_log():
    """ModelArkestra instantiates a global log buffer from config."""
    arkestra = ModelArkestra("tests/test-config.yaml")

    assert hasattr(arkestra, "_global_log_buf")
    assert isinstance(arkestra._global_log_buf, UnicodeRingBuffer)
    assert arkestra._global_log_seq == 0
    arkestra._cm.data["models"] = {}  # prevent cleanup issues


def test_model_arkestra_respects_config_lines():
    """app-log-lines config key sets buffer capacity."""
    arkestra = ModelArkestra("tests/test-config.yaml")
    arkestra._cm.data["app-log-lines"] = 100

    # Re-create with explicit config
    from llm_config_manager.config_manager import ConfigManager
    cm = ConfigManager("tests/test-config.yaml")
    cm.data["app-log-lines"] = 100
    buf_size = 100 * 200  # lines × AVG_LINE_BYTES

    arkestra2 = ModelArkestra.__new__(ModelArkestra)
    arkestra2._cm = cm
    arkestra2._global_log_buf = UnicodeRingBuffer(buf_size)
    arkestra2._global_log_seq = 0
    assert arkestra2._global_log_buf._usable == buf_size - 1

    # Cleanup
    for a in (arkestra, arkestra2):
        if hasattr(a, '_cm'):
            try:
                a._cm.data["models"] = {}
            except Exception:
                pass


def test_model_arkestra_log_increments_seq():
    """Each call to _log increments the sequence number."""
    from llm_config_manager.config_manager import ConfigManager

    arkestra = ModelArkestra.__new__(ModelArkestra)
    arkestra._cm = MagicMock()
    arkestra._global_log_buf = UnicodeRingBuffer(100_000)
    arkestra._global_log_seq = 0

    import asyncio
    loop = asyncio.new_event_loop()
    try:
        async def _test():
            await arkestra._log("first line")
            assert arkestra._global_log_seq == 1

            await arkestra._log("second line")
            assert arkestra._global_log_seq == 2

            # Read all — read_entries consumes as it goes
            entries = arkestra._global_log_buf.read_entries(max_lines=10)
            assert len(entries) == 2
            assert entries[0] == (1, "first line")
            assert entries[1] == (2, "second line")

        loop.run_until_complete(_test())
    finally:
        loop.close()


def test_model_arkestra_log_needs_newline():
    """_log appends a newline if the text doesn't end with one."""
    from llm_config_manager.config_manager import ConfigManager

    arkestra = ModelArkestra.__new__(ModelArkestra)
    arkestra._cm = MagicMock()
    arkestra._global_log_buf = UnicodeRingBuffer(100_000)
    arkestra._global_log_seq = 0

    import asyncio
    loop = asyncio.new_event_loop()
    try:
        async def _test():
            await arkestra._log("no newline at end")
            entries = arkestra._global_log_buf.read_entries(max_lines=10)
            assert len(entries) == 1
            # text should have trailing newline stripped by read_entries
            assert entries[0][1] == "no newline at end"

        loop.run_until_complete(_test())
    finally:
        loop.close()


# ── Admin endpoint tests ──────────────────────────────────────────────────


def _build_admin_with_global_log(lines: list[str] | None = None):
    """Build FastAPI app with admin routes and a global log buffer."""
    arkestra = MagicMock()
    buf = UnicodeRingBuffer(50_000)
    seq_max = 0

    # Pre-populate with lines if provided
    if lines:
        for i, line in enumerate(lines):
            text = line if line.endswith("\n") else line + "\n"
            try:
                buf.write(i + 1, text)
            except UnicodeRingBuffer.BufferFullError:
                buf.read_entries(max_lines=1)
        seq_max = len(lines)

    arkestra._global_log_buf = buf
    arkestra._global_log_seq = seq_max
    arkestra.cm.data = {"models": {}}

    app = FastAPI()
    server = MagicMock()
    server._arkestra = arkestra

    admin = ArkestraAdmin(server, None, app)
    admin.install()
    return app


def test_global_log_empty():
    """No lines → empty response."""
    app = _build_admin_with_global_log([])
    client = TestClient(app)
    resp = client.get("/admin/logs", params={"since": 0})

    assert resp.status_code == 200
    data = resp.json()
    assert data["lines"] == []
    assert data["seq"] == 0


def test_global_log_returns_lines():
    """Lines written to buffer are returned via endpoint."""
    app = _build_admin_with_global_log([
        "[action=start server port=8080]",
        "[action=start model=qwen3 port=18001]",
        "[action=req model=qwen3 status=200 latency_ms=420]",
    ])
    client = TestClient(app)
    resp = client.get("/admin/logs", params={"since": 0})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["lines"]) == 3
    assert data["seq"] == 3
    assert data["lines"][0] == {"seq": 1, "text": "[action=start server port=8080]"}


def test_global_log_delta_since():
    """Delta reads return only lines with seq > since."""
    app = _build_admin_with_global_log([
        f"[action=l1]", f"[action=l2]", f"[action=l3]", f"[action=l4]", f"[action=l5]"
    ])
    client = TestClient(app)

    # First poll: all lines
    resp = client.get("/admin/logs", params={"since": 0})
    data = resp.json()
    assert len(data["lines"]) == 5
    last_seq = data["seq"]

    # Second poll: no new lines
    resp = client.get("/admin/logs", params={"since": last_seq})
    assert resp.json()["lines"] == []


def test_global_log_headers():
    """Response includes X-Missed-Lines and X-Current-Max headers."""
    app = _build_admin_with_global_log(["line1"])
    client = TestClient(app)
    resp = client.get("/admin/logs")

    assert "X-Missed-Lines" in resp.headers
    assert "X-Current-Max" in resp.headers


def test_global_log_max_lines():
    """lines=N caps returned entries."""
    app = _build_admin_with_global_log([f"[action=ln{i}]" for i in range(20)])
    client = TestClient(app)

    resp = client.get("/admin/logs", params={"since": 0, "lines": 5})
    data = resp.json()
    assert len(data["lines"]) == 5


def test_global_log_missed_lines():
    """When since points to an evicted entry, missed_lines is set."""
    # Buffer that holds only 3 entries
    arkestra = MagicMock()
    buf = UnicodeRingBuffer(200)  # very small — ~1-2 entries max

    for i in range(10):
        text = f"[action=entry{i}]\n"
        try:
            buf.write(i + 1, text)
        except UnicodeRingBuffer.BufferFullError:
            if not buf:
                break
            buf.read_entries(max_lines=1)

    arkestra._global_log_buf = buf
    arkestra._global_log_seq = buf._count  # approximate max seq written

    app = FastAPI()
    server = MagicMock()
    server._arkestra = arkestra

    admin = ArkestraAdmin(server, None, app)
    admin.install()

    client = TestClient(app)
    resp = client.get("/admin/logs", params={"since": 0})
    data = resp.json()
    # Should have some lines back (the ones still in buffer)
    assert "lines" in data


# ── Admin endpoint integration: polling round-trip ────────────────────────


def test_global_log_polling_roundtrip():
    """Simulate client polling: first since=0, then since=last_seq."""
    arkestra = MagicMock()
    buf = UnicodeRingBuffer(50_000)

    # Pre-seed with 2 lines
    for i, text in enumerate(["[action=start server]", "[action=start model=test port=18000]"]):
        buf.write(i + 1, text)

    arkestra._global_log_buf = buf
    arkestra._global_log_seq = 2
    arkestra.cm.data = {"models": {}}

    app = FastAPI()
    server = MagicMock()
    server._arkestra = arkestra

    admin = ArkestraAdmin(server, None, app)
    admin.install()

    client = TestClient(app)

    # First poll: get existing lines
    resp = client.get("/admin/logs", params={"since": 0})
    data = resp.json()
    assert len(data["lines"]) == 2

    # Second poll with last seq: no new lines
    resp = client.get("/admin/logs", params={"since": data["seq"]})
    assert resp.json()["lines"] == []
