"""Tests for the HTTP delta log endpoint and ring buffer."""

import struct
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from model_arkestra.admin import ArkestraAdmin
from model_arkestra.types import _ModelContext


def _make_server_ctx_with_lines(lines: list[str]) -> _ModelContext:
    """Create a real ModelContext and populate it with log lines."""
    ctx = _ModelContext("test-model", 9999, max_log_lines=50)
    for line in lines:
        ctx._append_log_line(line)
    return ctx


def _build_admin_with_ctx(ctx: _ModelContext):
    """Build a FastAPI app with the admin routes and a mock arkestra that returns ctx."""
    arkestra = MagicMock()
    arkestra.find_context.return_value = ctx
    arkestra.cm.data = {"models": {"test-model": {}}}

    app = FastAPI()
    server = MagicMock()
    server._arkestra = arkestra

    admin = ArkestraAdmin(server, None, app)
    admin.install()
    return app


# ── Ring buffer unit tests ──────────────────────────────────────────────────


def test_basic_write_and_read():
    """Write lines, read them back with correct sequence numbers."""
    ctx = _ModelContext("test", 9999, max_log_lines=50)
    for i in range(3):
        seq = ctx._append_log_line(f"line{i}")
        assert seq == i + 1

    result, oldest = ctx._get_lines_since(0, 100)
    assert len(result) == 3
    for i in range(3):
        assert result[i] == (i + 1, f"line{i}")


def test_since_filters():
    """since=N returns only lines with seq > N."""
    ctx = _ModelContext("test", 9999, max_log_lines=50)
    for i in range(10):
        ctx._append_log_line(f"line{i}")

    result, _ = ctx._get_lines_since(5, 100)
    assert len(result) == 5
    assert result[0] == (6, "line5")

    ctx2 = _ModelContext("test", 9999, max_log_lines=50)
    for i in range(10):
        ctx2._append_log_line(f"line{i}")

    result, _ = ctx2._get_lines_since(9, 100)
    assert len(result) == 1
    assert result[0] == (10, "line9")


def test_max_lines_limit():
    """max_lines caps returned entries."""
    ctx = _ModelContext("test", 9999, max_log_lines=50)
    for i in range(10):
        ctx._append_log_line(f"line{i}")

    result, _ = ctx._get_lines_since(0, 3)
    assert len(result) == 3


def test_utf8_roundtrip():
    """Non-ASCII log lines survive the ring buffer."""
    ctx = _ModelContext("test", 9999, max_log_lines=50)
    line = "Hello, 世界! 🌍"
    ctx._append_log_line(line)

    result, _ = ctx._get_lines_since(0, 100)
    assert result[0] == (1, line)


def test_wrap_truncates_oldest():
    """Buffer full → oldest entries evicted, only newest remain."""
    # Create context with a very small buffer so lines actually wrap.
    # Each entry is ~6 bytes (4-byte seq prefix + 2 bytes for text like "x0")
    from model_arkestra.unicode_ringbuffer import UnicodeRingBuffer
    ctx = _ModelContext("test", 9999, max_log_lines=2)
    tiny_buf = UnicodeRingBuffer(50)  # ~8 entries max
    ctx._log_ring = tiny_buf

    for i in range(15):
        ctx._append_log_line(f"x{i}")

    result, _ = ctx._get_lines_since(0, 100)
    assert len(result) <= 2


# ── Admin endpoint tests ────────────────────────────────────────────────────


def test_empty_result_no_lines():
    """No new lines → empty lines array."""
    ctx = _make_server_ctx_with_lines([])
    app = _build_admin_with_ctx(ctx)
    client = TestClient(app)
    resp = client.get("/admin/log/test-model", params={"since": 0})

    assert resp.status_code == 200
    data = resp.json()
    assert data["lines"] == []


def test_returns_new_lines():
    """Delta query returns only lines with seq > since."""
    ctx = _make_server_ctx_with_lines(["a", "b", "c", "d"])
    app = _build_admin_with_ctx(ctx)
    client = TestClient(app)
    resp = client.get("/admin/log/test-model", params={"since": 2})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["lines"]) == 2
    assert data["lines"][0] == {"seq": 3, "text": "c"}


def test_headers_present():
    """Response includes X-Missed-Lines and X-Current-Max headers."""
    ctx = _make_server_ctx_with_lines(["a", "b"])
    app = _build_admin_with_ctx(ctx)
    client = TestClient(app)
    resp = client.get("/admin/log/test-model")

    assert "X-Missed-Lines" in resp.headers
    assert "X-Current-Max" in resp.headers


def test_404_unknown_model():
    """Model not in config → 404."""
    arkestra = MagicMock()
    arkestra.cm.data = {"models": {}}

    app = FastAPI()
    server = MagicMock()
    server._arkestra = arkestra

    admin = ArkestraAdmin(server, None, app)
    admin.install()

    client = TestClient(app)
    resp = client.get("/admin/log/no-such-model")
    assert resp.status_code == 404


# ── Integration: round-trip polling pattern ─────────────────────────────────


def test_polling_roundtrip():
    """Simulate client polling: first since=0, then since=last_seq."""
    ctx = _make_server_ctx_with_lines(["line1", "line2"])
    app = _build_admin_with_ctx(ctx)
    client = TestClient(app)

    # First poll: get all lines
    resp = client.get("/admin/log/test-model", params={"since": 0})
    data = resp.json()
    assert len(data["lines"]) == 2

    # Second poll with last known seq: no new lines
    resp = client.get("/admin/log/test-model", params={"since": data["since"]})
    assert resp.json()["lines"] == []


def test_format_is_correct():
    """Each line in response has 'seq' and 'text' keys."""
    ctx = _make_server_ctx_with_lines(["hello"])
    app = _build_admin_with_ctx(ctx)
    client = TestClient(app)
    resp = client.get("/admin/log/test-model", params={"since": 0})

    for item in resp.json()["lines"]:
        assert "seq" in item
        assert "text" in item
