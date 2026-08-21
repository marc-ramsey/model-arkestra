"""Tests for _ModelContext ring buffer — direct, no mocks."""
import sys
sys.path.insert(0, "src")

from model_arkestra.types import _ModelContext


class TestContextLogBuffer:
    """_ModelContext._append_log_line + _get_lines_since on real UnicodeRingBuffer.

    Note: callers (process.py) strip \\r\\n before calling _append_log_line,
    so we pass plain strings here — no trailing newlines."""

    def test_empty_returns_nothing(self):
        ctx = _ModelContext("empty", 1000)
        result, oldest = ctx._get_lines_since(0, 5)
        assert result == []
        assert oldest == 0

    def test_one_line(self):
        ctx = _ModelContext("test", 2000)
        ctx._append_log_line("hello")
        result, _ = ctx._get_lines_since(0, 10)
        assert len(result) == 1
        assert result[0][1] == "hello"

    def test_multiple_lines(self):
        ctx = _ModelContext("test", 2000)
        for line in ["alpha", "beta", "gamma"]:
            ctx._append_log_line(line)
        result, _ = ctx._get_lines_since(0, 10)
        assert len(result) == 3
        assert [t for _, t in result] == ["alpha", "beta", "gamma"]

    def test_max_lines_limit(self):
        ctx = _ModelContext("test", 2000)
        for i in range(10):
            ctx._append_log_line(f"line{i}")
        result, _ = ctx._get_lines_since(0, 3)
        assert len(result) == 3
        assert [t for _, t in result] == ["line7", "line8", "line9"]

    def test_unicode_roundtrip(self):
        ctx = _ModelContext("test", 2000)
        ctx._append_log_line("caf\u00e9")
        ctx._append_log_line("\U0001f680 launch")
        result, _ = ctx._get_lines_since(0, 5)
        assert [t for _, t in result] == ["caf\u00e9", "\U0001f680 launch"]

    def test_buffer_overflow_discards_oldest(self):
        """Small buffer: write more lines than capacity holds."""
        ctx = _ModelContext("test", 2000, max_log_lines=2)
        # Lines are short (~1 byte + newline). Capacity ~400 bytes → plenty of slots.
        # Use very short strings to actually fill it fast.
        for i in range(50):
            ctx._append_log_line(f"L{i}")
        result, _ = ctx._get_lines_since(0, 20)
        assert len(result) < 50  # ring buffer discarded oldest entries

    def test_ring_wrap_arounds(self):
        """Write enough to force head wrap, then read."""
        ctx = _ModelContext("test", 2000)
        for i in range(100):
            ctx._append_log_line(f"line{i:03d}")
        result, _ = ctx._get_lines_since(0, 5)
        assert len(result) == 5
        assert [t for _, t in result] == ["line095", "line096", "line097", "line098", "line099"]
