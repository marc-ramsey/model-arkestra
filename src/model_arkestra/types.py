from __future__ import annotations
import asyncio
import logging
import struct
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class RunnerState(Enum):
    STOPPED = auto()
    LOADING = auto()
    RUNNING = auto()
    ERROR = auto()
    STOPPING = auto()  # stop() called — do not restart
    UNCACHED = auto()  # checkpoint not yet downloaded

    @property
    def is_terminal(self) -> bool:
        """True when the model is not currently active (stopped, stopping)."""
        return self in (RunnerState.STOPPED, RunnerState.STOPPING)

class RunnerError(Exception): """Base exception for all ProcessModelRunner failures."""
class ServerReadyTimeout(RunnerError): """Server did not become ready within timeout."""
class ModelNotStarted(RunnerError): """Request on a model that hasn't been started."""
class MaxRestartsExceeded(RunnerError): """Process has crashed too many times."""
class ModelShutdown(RunnerError): """Request made after the model was stopped."""

# ── Ring-buffer log entry format ────────────────────────────────────────────
# Each entry: [4-byte big-endian line length][UTF-8 bytes]
_LINE_HEADER = struct.Struct('>I')  # 4-byte unsigned int, big-endian

class _ModelContext:
    DEFAULT_LOG_LINES = 500
    AVG_LINE_BYTES = 200  # estimate — used to size the fixed ring buffer

    def __init__(self, name: str, port: int, max_log_lines: int = DEFAULT_LOG_LINES):
        self.name = name
        self.port = port
        self.runner_type: Optional[str] = None
        self.backend_id: Optional[str] = None
        self._runner: Optional[BaseModelRunner] = None
        self.process: Optional[asyncio.subprocess.Process] = None
        self.container_id: Optional[str] = None
        self.state = RunnerState.STOPPED
        self.restart_count = 0
        self.last_error: Optional[str] = None
        self.broadcast_addr: str = "0.0.0.0"

        # ── Ring-buffer log storage (fixed allocation, no per-line objects) ───
        # One contiguous bytearray that wraps when full — entries are prefixed
        # with their own length (4 bytes), so the buffer is self-indexing:
        #   [len1|data1][len2|data2]... and old data gets overwritten on wrap.
        self._log_buf_size = max_log_lines * self.AVG_LINE_BYTES  # fixed at init
        self._log_buffer = bytearray(self._log_buf_size)

        # Write cursor: next byte position (wraps around _log_buf_size)
        self._log_pos: int = 0
        # Oldest valid data start position (advanced when old data is pruned)
        self._log_tail: int = 0
        # Monotonically increasing line counter — each log entry gets seq=log_seq
        self._log_seq: int = 0
        # Oldest sequence number still possibly in the ring buffer.
        # Used to calculate X-Missed-Lines when a client's `since` value is stale.
        self._oldest_valid_seq: int = 0

    def _append_log_line(self, line: str) -> int:
        """Append a log line into the ring buffer. Returns its seq number."""
        data = line.encode("utf-8", errors="replace")
        header = _LINE_HEADER.pack(len(data))
        entry_len = len(header) + len(data)

        # Advance sequence counter — this is the line's identity
        self._log_seq += 1
        seq = self._log_seq

        if entry_len > self._log_buf_size:
            # Line is larger than entire buffer — write what fits and prune
            self._log_buffer[0:entry_len] = header + data[: self._log_buf_size - len(header)]
            self._log_pos = entry_len % self._log_buf_size
            self._oldest_valid_seq = seq  # everything before this is gone
        else:
            remaining = self._log_buf_size - self._log_pos
            if entry_len <= remaining:
                # Fits in one contiguous chunk — write, wrap if needed
                pos = self._log_pos
                self._log_buffer[pos:pos + len(header)] = header
                self._log_buffer[pos + len(header):pos + entry_len] = data
                self._log_pos = (pos + entry_len) % self._log_buf_size
            else:
                # Wraps around end of buffer — two writes
                pos = self._log_pos
                first = remaining
                self._log_buffer[pos:] = header[:first]
                second = len(header) - first
                rest = entry_len - first
                self._log_buffer[:second + (len(data))] = header[first:] + data
                self._log_pos = (pos + entry_len) % self._log_buf_size

        return seq

    def _get_lines_since(self, since: int, max_lines: int) -> Tuple[List[Tuple[int, str]], int]:
        """Read log lines with seq > since. Returns ([(seq, text),...], oldest_seq)."""
        result = []
        pos = self._log_tail
        count = 0
        header_size = _LINE_HEADER.size

        # Walk ring buffer: read each entry's length header, skip until we find seq == since
        max_walk = self._log_buf_size // (header_size + 1)  # safe upper bound on entries
        for _ in range(max_walk):
            if pos >= self._log_buf_size - header_size:
                # Header spans wrap boundary — read first part, rest from beginning
                h_bytes = bytes(self._log_buffer[pos:]) + bytes(self._log_buffer[:header_size - (self._log_buf_size - pos)])
            else:
                h_bytes = bytes(self._log_buffer[pos:pos + header_size])

            if len(h_bytes) < header_size:
                break

            line_len = _LINE_HEADER.unpack(h_bytes)[0]
            entry_total = header_size + line_len

            # Check if this entry's start position is valid (>= tail, accounting for wrap)
            abs_pos = self._log_tail + count * (entry_total)  # approximation
            actual_start = self._log_tail  # we're walking sequentially from tail

            # We can't easily compute seq from offset alone — we need to scan
            # Actually, we don't store seq in the buffer. Seq is just a counter.
            # So scanning means: walk all entries, assign seq numbers relative
            # to oldest_valid_seq. Each entry gets oldest_valid_seq + N+1.
            break

        # Simpler approach: scan all entries from tail, compute seq = _oldest_valid_seq + N
        pos = self._log_tail
        entry_index = 0

        for _ in range(max_walk):
            remaining_buf = self._log_buf_size - pos
            if remaining_buf < header_size:
                # Header spans wrap — reconstruct from two parts
                h_part1 = bytes(self._log_buffer[pos:])
                h_part2 = bytes(self._log_buffer[:header_size - len(h_part1)])
                h_bytes = h_part1 + h_part2
            else:
                h_bytes = bytes(self._log_buffer[pos:pos + header_size])

            if len(h_bytes) < header_size:
                break

            try:
                line_len = _LINE_HEADER.unpack(h_bytes)[0]
            except struct.error:
                break  # corrupt header — stop

            entry_total = header_size + line_len
            pos += header_size

            if pos + line_len > self._log_buf_size:
                data_part1 = bytes(self._log_buffer[pos:])
                data_part2 = bytes(self._log_buffer[:line_len - len(data_part1)])
                text = (data_part1 + data_part2).decode("utf-8", errors="replace")
            else:
                text = bytes(self._log_buffer[pos:pos + line_len]).decode("utf-8", errors="replace")

            pos += line_len
            entry_index += 1

            seq = self._oldest_valid_seq + entry_index
            if seq > since and len(result) < max_lines:
                result.append((seq, text))

        return result, self._oldest_valid_seq

    def _prune(self) -> None:
        """Prune entries that have been fully overwritten by ring wrap."""
        # Walk from current tail position. If we've cycled past oldest_valid_seq,
        # advance it. This happens when pos catches up to and passes tail (full wrap).
        if self._log_pos == self._log_tail:
            # Full wrap — all data was overwritten
            self._oldest_valid_seq = self._log_seq

    def __repr__(self) -> str:
        return f"<{self.name} port={self.port} state={self.state.name}>"
