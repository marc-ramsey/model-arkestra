from __future__ import annotations
import asyncio
import logging
import struct
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from model_arkestra.unicode_ringbuffer import UnicodeRingBuffer

logger = logging.getLogger(__name__)

class RunnerState(Enum):
    STOPPED = auto()
    LOADING = auto()
    RUNNING = auto()
    ERROR = auto()
    STOPPING = auto()  # stop() called — do not restart
    UNCACHED = auto()  # checkpoint not yet downloaded
    DOWNLOADING = auto()  # model checkpoint actively being downloaded

    @property
    def is_terminal(self) -> bool:
        """True when the model is not currently active (stopped, stopping)."""
        return self in (RunnerState.STOPPED, RunnerState.STOPPING)

class RunnerError(Exception): """Base exception for all ProcessModelRunner failures."""
class ServerReadyTimeout(RunnerError): """Server did not become ready within timeout."""
class ModelNotStarted(RunnerError): """Request on a model that hasn't been started."""
class MaxRestartsExceeded(RunnerError): """Process has crashed too many times."""
class ModelShutdown(RunnerError): """Request made after the model was stopped."""


class _ModelContext:
    DEFAULT_LOG_LINES = 500
    AVG_LINE_BYTES = 200

    def __init__(self, name: str, port: int, max_log_lines: int = DEFAULT_LOG_LINES):
        self.name = name
        self.port = port
        self.runner_type: Optional[str] = None
        self.backend_id: Optional[str] = None
        self._runner: Optional["BaseModelRunner"] = None
        self.process: Optional[asyncio.subprocess.Process] = None
        self.container_id: Optional[str] = None
        self.state = RunnerState.STOPPED
        self.restart_count = 0
        self.last_error: Optional[str] = None
        self.broadcast_addr: str = "0.0.0.0"
        self._remote_base_url: Optional[str] = None  # actual URL for remote models (callers can bypass proxy)
        self.download_task: Optional[asyncio.Task] = None  # background download task

        # Line sequence counter
        self._log_seq: int = 0

        # Ring buffer for log lines
        buf_bytes = max_log_lines * self.AVG_LINE_BYTES
        if buf_bytes < 10:
            raise ValueError(
                f"log buffer too small: {buf_bytes} bytes (max_log_lines={max_log_lines})"
            )
        self._log_ring = UnicodeRingBuffer(buf_bytes)

    def _append_log_line(self, line: str) -> int:
        """Append a log line into the ring buffer."""
        self._log_seq += 1
        if not line.endswith("\n"):
            line = line + "\n"
        for _ in range(20):  # evict at most 20 oldest entries
            try:
                self._log_ring.write(self._log_seq, line)
                break
            except UnicodeRingBuffer.BufferFullError:
                if not self._log_ring:  # ring is empty, nothing to evict
                    return self._log_seq
                self._log_ring.read_entries(max_lines=1)  # discard one oldest entry
        else:
            pass  # gave up — discard oldest
        return self._log_seq

    def _get_lines_since(self, since: int, max_lines: int) -> Tuple[List[Tuple[int, str]], int]:
        """Read log lines with seq > since. Returns ([(seq, text),...], oldest_seq)."""
        entries = self._log_ring.read_entries(max_lines=max_lines, next_line=since)
        oldest_seq = entries[0][0] if entries else 0
        return entries, oldest_seq

    def __repr__(self) -> str:
        return f"<{self.name} port={self.port} state={self.state.name}>"
