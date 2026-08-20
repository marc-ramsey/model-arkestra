from __future__ import annotations
import asyncio
import logging
from collections import deque
from enum import Enum, auto
from typing import Any, Deque, Dict, Optional, List

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

class _ModelContext:
    DEFAULT_LOG_LINES = 2000

    def __init__(self, name: str, port: int, max_log_lines: int = DEFAULT_LOG_LINES):
        self.name = name
        self.port = port
        self.runner_type: Optional[str] = None
        self.backend_id: Optional[str] = None
        self.process: Optional[asyncio.subprocess.Process] = None
        self.container_id: Optional[str] = None
        self.state = RunnerState.STOPPED
        self.restart_count = 0
        self.last_error: Optional[str] = None
        self.broadcast_addr: str = "0.0.0.0"
        self._log_buffer: Deque[str] = deque(maxlen=max_log_lines)

    def __repr__(self) -> str:
        return f"<{self.name} port={self.port} state={self.state.name}>"
