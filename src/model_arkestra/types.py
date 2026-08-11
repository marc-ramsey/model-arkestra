from __future__ import annotations
import asyncio
import logging
from enum import Enum, auto
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class RunnerState(Enum):
    STOPPED = auto()
    LOADING = auto()
    RUNNING = auto()
    ERROR = auto()
    STOPPING = auto()  # stop() called — do not restart

class RunnerError(Exception): """Base exception for all ProcessModelRunner failures."""
class ServerReadyTimeout(RunnerError): """Server did not become ready within timeout."""
class ModelNotStarted(RunnerError): """Request on a model that hasn't been started."""
class MaxRestartsExceeded(RunnerError): """Process has crashed too many times."""
class ModelShutdown(RunnerError): """Request made after the model was stopped."""

class _ModelContext:
    def __init__(self, name: str, port: int):
        self.name = name
        self.port = port
        self.runner_type: Optional[str] = None
        self.backend_id: Optional[str] = None
        self.process: Optional[asyncio.subprocess.Process] = None
        self.container_id: Optional[str] = None
        self.state = RunnerState.STOPPED
        self.restart_count = 0
        self.last_error: Optional[str] = None

    def __repr__(self) -> str:
        return f"<{self.name} port={self.port} state={self.state.name}>"
