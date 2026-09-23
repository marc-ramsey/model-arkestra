"""The per-model object. Owns everything about one model instance:

  state machine · port · log ring · backend (how it runs) · client (how it's talked to)

One Model per configured model name. The registry maps name → Model. No runner
keeps a ``_models`` dict any more — the Model *is* the unit of ownership.
"""
from __future__ import annotations

import asyncio
from typing import Any, FrozenSet, Optional

from model_arkestra.models.state import RunnerState, transition
from model_arkestra.models.capabilities import derive_capabilities
from model_arkestra.unicode_ringbuffer import UnicodeRingBuffer


class _Model:
    DEFAULT_LOG_LINES = 500
    AVG_LINE_BYTES = 200

    def __init__(self, name: str, port: Optional[int] = None,
                 max_log_lines: int = DEFAULT_LOG_LINES):
        self.name = name
        self.port = port

        # ── routing / identity ────────────────────────────────
        self.backend_id: Optional[str] = None
        self.runner_type: Optional[str] = None   # process|podman|docker|onnx|remote
        self.cluster: str = "local"

        # ── lifecycle handles (owned by this model, not a runner) ──
        self.process: Optional[asyncio.subprocess.Process] = None
        self.container_id: Optional[str] = None
        self._provider: Optional[Any] = None     # Provider (inference) instance

        # ── capability derivation inputs (set by the registry/factory) ──
        self._model_cfg: dict = {}
        self._backend_cfg: dict = {}
        # sock_read bound for stream chunks (seconds); None → provider default
        self._stream_sock_timeout: Optional[float] = None

        # ── state machine (mutated only via set_state) ─────────
        self._state = RunnerState.STOPPED
        self.restart_count = 0
        self.last_error: Optional[str] = None

        # ── download / pull ───────────────────────────────────
        self.download_task: Optional[asyncio.Task] = None
        self.download_kind: str = "checkpoint"   # checkpoint | image
        self.download_pct: Optional[float] = None
        self.download_downloaded: Optional[int] = None
        self.download_speed_mbps: Optional[float] = None
        self.download_current: str = ""

        # ── remote routing (remote models only) ───────────────
        self._remote_base_url: Optional[str] = None
        self._admin_key: str = ""

        # ── log ring buffer ───────────────────────────────────
        buf_bytes = max_log_lines * self.AVG_LINE_BYTES
        if buf_bytes < 10:
            raise ValueError(f"log buffer too small: {buf_bytes} bytes (max_log_lines={max_log_lines})")
        self._log_ring = UnicodeRingBuffer(buf_bytes)
        self._log_seq = 0
        self._log_buf_lines = max_log_lines

    # ── state machine choke point ───────────────────────────────
    @property
    def state(self) -> RunnerState:
        return self._state

    def set_state(self, event: str) -> RunnerState:
        """Validate ``event`` from the current state and apply it."""
        self._state = transition(self._state, event)
        return self._state

    # ── capabilities (derived, cached on first access) ───────────
    @property
    def capabilities(self) -> FrozenSet[str]:
        if not hasattr(self, "_capabilities"):
            self._capabilities = derive_capabilities(self._model_cfg, self._backend_cfg)
        return self._capabilities

    # ── inference provider (built lazily once routing is known) ───
    @property
    def provider(self) -> Any:
        """Return the model's active Provider, building it on first use.

        The provider kind follows ``runner_type``: remote → RemoteProvider,
        onnx → OnnxProvider, anything else (process/container) → LlamaProvider.
        """
        if self._provider is not None:
            return self._provider
        from model_arkestra.providers import (
            LlamaProvider, OnnxProvider, RemoteProvider,
        )
        rt = self.runner_type
        if rt == "remote":
            self._provider = RemoteProvider(
                self.name, self._remote_base_url or "", self._admin_key
            )
        elif rt == "onnx":
            prov = OnnxProvider(self)
            prov.model._onnx_capabilities = self.capabilities
            self._provider = prov
        else:
            # process / podman / docker — local llama-server on the model port
            self._provider = LlamaProvider(self.name, self.port or 0)
        return self._provider

    # ── log ring ────────────────────────────────────────────────
    def _append_log_line(self, line: str) -> int:
        self._log_seq += 1
        if not line.endswith("\n"):
            line = line + "\n"
        self._log_ring.write_force(self._log_seq, line)
        return self._log_seq

    def _get_lines_since(self, since: int, max_lines: int):
        entries = self._log_ring.read_entries(max_lines=max_lines, next_line=since)
        oldest_seq = entries[0][0] if entries else 0
        return entries, oldest_seq

    def __repr__(self) -> str:
        return f"<{self.name} port={self.port} state={self._state.name}>"
