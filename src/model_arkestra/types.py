"""Backward-compat facade.

The canonical state machine and per-model object now live in
``model_arkestra.models``. Everything here is re-exported so existing
imports (``from model_arkestra.types import RunnerState, _ModelContext``)
keep working during the migration. ``_ModelContext`` is an alias of the new
``_Model`` — a drop-in superset.
"""
from __future__ import annotations

# ── state machine (single source of truth: models/state.py) ────────────
from model_arkestra.models.state import (  # noqa: F401
    RunnerState, IllegalTransition, can, target, transition,
)

# ── per-model object (models/model.py) ─────────────────────────────────
from model_arkestra.models.model import _Model  # noqa: F401

# Legacy alias — existing code references _ModelContext.
_ModelContext = _Model

# ── runner exception hierarchy (unchanged) ─────────────────────────────
class RunnerError(Exception): """Base exception for all runner failures."""
class ServerReadyTimeout(RunnerError): """Server did not become ready within timeout."""
class ModelNotStarted(RunnerError): """Request on a model that hasn't been started."""
class MaxRestartsExceeded(RunnerError): """Process has crashed too many times."""
class ModelShutdown(RunnerError): """Request made after the model was stopped."""


__all__ = [
    "RunnerState", "IllegalTransition", "can", "target", "transition",
    "_Model", "_ModelContext",
    "RunnerError", "ServerReadyTimeout", "ModelNotStarted",
    "MaxRestartsExceeded", "ModelShutdown",
]
