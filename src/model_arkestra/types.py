"""Backward-compat facade.

The canonical state machine and per-model object live in
``model_arkestra.models``. They are re-exported here so existing imports
(``from model_arkestra.types import RunnerState, _Model``) keep working.
The runner exception hierarchy is defined below.
"""
from __future__ import annotations

# ── state machine (single source of truth: models/state.py) ────────────
from model_arkestra.models.state import (  # noqa: F401
    RunnerState, IllegalTransition, can, target, transition,
)

# ── per-model object (models/model.py) ─────────────────────────────────
from model_arkestra.models.model import _Model  # noqa: F401

# ── runner exception hierarchy (unchanged) ─────────────────────────────
class RunnerError(Exception): """Base exception for all runner failures."""
class ServerReadyTimeout(RunnerError): """Server did not become ready within timeout."""
class ModelNotStarted(RunnerError): """Request on a model that hasn't been started."""
class MaxRestartsExceeded(RunnerError): """Process has crashed too many times."""
class ModelShutdown(RunnerError): """Request made after the model was stopped."""


__all__ = [
    "RunnerState", "IllegalTransition", "can", "target", "transition",
    "_Model",
    "RunnerError", "ServerReadyTimeout", "ModelNotStarted",
    "MaxRestartsExceeded", "ModelShutdown",
]
