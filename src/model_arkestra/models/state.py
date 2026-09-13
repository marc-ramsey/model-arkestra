"""Single state machine for every model, regardless of backend kind.

One table, one transition() choke point. No other code path mutates
``Model.state``. Illegal transitions raise ``IllegalTransition`` so bugs
surface immediately instead of corrupting lifecycle state silently.
"""
from __future__ import annotations

from enum import Enum, auto


class RunnerState(Enum):
    UNCACHED = auto()     # checkpoint not yet pulled (local models)
    DOWNLOADING = auto()  # checkpoint or image actively being pulled
    STOPPED = auto()      # allocated, not running — restartable in place
    LOADING = auto()      # launch issued, waiting for readiness
    RUNNING = auto()      # ready and serving
    STOPPING = auto()     # stop requested — no restarts until done
    ERROR = auto()        # terminal (restart limit exhausted / health error)

    @property
    def is_terminal(self) -> bool:
        """True when the model is not currently active (stopped, stopping)."""
        return self in (RunnerState.STOPPED, RunnerState.STOPPING)


class IllegalTransition(RuntimeError):
    """Raised when a state change is not allowed by the table."""


# event -> {from_state: to_state}
_TRANSITIONS: dict[str, dict[RunnerState, RunnerState]] = {
    # begin a launch (fresh start or crash-restart)
    "load": {
        RunnerState.UNCACHED: RunnerState.LOADING,
        RunnerState.STOPPED: RunnerState.LOADING,
        RunnerState.ERROR: RunnerState.LOADING,
        RunnerState.RUNNING: RunnerState.LOADING,   # crash-restart
        RunnerState.STOPPING: RunnerState.LOADING,  # restart during stop race
    },
    "pull": {
        RunnerState.UNCACHED: RunnerState.DOWNLOADING,
        RunnerState.STOPPED: RunnerState.DOWNLOADING,
        RunnerState.ERROR: RunnerState.DOWNLOADING,
    },
    "download_ok": {RunnerState.DOWNLOADING: RunnerState.STOPPED},
    "download_fail": {RunnerState.DOWNLOADING: RunnerState.ERROR},
    "download_cancel": {RunnerState.DOWNLOADING: RunnerState.UNCACHED},
    "ready": {RunnerState.LOADING: RunnerState.RUNNING},
    "start_fail": {RunnerState.LOADING: RunnerState.STOPPED},
    "stop": {
        RunnerState.RUNNING: RunnerState.STOPPING,
        RunnerState.LOADING: RunnerState.STOPPING,
        RunnerState.DOWNLOADING: RunnerState.STOPPING,
    },
    "stopped": {RunnerState.STOPPING: RunnerState.STOPPED},
    "eject": {
        RunnerState.STOPPED: RunnerState.UNCACHED,
        RunnerState.ERROR: RunnerState.UNCACHED,
        RunnerState.UNCACHED: RunnerState.UNCACHED,  # no cache to delete — stay
    },
    "crash_limit": {RunnerState.RUNNING: RunnerState.ERROR},
    "health_loading": {RunnerState.RUNNING: RunnerState.LOADING},
    "health_error": {RunnerState.RUNNING: RunnerState.ERROR},
}


def can(state: RunnerState, event: str) -> bool:
    return state in _TRANSITIONS.get(event, {})


def target(state: RunnerState, event: str) -> RunnerState:
    """Return the destination state for ``event`` from ``state``.

    Raises IllegalTransition if the pair is not in the table.
    """
    try:
        return _TRANSITIONS[event][state]
    except KeyError as exc:
        raise IllegalTransition(f"{event!r} not allowed from {state.name}") from exc


def transition(state: RunnerState, event: str) -> RunnerState:
    """Validate and return the next state (pure — caller assigns)."""
    return target(state, event)
