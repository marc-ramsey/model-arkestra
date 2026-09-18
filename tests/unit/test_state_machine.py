"""Tests for the model state machine (models/state.py).

Covers the transition table and the crash-restart race: a request arriving
while the watcher is mid-restart must not crash with IllegalTransition.
"""
from __future__ import annotations

import pytest

from model_arkestra.models.state import (
    RunnerState,
    IllegalTransition,
    can,
    transition,
)
from model_arkestra.types import _Model


class TestLoadTransitions:
    """The 'load' event — begin a launch (fresh start or crash-restart)."""

    @pytest.mark.parametrize("src", [
        RunnerState.UNCACHED,
        RunnerState.STOPPED,
        RunnerState.ERROR,
        RunnerState.RUNNING,
        RunnerState.STOPPING,
    ])
    def test_load_from_allowed_states(self, src):
        assert can(src, "load")
        assert transition(src, "load") is RunnerState.LOADING

    def test_load_idempotent_from_loading(self):
        """Crash-restart race: the watcher sets LOADING via _before_restart,
        then an in-flight request calls start() which issues 'load' again.
        Must be a legal no-op transition, not an IllegalTransition."""
        assert can(RunnerState.LOADING, "load")
        assert transition(RunnerState.LOADING, "load") is RunnerState.LOADING

    def test_load_rejected_from_downloading(self):
        assert not can(RunnerState.DOWNLOADING, "load")
        with pytest.raises(IllegalTransition):
            transition(RunnerState.DOWNLOADING, "load")


class TestModelSetState:
    """_Model.set_state — the choke point every lifecycle path uses."""

    def _ctx(self, state: RunnerState) -> _Model:
        ctx = _Model("m", 18000)
        ctx._state = state
        return ctx

    def test_set_state_load_from_loading_is_noop(self):
        """Mirrors base.py:_before_restart followed by BaseRunner.start()."""
        ctx = self._ctx(RunnerState.RUNNING)
        # Watcher crash-restart path: RUNNING -> LOADING
        ctx.set_state("load")
        assert ctx.state is RunnerState.LOADING
        # In-flight request path: LOADING -> LOADING (must not raise)
        ctx.set_state("load")
        assert ctx.state is RunnerState.LOADING

    def test_illegal_transition_still_raises(self):
        """Unrelated illegal pairs must keep raising — the table stays strict."""
        ctx = self._ctx(RunnerState.DOWNLOADING)
        with pytest.raises(IllegalTransition):
            ctx.set_state("load")


class TestCrashRestartSequence:
    """Full sequence from the log: crash -> restart -> request during LOADING."""

    def test_crash_restart_then_request(self):
        ctx = _Model("gemma", 18000)
        ctx._state = RunnerState.RUNNING

        # 1. Process SIGKILLed; watcher runs _before_restart: 'load' event
        ctx.set_state("load")
        assert ctx.state is RunnerState.LOADING

        # 2. New process also dies; second restart cycle begins
        ctx.set_state("load")
        assert ctx.state is RunnerState.LOADING

        # 3. Client request arrives, route calls start() -> set_state("load")
        ctx.set_state("load")
        assert ctx.state is RunnerState.LOADING
