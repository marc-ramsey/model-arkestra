"""Unit tests for BaseModelRunner.resolve_defaults().

Tests that effective (backend_id, runner_type) resolution follows the correct
priority chain with hardwired fallbacks:

  Backend: model["backend"] → backends.default → _DEFAULT_BACKEND ("cpu")
  Runner:  backend.runner → runners.default → _DEFAULT_RUNNER ("process")
"""
from __future__ import annotations

import pytest

from model_arkestra.base import BaseModelRunner


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def backends_cfg():
    return {
        "default": "vulkan-radv",
        "rocm": {"runner": "podman", "image": "ark-llama:rocm"},
        "vulkan-radv": {"runner": "process", "image": "ark-llama:vulkan-radv"},
    }


@pytest.fixture
def runners_cfg():
    return {
        "default": "ProcessModelRunner",
    }


# ── Tests: Backend priority ────────────────────────────────────────────────


class TestBackendPriority:
    def test_model_backend_takes_precedence(self, backends_cfg):
        """model["backend"] overrides everything."""
        backend_id, _ = BaseModelRunner.resolve_defaults(
            backends_cfg, {}, {"backend": "rocm"}
        )
        assert backend_id == "rocm"

    def test_falls_back_to_config_default(self, backends_cfg):
        """No model backend → backends.default."""
        backend_id, _ = BaseModelRunner.resolve_defaults(backends_cfg, {})
        assert backend_id == "vulkan-radv"

    def test_hardwired_default_when_no_backends_section(self, runners_cfg):
        """No backends at all → _DEFAULT_BACKEND ("cpu")."""
        backend_id, _ = BaseModelRunner.resolve_defaults(None, runners_cfg)
        assert backend_id == "cpu"

    def test_hardwired_default_when_no_backends_default_key(self, runners_cfg):
        """backends dict exists but no "default" key → _DEFAULT_BACKEND."""
        backends_no_default = {
            "rocm": {"runner": "podman"},
        }
        backend_id, _ = BaseModelRunner.resolve_defaults(
            backends_no_default, runners_cfg
        )
        assert backend_id == "cpu"

    def test_empty_backend_config_uses_hardwired(self):
        """{} backends → _DEFAULT_BACKEND."""
        backend_id, _ = BaseModelRunner.resolve_defaults({}, {})
        assert backend_id == "cpu"


# ── Tests: Runner priority ─────────────────────────────────────────────────


class TestRunnerPriority:
    def test_runner_from_backend_config(self, backends_cfg):
        """backend.runner takes precedence over runners.default."""
        _, runner = BaseModelRunner.resolve_defaults(backends_cfg, {}, {"backend": "rocm"})
        assert runner == "podman"

    def test_falls_back_to_runners_default(self, backends_cfg, runners_cfg):
        """No backend.runner in resolved backend → runners.default."""
        # Use a backend that has NO runner key so it falls through
        backends_no_runner = {
            "default": "stub",
            "stub": {"image": "test"},
        }
        _, runner = BaseModelRunner.resolve_defaults(backends_no_runner, runners_cfg)
        assert runner == "ProcessModelRunner"

    def test_hardwired_runner_when_no_backend_runner_and_no_runners_section(
        self, backends_cfg
    ):
        """backend has no runner key and no runners config → _DEFAULT_RUNNER."""
        backends_no_runner = {
            "vulkan-radv": {"image": "ark-llama:vulkan-radv"},
        }
        _, runner = BaseModelRunner.resolve_defaults(backends_no_runner, {})
        assert runner == "process"

    def test_hardwired_runner_when_both_sections_missing(self):
        """No backends and no runners → both hardwired."""
        backend_id, runner = BaseModelRunner.resolve_defaults(None, None)
        assert backend_id == "cpu"
        assert runner == "process"


# ── Tests: Defaults are class constants ─────────────────────────────────────


class TestHardwiredConstants:
    def test_default_backend_is_class_constant(self):
        assert BaseModelRunner._DEFAULT_BACKEND == "cpu"

    def test_default_runner_is_class_constant(self):
        assert BaseModelRunner._DEFAULT_RUNNER == "process"
