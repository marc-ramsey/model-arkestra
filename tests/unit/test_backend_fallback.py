"""Tests for backend runtime fallback (warn + detected default).

Covers the Slice 4 change: a missing GPU runtime no longer hard-errors at
startup. Instead the effective default backend is recorded in a runtime-only
override on the config manager, leaving the user's config file untouched.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from model_arkestra.config_manager import ModelConfigManager


def _make_cm(tmp_path: Path) -> ModelConfigManager:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(
        """\
        default:
          model-start-port: 18000
        backends:
          default: rocm
        models: {}
        """
    ))
    return ModelConfigManager(str(cfg))


class TestEffectiveDefaultBackend:
    def test_returns_configured_default_when_no_override(self, tmp_path):
        cm = _make_cm(tmp_path)
        assert cm.effective_default_backend() == "rocm"

    def test_prefers_runtime_override(self, tmp_path):
        cm = _make_cm(tmp_path)
        cm._effective_default_backend = "cpu"
        assert cm.effective_default_backend() == "cpu"
        # The configured value is untouched.
        assert cm.data["backends"]["default"] == "rocm"

    def test_none_when_no_backends(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("models: {}\n")
        cm = ModelConfigManager(str(cfg))
        assert cm.effective_default_backend() is None


class TestValidateBackendRuntimeFallback:
    def test_missing_runtime_sets_override_and_keeps_config(self, tmp_path, monkeypatch):
        from model_arkestra import arkestra as ark_mod

        cfg = tmp_path / "config.yaml"
        cfg.write_text(textwrap.dedent(
            """\
            default:
              model-start-port: 18000
            backends:
              default: rocm
            models: {}
            """
        ))
        cm = ModelConfigManager(str(cfg))

        # A minimal stand-in exposing only what _validate_backend_runtime reads.
        class FakeArkestra:
            _cm = cm
            device_detection = {"recommendation": ("cpu", "no ROCm runtime")}

        monkeypatch.setattr(ark_mod, "has_rocm", lambda: False)

        # Bind the real method to the fake so we exercise production logic.
        ark_mod.ModelArkestra._validate_backend_runtime(FakeArkestra())

        assert cm._effective_default_backend == "cpu"
        assert cm.effective_default_backend() == "cpu"
        # Config file value unchanged — a save would not persist the fallback.
        assert cm.data["backends"]["default"] == "rocm"

    def test_present_runtime_leaves_override_unset(self, tmp_path, monkeypatch):
        from model_arkestra import arkestra as ark_mod

        cfg = tmp_path / "config.yaml"
        cfg.write_text(textwrap.dedent(
            """\
            backends:
              default: rocm
            models: {}
            """
        ))
        cm = ModelConfigManager(str(cfg))

        class FakeArkestra:
            _cm = cm
            device_detection = {"recommendation": ("rocm", "ROCm detected")}

        monkeypatch.setattr(ark_mod, "has_rocm", lambda: True)

        ark_mod.ModelArkestra._validate_backend_runtime(FakeArkestra())

        assert cm._effective_default_backend is None
        assert cm.effective_default_backend() == "rocm"
