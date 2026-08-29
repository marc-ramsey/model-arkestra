"""Unit tests for ArkestraAdmin._args_schema().

Resolution chain tested:
  1. Model-level args_schema override (highest)
  2. Backend.engine → schemas["model-args"][engine_name]
  3. engines.default-engine fallback (from backends.yaml)
  4. No engine → empty schema
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent.parent


def _resolve_backend_identity(cm, cfg, name):
    """No-op: backend_id = cfg['backend'] or None."""
    return cfg.get("backend")


def _make_admin(backends_cfg, models_data, schemas_cfg):
    from model_arkestra.admin import ArkestraAdmin

    admin = ArkestraAdmin.__new__(ArkestraAdmin)
    admin.server = MagicMock()
    # Full merged config: backends + engines at top level, models nested
    cm_data = {"models": models_data}
    if isinstance(backends_cfg, dict):
        cm_data.update({k: v for k, v in backends_cfg.items() if k not in ("models",)})
    admin.server._arkestra.cm.data = cm_data
    admin._schemas = schemas_cfg
    return admin


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def full_schemas():
    with open(ROOT / "schemas.yaml") as f:
        return __import__("yaml").safe_load(f)


@pytest.fixture
def full_backends():
    with open(ROOT / "backends.yaml") as f:
        return __import__("yaml").safe_load(f)


# ── Tests: Model override ──────────────────────────────────────────────────


class TestModelOverride:
    def test_model_args_schema_takes_priority(self):
        models = {
            "m": {"checkpoint": "x", "args": {"my-float": 1.5},
                  "args_schema": {"my-float": {"type": "float"}}},
        }
        admin = _make_admin({}, models, {})

        result = admin._args_schema("m")

        assert result == {"my-float": {"type": "float"}}


# ── Tests: Backend engine → schema lookup ──────────────────────────────────


class TestBackendEngineLookup:
    def test_backend_engine_resolved(self, full_backends, full_schemas):
        backends = {**full_backends, "backends": {"vulkan-radv": {"engine": "llama-cpp"}}}
        models = {
            "m": {"checkpoint": "x", "backend": "vulkan-radv", "args": {"temp": 1.0}},
        }
        admin = _make_admin(backends, models, full_schemas)

        with patch("model_arkestra.admin._resolve_backend", side_effect=_resolve_backend_identity):
            result = admin._args_schema("m")

        assert "temp" in result
        assert result["temp"]["type"] == "float"

    def test_extra_keys_not_in_model_args_ignored(self, full_backends, full_schemas):
        backends = {**full_backends, "backends": {"vulkan-radv": {"engine": "llama-cpp"}}}
        models = {
            "m": {"checkpoint": "x", "backend": "vulkan-radv", "args": {"temp": 1.0}},
        }
        admin = _make_admin(backends, models, full_schemas)

        with patch("model_arkestra.admin._resolve_backend", side_effect=_resolve_backend_identity):
            result = admin._args_schema("m")

        assert "temp" in result


# ── Tests: Default engine fallback ─────────────────────────────────────────


class TestDefaultEngine:
    def test_falls_back_to_default_engine(self, full_backends, full_schemas):
        backends = {**full_backends, "backends": {"vulkan-radv": {"runner": "process"}}}  # no engine
        models = {
            "m": {"checkpoint": "x", "backend": "vulkan-radv", "args": {"temp": 1.0}},
        }
        admin = _make_admin(backends, models, full_schemas)

        with patch("model_arkestra.admin._resolve_backend", side_effect=_resolve_backend_identity):
            result = admin._args_schema("m")

        assert "temp" in result


# ── Tests: No engine ───────────────────────────────────────────────────────


class TestNoEngine:
    def test_no_engine_returns_empty(self):
        backends = {"backends": {"onnx": {"runner": "onnx"}}}
        models = {
            "m": {"checkpoint": "x", "backend": "onnx", "args": {"something": 1}},
        }
        admin = _make_admin(backends, models, {})

        with patch("model_arkestra.admin._resolve_backend", side_effect=_resolve_backend_identity):
            result = admin._args_schema("m")

        assert result == {}
