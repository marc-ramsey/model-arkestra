"""Tests for unified argument merging: config args + inference kwargs → single CLI list.

Every assertion checks the end-to-end output of build_model_args() — one function,
one conversion path, no _build_cmd_line detour.
"""
from __future__ import annotations
import pytest
import tempfile
import os
from model_arkestra.common import build_model_args
from llm_config_manager.config_manager import ConfigManager


# ── helpers ───────────────────────────────────────────────────────────

def _make_cm(yaml_content: str) -> ConfigManager:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        os.write(fd, yaml_content.encode())
        os.close(fd)
        return ConfigManager(path)
    except Exception:
        os.close(fd)
        raise


SAMPLE_CONFIG = """\
macros:
  ctx-size: 8192

defaults:
  jinja: on
  n-gpu-layers: 99

backends:
  default: vulkan
  vulkan:
    args:
      model: ${CHECKPOINT}
      port: "${PORT}"

models:
  small-model:
    checkpoint: fake/model:Q4
    args:
      temp: 0.7
      top-p: 0.95
"""


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture()
def cm():
    return _make_cm(SAMPLE_CONFIG)


# ── tests ─────────────────────────────────────────────────────────────

class TestConfigDefaultsCascade:
    """Default → backend → model args cascade (last-wins)."""

    def test_defaults_are_present(self, cm):
        result = build_model_args(cm, "small-model", env_vars={"PORT": "18000"})
        assert "--jinja" in result[0]
        assert "--n-gpu-layers" in result[0]

    def test_model_args_override_defaults(self, cm):
        result = build_model_args(cm, "small-model", env_vars={"PORT": "18000"})
        idx = result[0].index("--temp")
        assert result[0][idx + 1] == "0.7"


class TestBackendMacroResolution:
    """Backend args with ${CHECKPOINT} and ${PORT} macros must resolve."""

    def test_checkpoint_macro_resolved(self, cm):
        result = build_model_args(cm, "small-model", env_vars={"PORT": "18000"})
        idx = result[0].index("--model")
        assert result[0][idx + 1] == "fake/model:Q4"

    def test_port_macro_resolved(self, cm):
        result = build_model_args(cm, "small-model", env_vars={"PORT": "18000"})
        idx = result[0].index("--port")
        assert result[0][idx + 1] == "18000"


class TestInferenceKwargsMerge:
    """Inference kwargs are merged into model args (last-wins) and converted to CLI."""

    def test_single_kwarg_converted(self, cm):
        result = build_model_args(
            cm, "small-model", env_vars={"PORT": "18000"},
            inference_kwargs={"temp": 1.0},
        )
        idx = result[0].index("--temp")
        assert result[0][idx + 1] == "1.0"

    def test_kwarg_overrides_model_default(self, cm):
        # Model args: temp: 0.7. Inference kwarg overrides to 1.0.
        result = build_model_args(
            cm, "small-model", env_vars={"PORT": "18000"},
            inference_kwargs={"temp": 1.0},
        )
        idx = result[0].index("--temp")
        assert result[0][idx + 1] == "1.0"

    def test_kwarg_missing_from_config_appears(self, cm):
        result = build_model_args(
            cm, "small-model", env_vars={"PORT": "18000"},
            inference_kwargs={"presence-penalty": 1.5},
        )
        idx = result[0].index("--presence-penalty")
        assert result[0][idx + 1] == "1.5"

    def test_bool_kwarg_true_is_presence_only(self, cm):
        result = build_model_args(
            cm, "small-model", env_vars={"PORT": "18000"},
            inference_kwargs={"flash-attn": True},
        )
        assert "--flash-attn" in result[0]
        idx = result[0].index("--flash-attn")
        if idx + 1 < len(result[0]):
            assert result[idx + 1] != "True"

    def test_bool_kwarg_false_is_omitted(self, cm):
        result = build_model_args(
            cm, "small-model", env_vars={"PORT": "18000"},
            inference_kwargs={"no-mmap": False},
        )
        assert "--no-mmap" not in result[0]

    def test_infra_keys_skipped(self, cm):
        # backend and checkpoint keys should NOT become CLI flags.
        result = build_model_args(
            cm, "small-model", env_vars={"PORT": "18000"},
            inference_kwargs={"backend": "rocm", "checkpoint": "other:Q4"},
        )
        assert "--backend" not in result[0]

    def test_kwarg_string_with_macro_resolved(self, cm):
        """String values in inference kwargs containing ${...} must be macro-resolved."""
        result = build_model_args(
            cm, "small-model", env_vars={"PORT": "18000"},
            inference_kwargs={"ctx-size": "${ctx-size}"},
        )
        idx = result[0].index("--ctx-size")
        assert result[0][idx + 1] == "8192"


class TestBoolInConfigDict:
    """Boolean values in YAML config dicts must become presence-only flags."""

    def test_jinja_on_becomes_presence_flag(self, cm):
        result = build_model_args(cm, "small-model", env_vars={"PORT": "18000"})
        assert "--jinja" in result[0]
        idx = result[0].index("--jinja")
        # Next element (if any) must not be "True" — True → --flag only
        if idx + 1 < len(result[0]):
            assert result[idx + 1] != "True"

    def test_n_gpu_layers_scalar_keeps_value(self, cm):
        """Non-boolean values still get their scalar value."""
        result = build_model_args(cm, "small-model", env_vars={"PORT": "18000"})
        idx = result[0].index("--n-gpu-layers")
        assert result[0][idx + 1] == "99"


class TestSinglePath:
    """Verify there is only ONE conversion path through build_model_args()."""

    def test_all_args_in_single_list(self, cm):
        """Config defaults, backend args, model args, and inference kwargs all appear in one list."""
        result = build_model_args(
            cm, "small-model", env_vars={"PORT": "18000"},
            inference_kwargs={"top-p": 0.99, "presence-penalty": -0.5},
        )
        # Defaults: jinja (True → --jinja), n-gpu-layers (99 → --n-gpu-layers 99)
        assert "--jinja" in result[0]
        assert "--n-gpu-layers" in result[0]
        # Backend: model, port
        assert "--model" in result[0]
        assert "--port" in result[0]
        # Model + kwarg: temp (from model), top-p (kwarg overrides 0.95)
        idx = result[0].index("--top-p")
        assert result[0][idx + 1] == "0.99"
        # Kwarg only: presence-penalty
        assert "--presence-penalty" in result[0]
