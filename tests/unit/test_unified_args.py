"""Tests for unified argument merging: config args + inference kwargs → flat dict.

``build_model_args()`` now returns only the merged data dict — no CLI conversion,
no defaults cascade, no backend args.  The engine layer handles all CLI generation.
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
defaults:
  jinja: on
  n-gpu-layers: 99

models:
  small-model:
    repo: hugging-face
    model: fake/model:Q4
    args:
      temp: 0.7
      top-p: 0.95
      flash-attn: true
"""


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture()
def cm():
    return _make_cm(SAMPLE_CONFIG)


# ── tests ─────────────────────────────────────────────────────────────

class TestConfigDefaultsNotMerged:
    """build_model_args returns ONLY model args + inference kwargs.
    Defaults and backend args are resolved by the engine layer.
    """

    def test_no_defaults_in_merged_dict(self, cm):
        """Default section keys must NOT appear in merged dict."""
        result = build_model_args(cm, "small-model")
        assert result is not None
        assert "jinja" not in result
        assert "n-gpu-layers" not in result

    def test_model_args_present(self, cm):
        """Model args from config must appear in merged dict."""
        result = build_model_args(cm, "small-model")
        assert result is not None
        assert result["temp"] == 0.7
        assert result["top-p"] == 0.95


class TestInferenceKwargsMerge:
    """Inference kwargs merge into model args with last-wins semantics."""

    def test_single_kwarg_overrides_config(self, cm):
        result = build_model_args(
            cm, "small-model",
            inference_kwargs={"temp": 1.0},
        )
        assert result["temp"] == 1.0

    def test_new_kwarg_appears(self, cm):
        """Inference kwarg not in config should still appear."""
        result = build_model_args(
            cm, "small-model",
            inference_kwargs={"presence-penalty": 1.5},
        )
        assert result["presence-penalty"] == 1.5

    def test_kwarg_overrides_bool_config(self, cm):
        """Inference kwarg overrides config value including booleans."""
        result = build_model_args(
            cm, "small-model",
            inference_kwargs={"flash-attn": False},
        )
        assert result["flash-attn"] is False

    def test_infra_keys_dropped(self, cm):
        """Infra metadata keys must NOT appear in merged dict."""
        result = build_model_args(
            cm, "small-model",
            inference_kwargs={"backend": "rocm", "model": "other:Q4"},
        )
        assert "backend" not in result


class TestBoolInConfigDict:
    """Boolean values from config must be preserved as Python bools."""

    def test_bool_true_preserved(self, cm):
        result = build_model_args(cm, "small-model")
        assert result is not None
        assert result["flash-attn"] is True  # Python True, not string

    def test_non_bool_values_kept(self, cm):
        """Non-boolean values retain their type."""
        result = build_model_args(cm, "small-model")
        assert result is not None
        assert result["temp"] == 0.7
        assert isinstance(result["temp"], float)


class TestUnknownModelReturnsNone:
    """Unknown model names must return None."""

    def test_nonexistent_model(self, cm):
        assert build_model_args(cm, "ghost-model") is None


class TestMissingModelsSection:
    """Config without models section should be handled gracefully."""

    def test_no_models_section_returns_none(self):
        empty_cm = _make_cm("defaults:\n  jinja: on\n")
        result = build_model_args(empty_cm, "small-model")
        assert result is None


class TestEngineCLIConversion:
    """Verify LlamaCppEngine.build_cli_args() converts the dict properly."""

    def test_dict_converts_to_cli(self):
        from model_arkestra.llama_cpp import LlamaCppEngine
        merged = {"temp": 0.7, "top-p": 0.95, "jinja": True, "ngl": "33"}
        cli = LlamaCppEngine.build_cli_args(merged, port=18000)

        assert "--temp" in cli
        assert "0.7" in cli
        assert "--top-p" in cli
        assert "0.95" in cli
        assert "--jinja" in cli
        assert "-ngl" in cli
        assert "33" in cli
        assert "--port" in cli
        assert "18000" in cli

    def test_bool_true_is_presence_only(self):
        from model_arkestra.llama_cpp import LlamaCppEngine
        merged = {"flash-attn": True, "jinja": True}
        cli = LlamaCppEngine.build_cli_args(merged, port=18000)

        assert "--flash-attn" in cli
        idx = cli.index("--flash-attn")
        # Next element should not be "True" — presence-only flag.
        if idx + 1 < len(cli):
            assert cli[idx + 1] != "True"

    def test_bool_false_is_omitted(self):
        from model_arkestra.llama_cpp import LlamaCppEngine
        merged = {"flash-attn": False}
        cli = LlamaCppEngine.build_cli_args(merged, port=18000)

        assert "--flash-attn" not in cli

    def test_metadata_keys_skipped(self):
        from model_arkestra.llama_cpp import LlamaCppEngine
        # port is injected — skipped.
        merged = {"temp": 0.7, "port": 9999}
        cli = LlamaCppEngine.build_cli_args(merged, port=18000)

        assert "-hf" not in cli
        assert "--port" in cli
        assert "18000" in cli

    def test_hf_model_emitted(self):
        from model_arkestra.llama_cpp import LlamaCppEngine
        merged = {"model": "unsloth/gemma-4-E2B-it-GGUF:Q4_K_XL", "repo": "hf", "temp": 0.7}
        cli = LlamaCppEngine.build_cli_args(merged, port=12000)

        assert "-hf" in cli
        idx = cli.index("-hf")
        assert cli[idx + 1] == "unsloth/gemma-4-E2B-it-GGUF:Q4_K_XL"
        assert "--alias" in cli
        assert "--mmproj" not in cli
