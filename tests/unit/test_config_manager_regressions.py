"""Regression tests for ModelConfigManager bug fixes.

Covers:
- get_backend() with a null ``backends:`` section (was AttributeError).
- get_model(env_vars=...) strictness always enforced (was gated on the
  manager's strict_expansion flag, contradicting the docstring).
- An instance-level ``model:`` key survives checkpoint merge.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from model_arkestra.config_manager import ModelConfigManager


def _make_cm(tmp_path: Path, content: str) -> ModelConfigManager:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(content))
    return ModelConfigManager(str(cfg))


class TestGetBackendNullSection:
    def test_null_backends_returns_none(self, tmp_path):
        cm = _make_cm(tmp_path, """\
            backends:
            models: {}
            """)
        assert cm.get_backend("rocm") is None

    def test_missing_backends_returns_none(self, tmp_path):
        cm = _make_cm(tmp_path, "models: {}\n")
        assert cm.get_backend("rocm") is None


class TestGetModelStrictness:
    """get_model(env_vars=...) must raise on unresolved placeholders
    regardless of the manager's strict_expansion flag."""

    CFG = """\
        checkpoints:
          c1:
            ref: /tmp/fake.gguf
        models:
          m1:
            checkpoint: c1
            port-arg: "${PORT}"
        """

    def test_strict_manager_raises(self, tmp_path):
        cm = _make_cm(tmp_path, self.CFG)
        with pytest.raises(ValueError, match=r"\$\{PORT\}"):
            cm.get_model("m1", env_vars={})

    def test_lenient_manager_still_raises_at_runtime(self, tmp_path):
        # strict_expansion=False relaxes boot-time expansion only; an
        # explicit runtime resolution with env_vars must still fail fast.
        cfg = tmp_path / "config.yaml"
        cfg.write_text(textwrap.dedent(self.CFG))
        cm = ModelConfigManager(str(cfg), strict_expansion=False)
        with pytest.raises(ValueError, match=r"\$\{PORT\}"):
            cm.get_model("m1", env_vars={})

    def test_resolved_placeholder_ok_when_lenient(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(textwrap.dedent(self.CFG))
        cm = ModelConfigManager(str(cfg), strict_expansion=False)
        merged = cm.get_model("m1", env_vars={"PORT": "18000"})
        assert merged["port-arg"] == "18000"

    def test_no_env_vars_leaves_placeholder(self, tmp_path):
        # Without env_vars the placeholder survives for runtime resolution.
        cm = _make_cm(tmp_path, self.CFG)
        merged = cm.get_model("m1")
        assert merged["port-arg"] == "${PORT}"


class TestInstanceModelKey:
    def test_instance_model_key_not_clobbered(self, tmp_path):
        cm = _make_cm(tmp_path, """\
            checkpoints:
              c1:
                ref: /tmp/ckpt.gguf
            models:
              m1:
                checkpoint: c1
                model: /tmp/instance-override.gguf
            """)
        merged = cm.get_model("m1")
        assert merged["model"] == "/tmp/instance-override.gguf"

    def test_checkpoint_ref_surfaces_as_model(self, tmp_path):
        cm = _make_cm(tmp_path, """\
            checkpoints:
              c1:
                ref: /tmp/ckpt.gguf
            models:
              m1:
                checkpoint: c1
            """)
        merged = cm.get_model("m1")
        assert merged["model"] == "/tmp/ckpt.gguf"
        assert "ref" not in merged


class TestNestedArgsWarning:
    """Nested ``args:`` on models/checkpoints is dead config — warn at load."""

    def test_model_nested_args_warns(self, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="model_arkestra.config_manager"):
            _make_cm(tmp_path, """\
                models:
                  m1:
                    model: /tmp/fake.gguf
                    args:
                      temp: 0.7
                """)
        assert any("'m1' has a nested 'args' block" in r.message for r in caplog.records)

    def test_checkpoint_nested_args_warns(self, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="model_arkestra.config_manager"):
            _make_cm(tmp_path, """\
                checkpoints:
                  c1:
                    ref: /tmp/fake.gguf
                    args:
                      ngl: 999
                models: {}
                """)
        assert any("'c1' has a nested 'args' block" in r.message for r in caplog.records)

    def test_flat_keys_no_warning(self, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="model_arkestra.config_manager"):
            _make_cm(tmp_path, """\
                models:
                  m1:
                    model: /tmp/fake.gguf
                    temp: 0.7
                """)
        assert not [r for r in caplog.records if "nested 'args'" in r.message]
