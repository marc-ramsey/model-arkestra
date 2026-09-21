"""Unit tests for the layered config model.

Shipped backends.yaml (package data) is the read-only base; config.yaml
overlays it per-key (deep merge, overlay wins). All user state lives in
config.yaml.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from model_arkestra.arkestra import ModelArkestra


@pytest.fixture
def config_dir(tmp_path):
    """A config dir with a minimal config.yaml (no backends section)."""
    cfg = {
        "default": {"admin-key": "k", "ctx-size": 4096},
        "models": {"m1": {"checkpoint": "c1"}},
        "checkpoints": {"c1": {"ref": "unsloth/x:Q4"}},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path


class TestShippedBase:
    def test_shipped_backends_visible_without_user_copy(self, config_dir):
        ma = ModelArkestra(str(config_dir / "config.yaml"), ready_timeout=2)
        backends = ma.cm.data.get("backends", {})
        # Stock definitions come from package data, no user backends.yaml.
        assert "vulkan-radv" in backends
        assert backends["vulkan-radv"]["runner"] == "process"

    def test_backend_source_specs_from_package(self, config_dir):
        ma = ModelArkestra(str(config_dir / "config.yaml"), ready_timeout=2)
        be = ma.cm.data.get("backends", {})
        # Provisioning specs ship with the package, no user binaries.yaml.
        assert be["vulkan-radv"]["source"]["type"] == "remote"
        assert be["rocm-gfx1151"]["source"]["repo"] == "lemonade-sdk/llamacpp-rocm"
        assert "sources" not in ma.cm.data


class TestOverlay:
    def test_sparse_override_wins(self, config_dir):
        cfg = yaml.safe_load((config_dir / "config.yaml").read_text())
        cfg["backends"] = {"vulkan-radv": {"args": {"ngl": 7}}}
        (config_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

        ma = ModelArkestra(str(config_dir / "config.yaml"), ready_timeout=2)
        be = ma.get_backend("vulkan-radv")
        # Override wins...
        assert be["args"]["ngl"] == 7
        # ...base keys survive the deep merge.
        assert be["runner"] == "process"
        assert be["source"]["type"] == "remote"

    def test_custom_backend_declared_in_config(self, config_dir):
        cfg = yaml.safe_load((config_dir / "config.yaml").read_text())
        cfg["backends"] = {
            "default": "my-local",
            "my-local": {"runner": "process", "binary_dir": "/opt/bin",
                         "engine": "llama-cpp"},
        }
        (config_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

        ma = ModelArkestra(str(config_dir / "config.yaml"), ready_timeout=2)
        assert ma.get_backend("my-local")["binary_dir"] == "/opt/bin"

    def test_fat_override_logs_warning(self, config_dir, caplog):
        cfg = yaml.safe_load((config_dir / "config.yaml").read_text())
        cfg["backends"] = {"vulkan-radv": {"runner": "process"}}
        (config_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

        with caplog.at_level(logging.WARNING):
            ModelArkestra(str(config_dir / "config.yaml"), ready_timeout=2)
        assert any("backends.vulkan-radv" in r.message for r in caplog.records)

    def test_default_selection_no_warning(self, config_dir, caplog):
        cfg = yaml.safe_load((config_dir / "config.yaml").read_text())
        cfg["backends"] = {"default": "cpu"}
        (config_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

        with caplog.at_level(logging.WARNING):
            ModelArkestra(str(config_dir / "config.yaml"), ready_timeout=2)
        assert not any("treated as override" in r.message for r in caplog.records)


class TestSchemaRegistry:
    def test_schemas_loaded_from_package(self, config_dir):
        from model_arkestra.common import _load_schema_registry
        schemas = _load_schema_registry()
        assert "llama-cpp" in schemas.get("model-args", {})
