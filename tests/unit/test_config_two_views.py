"""Tests for ModelConfigManager's two-view model.

self.data   — merged view (shipped base + user overlay), macros expanded.
self._config — raw user file, placeholders intact, the only export target.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from model_arkestra.config_manager import ModelConfigManager


def _make_cm(tmp_path: Path, content: str) -> ModelConfigManager:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(content))
    return ModelConfigManager(str(cfg))


class TestMergedView:
    def test_shipped_base_present(self, tmp_path):
        cm = _make_cm(tmp_path, "models: {}\n")
        # Stock backend visible in the merged view...
        assert "vulkan-radv" in cm.data["backends"]
        # ...but absent from the user view.
        assert "backends" not in cm._config.data


class TestUserViewRaw:
    def test_placeholders_not_expanded_in_user_view(self, tmp_path):
        cm = _make_cm(tmp_path, """\
            macros:
              N: 42
            models:
              m1:
                val: ${N}
            """)
        # Merged view is expanded (macro values render as strings)...
        assert cm.data["models"]["m1"]["val"] == "42"
        # ...user view keeps the placeholder.
        assert cm._config.data["models"]["m1"]["val"] == "${N}"


class TestMutatorsPropagate:
    CFG = "models:\n  m1:\n    a: 1\n"

    def test_setitem_writes_both_views(self, tmp_path):
        cm = _make_cm(tmp_path, self.CFG)
        cm["models/m1/a"] = 2
        assert cm.data["models"]["m1"]["a"] == 2
        assert cm._config.data["models"]["m1"]["a"] == 2

    def test_merge_writes_both_views(self, tmp_path):
        cm = _make_cm(tmp_path, self.CFG)
        cm.merge({"models": {"m2": {"b": 3}}})
        assert cm.data["models"]["m2"] == {"b": 3}
        assert cm._config.data["models"]["m2"] == {"b": 3}

    def test_delitem_writes_both_views(self, tmp_path):
        cm = _make_cm(tmp_path, self.CFG)
        del cm["models"]
        assert "models" not in cm.data
        assert "models" not in cm._config.data

    def test_delitem_user_only_key(self, tmp_path):
        # Key absent from the merged view's user portion but present in both —
        # top-level delete applies to each view independently.
        cm = _make_cm(tmp_path, "models: {}\nmacros: {N: 1}\n")
        del cm["macros"]
        assert "macros" not in cm._config.data


class TestExportUserOnly:
    def test_export_omits_shipped_base(self, tmp_path):
        cm = _make_cm(tmp_path, "models:\n  m1: {}\n")
        out = tmp_path / "out.yaml"
        cm.export(str(out))
        dumped = yaml.safe_load(out.read_text())
        # User content round-trips...
        assert dumped["models"] == {"m1": {}}
        # ...but the shipped base never leaks into the user file.
        assert "backends" not in dumped

    def test_export_keeps_placeholders(self, tmp_path):
        cm = _make_cm(tmp_path, "models:\n  m1:\n    val: ${UNRESOLVED}\n")
        out = tmp_path / "out.yaml"
        cm.export(str(out))
        dumped = yaml.safe_load(out.read_text())
        assert dumped["models"]["m1"]["val"] == "${UNRESOLVED}"

    def test_export_json(self, tmp_path):
        import json
        cm = _make_cm(tmp_path, "models: {}\n")
        out = tmp_path / "out.json"
        cm.export(str(out), fmt="json")
        assert json.loads(out.read_text())["models"] == {}

    def test_export_rejects_bad_fmt(self, tmp_path):
        cm = _make_cm(tmp_path, "models: {}\n")
        with pytest.raises(ValueError):
            cm.export(str(tmp_path / "x.xml"), fmt="xml")
