"""Case-insensitive model-name resolution in Registry (user-boundary only).

Config keys stay byte-exact; folding exists solely so a name typed by hand
(any casing) resolves to the registered context. The folded index maps
casefold(name) → first-seen config spelling, written at register() time —
lookups are single hash hits with no scanning.
"""
from __future__ import annotations

import pytest

from model_arkestra.models.registry import Registry


class _FakeCM:
    """Minimal cm stub — registry only touches get()/get_model()/get_backend()."""

    def __init__(self):
        # ``data`` is read by Registry for the local cluster admin key.
        self.data: dict = {}
        self._data = {"default": {}, "clusters": {}}

    def get(self, path, default=None):  # mirror ConfigManager: '/'-separated walk
        value = self._data
        for seg in path.split("/"):
            if not isinstance(value, dict) or seg not in value:
                return default
            value = value[seg]
        return value

    def get_model(self, name):
        models = self._data.get("models") or {}
        cfg = models.get(name)
        return dict(cfg) if isinstance(cfg, dict) else None


@pytest.fixture()
def registry():
    reg = Registry(_FakeCM())
    # Two model names sharing one checkpoint; plus a second standalone.
    class Ctx:  # opaque stand-in for _Model; port=None → fresh allocation path
        port = None

    ctx_a, ctx_b = Ctx(), Ctx()
    reg.register("qwen3-4b", ctx_a, ["qwen3-4b", "QWEN3-4B-INSTRUCT"])
    reg.register("other-model", ctx_b, ["other-model"])
    return reg


def test_exact_case_resolves(registry):
    """Regression guard: the common correct-case path goes through the index too."""
    assert registry.get_checkpoint_id("qwen3-4b") == "qwen3-4b"
    assert registry.get_context("other-model").port is None  # same ctx object


def test_wrong_case_resolves_to_same_context(registry):
    """User-typed casing differences must not change resolution."""
    for variant in ("QWEN3-4B", "qwen3-4b".upper(), "QwEn3-4b"):
        assert registry.get_checkpoint_id(variant) == "qwen3-4b"


def test_alias_spelling_resolves(registry):
    """A second registered spelling folding identically aliases the first's checkpoint."""
    # Both spellings were registered; each exact key still exists in _models.
    assert registry._folded_to_name["qwen3-4b".casefold()] == "qwen3-4b"  # first wins


def test_unknown_name_any_casing_returns_none(registry):
    assert registry.get_checkpoint_id("nope") is None
    assert registry.get_context("NOPE-MODEL") is None
