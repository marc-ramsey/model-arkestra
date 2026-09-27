"""Tests for capability derivation — vision via mmproj, ONNX, remote."""
from __future__ import annotations

from model_arkestra.models.capabilities import derive_capabilities
from model_arkestra.types import _Model


class TestLlamaDefaults:
    def test_default(self):
        caps = derive_capabilities({}, {})
        assert caps == frozenset({"chat", "embed"})

    def test_embed_only_override(self):
        caps = derive_capabilities({"capabilities": ["embed"]}, {})
        assert caps == frozenset({"embed"})


class TestVision:
    """A llama model with an mmproj sidecar gains the vision capability."""

    def test_mmproj_on_model(self):
        caps = derive_capabilities({}, {}, mmproj="mmproj-F16.gguf")
        assert caps == frozenset({"chat", "embed", "vision"})

    def test_empty_mmproj(self):
        caps = derive_capabilities({}, {}, mmproj="")
        assert caps == frozenset({"chat", "embed"})

    def test_mmproj_with_embed_only_override(self):
        # Embedding-only models can't be vision.
        caps = derive_capabilities({"capabilities": ["embed"]}, {},
                                   mmproj="mmproj.gguf")
        assert caps == frozenset({"embed"})


class TestOnnx:
    def test_whisper(self):
        caps = derive_capabilities({"runner": "onnx", "type": "whisper"}, {})
        assert caps == frozenset({"asr"})

    def test_tts(self):
        caps = derive_capabilities({"runner": "onnx", "type": "tts"}, {})
        assert caps == frozenset({"tts"})

    def test_embedding(self):
        caps = derive_capabilities({"runner": "onnx", "type": "embedding"}, {})
        assert caps == frozenset({"embed"})


class TestRemote:
    def test_remote_empty(self):
        caps = derive_capabilities({}, {"runner": "remote"})
        assert caps == frozenset()


class TestSdcppEngine:
    """A sdcpp engine model advertises image-gen (not chat/embed)."""

    def test_sdcpp_engine(self):
        caps = derive_capabilities({}, {}, engine="sdcpp")
        assert caps == frozenset({"image-gen"})

    def test_llama_engine_unchanged(self):
        caps = derive_capabilities({}, {}, engine="llama-cpp")
        assert caps == frozenset({"chat", "embed"})


class TestModelVisionMmprojLookup:
    """_Model resolves mmproj from model entry or backend args."""

    def test_from_model_entry(self):
        m = _Model("m")
        m._model_cfg = {"mmproj": "mm.gguf"}
        m._backend_cfg = {}
        assert "vision" in m.capabilities

    def test_no_mmproj(self):
        m = _Model("m")
        m._model_cfg = {}
        m._backend_cfg = {"args": {}}
        assert "vision" not in m.capabilities
