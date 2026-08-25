"""Tests for ONNX auxiliary models and server-side proxy routing.

Follows test_proxy.py's pattern: no subprocess servers, no port binding.
Verifies logic through direct imports, route inspection, and mocked ArkestraServer.
"""
from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest


BASE_DIR = Path(__file__).resolve().parent.parent


def _generate_wav(freq: float = 440.0, duration: float = 1.0, sr: int = 16000) -> bytes:
    """Generate a WAV file containing a pure tone."""
    num_samples = int(sr * duration)
    t = np.arange(num_samples) / sr
    waveform = (np.sin(2 * np.pi * freq * t) * 32768).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(waveform.tobytes())
    return buf.getvalue()


# ── Mel spectrogram ───────────────────────────────────────────────

class TestMelSpectrogram:
    def test_shape_and_range(self):
        from model_arkestra.onnx_server import _extract_mel_spectrogram
        wav = _generate_wav(freq=440.0, duration=1.0)
        waveform = np.frombuffer(wav[44:], dtype=np.int16).astype(np.float32) / 32768.0
        mel = _extract_mel_spectrogram(waveform, 16000)
        assert mel.shape == (1, 80, 3000)
        assert mel.dtype == np.float32
        assert mel.min() >= 0.0 and mel.max() <= 1.0

    def test_short_audio_padded(self):
        from model_arkestra.onnx_server import _extract_mel_spectrogram
        wav = _generate_wav(freq=880.0, duration=0.5)
        waveform = np.frombuffer(wav[44:], dtype=np.int16).astype(np.float32) / 32768.0
        mel = _extract_mel_spectrogram(waveform, 16000)
        assert mel.shape == (1, 80, 3000)

    def test_long_audio_trimmed(self):
        from model_arkestra.onnx_server import _extract_mel_spectrogram
        wav = _generate_wav(freq=220.0, duration=60.0)
        waveform = np.frombuffer(wav[44:], dtype=np.int16).astype(np.float32) / 32768.0
        mel = _extract_mel_spectrogram(waveform, 16000)
        assert mel.shape == (1, 80, 3000)

    def test_has_energy_for_audio(self):
        from model_arkestra.onnx_server import _extract_mel_spectrogram
        wav = _generate_wav(freq=1000.0, duration=2.0)
        waveform = np.frombuffer(wav[44:], dtype=np.int16).astype(np.float32) / 32768.0
        mel = _extract_mel_spectrogram(waveform, 16000)
        assert mel.sum() > 0

    def test_silence_is_zeros(self):
        from model_arkestra.onnx_server import _extract_mel_spectrogram
        mel = _extract_mel_spectrogram(np.zeros(16000, dtype=np.float32), 16000)
        assert mel.shape == (1, 80, 3000)
        assert mel.sum() == 0.0


# ── Tokenization helper ───────────────────────────────────────────

class TestTokenize:
    def test_returns_int64_arrays(self):
        from model_arkestra.onnx_server import _tokenize

        class T:
            def __call__(self, text, **kw):
                return {"input_ids": np.array([[101, 2054]], dtype=np.int32),
                        "attention_mask": np.array([[1, 1]], dtype=np.int32)}

        result = _tokenize("hello", T())
        assert all(v.dtype == np.int64 for v in result.values())


# ── Project config ────────────────────────────────────────────────

class TestProjectConfig:
    def test_onnx_entry_point(self):
        import tomllib
        with open(BASE_DIR / "pyproject.toml", "rb") as f:
            cfg = tomllib.load(f)
        scripts = cfg["project"]["scripts"]
        assert "arkestra-onnx" in scripts
        assert scripts["arkestra-onnx"] == "model_arkestra.onnx_server:main"

    def test_main_callable(self):
        from model_arkestra.onnx_server import main
        assert callable(main)


# ── Server proxy routing (no subprocess, route inspection only) ───

class TestProxyRouting:
    """Verify server.py registers ONNX auxiliary endpoints."""

    def _build_app(self):
        """Build ArkestraServer app without starting any process."""
        import tempfile, os, shutil
        from model_arkestra.server import ArkestraServer

        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, "test.yaml")
            shutil.copy2(str(BASE_DIR / "tests/test-admin-config.yaml"), cfg)
            app = ArkestraServer(cfg).get_app()
        return app

    def _route_paths(self):
        return {r.path for r in self._build_app().routes}

    def test_embeddings_endpoint_registered(self):
        assert "/v1/embeddings" in self._route_paths()

    def test_transcriptions_endpoint_registered(self):
        assert "/v1/audio/transcriptions" in self._route_paths()

    def test_speech_endpoint_registered(self):
        assert "/v1/audio/speech" in self._route_paths()

    def test_onnx_lifecycle_methods_exist(self):
        from model_arkestra.server import ArkestraServer
        assert hasattr(ArkestraServer, "_get_aux_model_cfg")
        assert hasattr(ArkestraServer, "_is_onnx_model")


# ── Aux model config resolution ───────────────────────────────────

class TestAuxModelConfig:
    def test_gets_aux_from_config(self):
        import tempfile, os, shutil
        from model_arkestra.server import ArkestraServer

        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, "test.yaml")
            shutil.copy2(str(BASE_DIR / "tests/test-admin-config.yaml"), cfg)
            server = ArkestraServer(cfg)

            # No aux models in test-admin-config.yaml — should return None for unknown
            result = server._get_aux_model_cfg("nonexistent")
            assert result is None


# ── ONNX model download utility ───────────────────────────────────

class TestDownloadOnnxModel:
    def test_download_into_hf_hub_cache(self):
        from model_arkestra.common import download_onnx_model, default_cache_root
        cache = default_cache_root()
        path = download_onnx_model("Xenova/bge-small-en-v1.5", cache_dir=cache)
        assert path.exists()
        assert str(path).startswith(str(cache))
        assert str(path).endswith(".onnx")

    def test_resolve_existing_path(self):
        from model_arkestra.common import resolve_onnx_model_path
        resolved = resolve_onnx_model_path(str(BASE_DIR / "tests/test-admin-config.yaml"))
        assert resolved.exists()

    def test_resolve_repo_id(self):
        from model_arkestra.common import resolve_onnx_model_path
        path = resolve_onnx_model_path("Xenova/bge-small-en-v1.5")
        assert path.exists() and str(path).endswith(".onnx")
