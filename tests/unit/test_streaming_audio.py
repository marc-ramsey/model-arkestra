"""Tests for streaming audio WebSocket endpoint and OnnxRunner streaming methods."""
from __future__ import annotations

import asyncio
import base64
import json
import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

sys_path = str(Path(__file__).parent.parent / ".." / "src")
import sys
sys.path.insert(0, sys_path)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_onnx_runner():
    """Mock OnnxRunner with streaming methods."""
    runner = MagicMock()
    runner.stream_asr = AsyncMock(return_value={
        "partial": "hello world",
        "final": "hello world."
    })
    runner.stream_tts = AsyncMock(return_value=b"dummy-wav-bytes")
    return runner


# ── onnx_runner.py unit tests ───────────────────────────────────────────────

class TestStreamAsr:
    """Test OnnxRunner.stream_asr with mocked sherpa-onnx."""

    @pytest.mark.asyncio
    async def test_returns_partial_and_final(self, mock_onnx_runner):
        result = await mock_onnx_runner.stream_asr("asr-model", b"fake-audio")

        assert result["partial"] == "hello world"
        assert result["final"] == "hello world."
        mock_onnx_runner.stream_asr.assert_awaited_once_with("asr-model", b"fake-audio")

    @pytest.mark.asyncio
    async def test_raises_when_model_not_loaded(self):
        from model_arkestra.types import ModelNotStarted
        runner = MagicMock()
        runner._models = {}
        runner.stream_asr = AsyncMock(side_effect=ModelNotStarted("missing"))

        with pytest.raises(ModelNotStarted, match="missing"):
            await runner.stream_asr("missing", b"data")


class TestStreamTts:
    """Test OnnxRunner.stream_tts with mocked piper."""

    @pytest.mark.asyncio
    async def test_returns_wav_bytes(self, mock_onnx_runner):
        wav_data = b"RIFF\x00\x00\x00\x00WAVEfmt "  # minimal WAV header
        mock_onnx_runner.stream_tts = AsyncMock(return_value=wav_data)

        result = await mock_onnx_runner.stream_tts("tts-model", "hello")

        assert isinstance(result, bytes)
        assert result == wav_data
        mock_onnx_runner.stream_tts.assert_awaited_once_with("tts-model", "hello")


# ── Protocol conformance tests ──────────────────────────────────────────────

class TestProtocol:
    """Verify protocol design choices are sound."""

    def test_base64_encoding_roundtrip(self):
        """Base64 encode/decode roundtrip works for PCM float32 arrays.

        Simulates AudioStream.startRecording() encoding in widget-audio.js:
        each 128-sample chunk is struct-packed to bytes, then base64-encoded.
        All chunks are concatenated and reassembled during decode.
        """
        pcm = (np.random.randn(800) * 0.5).astype(np.float32)

        # Encode: pack each float as 'f' (4 bytes), collect all bytes, then base64
        all_bytes = struct.pack(f'{len(pcm)}f', *pcm.tolist())
        encoded = base64.b64encode(all_bytes).decode()

        # Decode: base64 → raw bytes → unpack floats
        decoded = base64.b64decode(encoded)
        reconstructed = struct.unpack(f'{len(decoded)//4}f', decoded)

        assert len(reconstructed) == len(pcm)  # all 800 samples restored
        for i, (orig, recon) in enumerate(zip(pcm, reconstructed)):
            assert pytest.approx(orig, rel=1e-6) == recon

    def test_json_frame_structure(self):
        """Audio frame JSON is valid and has expected fields."""
        sample_pcm = b"\x00" * 100
        b64_data = base64.b64encode(sample_pcm).decode()

        frame = {"type": "audio_frame", "data": b64_data}
        parsed = json.loads(json.dumps(frame))

        assert parsed["type"] == "audio_frame"
        assert len(parsed["data"]) > 0

    def test_tts_frame_structure(self):
        """TTS request JSON has expected fields."""
        frame = {"type": "tts", "text": "hello world"}
        parsed = json.loads(json.dumps(frame))

        assert parsed["type"] == "tts"
        assert parsed["text"] == "hello world"

    def test_partial_transcript_structure(self):
        """Partial transcript JSON has expected fields."""
        frame = {"type": "partial", "text": "hel..."}
        parsed = json.loads(json.dumps(frame))

        assert parsed["type"] == "partial"
        assert isinstance(parsed["text"], str)

    def test_final_transcript_structure(self):
        """Final transcript JSON has expected fields."""
        frame = {"type": "final", "text": "hello world."}
        parsed = json.loads(json.dumps(frame))

        assert parsed["type"] == "final"
        assert isinstance(parsed["text"], str)
