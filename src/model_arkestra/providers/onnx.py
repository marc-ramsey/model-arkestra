"""OnnxProvider — in-process ONNX inference, one modality per model.

No transport: the InferenceSession lives in this process. The provider holds a
reference to the Model (which owns the session and related handles populated at
``start()``). Heavy numpy/sherpa work runs in a thread pool via
``asyncio.to_thread`` so the event loop is never blocked.
"""
from __future__ import annotations

import asyncio
import io
import wave
from typing import Any, Dict

from model_arkestra.providers.base import Provider


class OnnxProvider(Provider):
    """Adapts a Model's ONNX handles into the provider surface."""

    def __init__(self, model: Any):
        self.model = model

    @property
    def capabilities(self) -> frozenset:
        # Derived from the model's config at construction by the factory.
        return getattr(self.model, "_onnx_capabilities", frozenset()) or frozenset()

    def _ctx(self):
        if self.model is None:
            from model_arkestra.types import ModelNotStarted
            raise ModelNotStarted("")
        return self.model

    async def probe(self) -> bool:
        # ONNX models are ready once the session is loaded.
        return getattr(self.model, "onnx_session", None) is not None

    async def embed(self, text: str) -> Dict[str, Any]:
        """Encode text → embedding vector."""
        import numpy as np
        ctx = self._ctx()

        def _do_embed():
            from model_arkestra.onnx_server import _tokenize
            session = ctx.onnx_session
            tokenizer = getattr(ctx, 'onnx_tokenizer', None)
            tokens = _tokenize(text, tokenizer)
            output = session.run(None, tokens)

            last_hidden = output[0]
            if len(output) > 1 and output[1] is not None:
                mask = output[1].astype(np.float32)[:, :, np.newaxis]
                pooled = (last_hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
            else:
                pooled = last_hidden.mean(axis=1)

            norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
            return (pooled / norm.clip(min=1e-9)).squeeze(0).tolist()

        embedding = await asyncio.to_thread(_do_embed)
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": embedding}],
            "model": ctx.name,
            "usage": {"prompt_tokens": len(text.split()), "total_tokens": len(text.split())},
        }

    async def transcribe(self, audio_bytes: bytes, language: str = "") -> Dict[str, Any]:
        """Transcribe audio → text using Whisper ONNX model."""
        ctx = self._ctx()

        def _do_transcribe():
            import numpy as np
            from model_arkestra.onnx_server import (
                _extract_mel_spectrogram, _greedy_decode, _load_whisper_tokenizer,
            )

            waveform = np.frombuffer(audio_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0
            mel = _extract_mel_spectrogram(waveform, 16000)

            if not hasattr(ctx, 'onnx_tokenizer'):
                ctx.onnx_tokenizer = _load_whisper_tokenizer(ctx.model_path)

            session = ctx.onnx_session
            output_names = [o.name for o in session.get_outputs()]
            encoder_outputs = session.run(
                [output_names[0]],  # last_hidden_state
                {"input_features": mel.astype(np.float32)},
            )
            return _greedy_decode(
                session, ctx.model_path, ctx.onnx_tokenizer,
                mel.astype(np.float32), encoder_outputs[0],
            ).strip()

        text = await asyncio.to_thread(_do_transcribe)
        return {"text": text, "language": language or "en"}

    async def synthesize(self, text: str, voice: str = "", speed: float = 1.0) -> bytes:
        """Generate speech from text using Kokoro ONNX TTS model."""
        ctx = self._ctx()

        def _do_synthesize():
            samples, sr = ctx.kokero_model.create(
                text, voice=voice or ctx.g2p_lang, speed=speed,
            )
            audio_int16 = (samples * 32767).astype("int16")
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_int16.tobytes())
            return buf.getvalue()

        return await asyncio.to_thread(_do_synthesize)

    async def stream_asr(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Streaming ASR with partial results via sherpa-ai paraformer."""
        import librosa
        ctx = self._ctx()

        def _do_stream():
            from sherpa_onnx import (
                OnlineRecognizer, OnlineStream, OfflineModelConfig, OnlineRecognizerConfig,
            )
            samples = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)[0]
            stream = OnlineStream()
            stream.accept_waveform(16000, samples.tolist())

            if not hasattr(ctx, '_sherpa_rec'):
                ctx._sherpa_rec = OnlineRecognizer(
                    config=OnlineRecognizerConfig(model_config=OfflineModelConfig())
                )
            rec = ctx._sherpa_rec

            partial_text = ""
            while rec.decode_stream(stream):
                result = stream.get_result()
                if result.text:
                    partial_text = result.text

            return {
                "partial": partial_text,
                "final": stream.final_result.text if hasattr(stream, 'final_result') else partial_text,
            }

        return await asyncio.to_thread(_do_stream)

    async def stream_tts(self, text: str) -> bytes:
        """Piper TTS — generates complete WAV in one call."""
        ctx = self._ctx()

        def _do_synthesize():
            samples, sr = ctx.piper_voice.synthesize(text)
            audio_int16 = (samples * 32767).astype("int16")
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_int16.tobytes())
            return buf.getvalue()

        return await asyncio.to_thread(_do_synthesize)
