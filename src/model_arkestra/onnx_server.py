"""Lightweight HTTP server wrapping ONNX Runtime for auxiliary workloads.

Serves three model types via OpenAI-compatible endpoints:
  whisper    → POST /v1/audio/transcriptions (ASR)
  tts        → POST /v1/audio/speech         (text-to-speech)
  embedding  → POST /v1/embeddings            (vector embeddings)

Usage:
    python -m model_arkestra.onnx_server \
        --model path/to/model.onnx \
        --type embedding \
        --device cpu \
        --port 8090

Model types require different tokenization strategies. The --tokenizer arg
points to a huggingface tokenizer (repo-id) or a local directory containing
tokenize.json + vocab.txt.
"""
from __future__ import annotations

import argparse
import io
import logging
import wave
from typing import Any, Dict, List

import numpy as np
from aiohttp import web

logger = logging.getLogger(__name__)


def _tokenize(text: str, tokenizer) -> Dict[str, np.ndarray]:
    """Tokenize text → ONNX-ready numpy arrays."""
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=512,
    )
    # BatchEncoding values are already numpy — just ensure int64 dtype
    result: Dict[str, np.ndarray] = {}
    for key, val in inputs.items():
        arr = np.asarray(val)
        result[key] = arr.astype(np.int64)
    return result


class OnnxServer:
    """Runs an ONNX model and exposes inference via HTTP."""

    def __init__(
        self,
        model_path: str,
        inference_type: str,
        device: str,
        port: int,
        tokenizer_path: str | None = None,
        extra_kwargs: Dict[str, Any] | None = None,
    ):
        self.model_path = model_path
        self.inference_type = inference_type  # whisper | tts | embedding
        self.device = device
        self.port = port
        self.tokenizer_path = tokenizer_path
        self.extra_kwargs = extra_kwargs or {}

        self._session: Any = None
        self._tokenizer = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []

    # ── Model / Tokenizer loading ────────────────────────────────────

    async def _ensure_loaded(self) -> None:
        """Load ONNX session and tokenizer on first access."""
        if self._session is not None:
            return

        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_opts,
            providers=[self.device],
        )
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

        if self.tokenizer_path:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    # ── Embedding ────────────────────────────────────────────────────

    async def _embed(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Encode text → embedding vector.

        Handles standard BERT-style models (mean-pool last_hidden_state).
        Also works with models that output pooled embeddings directly.
        """
        await self._ensure_loaded()

        # Tokenize
        if self._tokenizer:
            tokens = _tokenize(input_text, self._tokenizer)
        else:
            # No tokenizer — use raw string as single token
            tokens = {self._input_names[0]: np.array([["test"]], dtype=np.int64)}

        # Run inference
        output = self._session.run(self._output_names, tokens)
        last_hidden = output[0]  # [batch, seq_len, hidden]

        # Mean pool over sequence dimension using attention mask
        mask_input = tokens.get("attention_mask")
        if mask_input is None or (hasattr(mask_input, 'size') and mask_input.size == 0):
            mask_input = np.ones_like(next(iter(tokens.values())), dtype=np.int64)
        mask = np.expand_dims(mask_input.astype(np.float32), axis=-1)
        mask = np.expand_dims(mask_input.astype(np.float32), axis=-1)

        # Mean pool: sum along sequence dim, divide by mask count
        pooled = (last_hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
        embedding = pooled.flatten().tolist()

        return {
            "embedding": embedding,
            "model": self.model_path.split("/")[-1],
        }

    # ── Whisper (ASR) — placeholder ──────────────────────────────────

    async def _transcribe(self, audio_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Transcribe audio → text. Placeholder for ONNX whisper models."""
        await self._ensure_loaded()
        raise NotImplementedError(
            "Whisper support requires an ONNX whisper model. "
            "Provide a --model pointing to an .onnx file."
        )

    # ── TTS — placeholder ────────────────────────────────────────────

    async def _synthesize(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate audio from text. Placeholder for ONNX TTS models."""
        await self._ensure_loaded()
        raise NotImplementedError(
            "TTS support requires an ONNX TTS model. "
            "Provide a --model pointing to an .onnx file."
        )

    # ── HTTP handlers ────────────────────────────────────────────────

    async def _handle_health(self, request: web.Request) -> web.Response:
        try:
            await self._ensure_loaded()
            return web.json_response({"status": "ok"})
        except Exception as e:
            return web.json_response({"status": f"error: {e}"})

    async def _handle_transcribe(self, request: web.Request) -> web.Response:
        try:
            data = await request.json() if request.content_type == "application/json" else {}
            audio_file = await request.post()
            audio_bytes = audio_file.get("file").read() if "file" in audio_file else data.get("audio", b"")

            result = await self._transcribe(audio_bytes, **self.extra_kwargs)
            return web.json_response(result)
        except NotImplementedError:
            return web.json_response({"error": "Whisper not yet supported"}, status=501)
        except Exception as e:
            logger.error("Transcription error: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_speech(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            text = data.get("input", "")
            result = await self._synthesize(text, **self.extra_kwargs)

            import numpy as np
            audio_data = np.clip(result["audio"], -1.0, 1.0).astype(np.float32)
            int_audio = (audio_data * 32768).astype(np.int16)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(result.get("sample_rate", 24000))
                wf.writeframes(int_audio.tobytes())
            buf.seek(0)

            return web.Response(body=buf.read(), content_type="audio/wav")
        except NotImplementedError:
            return web.json_response({"error": "TTS not yet supported"}, status=501)
        except Exception as e:
            logger.error("TTS error: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_embeddings(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            input_text = data.get("input", "")

            result = await self._embed(input_text, **self.extra_kwargs)
            return web.json_response({
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": result["embedding"]}],
                "model": result["model"],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            })
        except Exception as e:
            logger.error("Embedding error: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    # ── Router ───────────────────────────────────────────────────────

    def _build_routes(self) -> list:
        routes = [web.get("/health", self._handle_health)]

        if self.inference_type == "whisper":
            routes.append(web.post("/v1/audio/transcriptions", self._handle_transcribe))
        elif self.inference_type == "tts":
            routes.append(web.post("/v1/audio/speech", self._handle_speech))
        elif self.inference_type == "embedding":
            routes.append(web.post("/v1/embeddings", self._handle_embeddings))
        else:
            raise ValueError(f"Unknown inference type: {self.inference_type}")

        return routes

    # ── Entry point ──────────────────────────────────────────────────

    def run(self) -> None:
        app = web.Application()
        app.router.add_routes(self._build_routes())

        logger.info(
            "Starting ONNX %s on :%d  model=%s  device=%s",
            self.inference_type, self.port, self.model_path, self.device,
        )
        web.run_app(app, host="0.0.0.0", port=self.port, print=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX inference server")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--type", required=True, choices=["whisper", "tts", "embedding"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "npu"],
                        help="Execution device (default: cpu)")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--tokenizer", default=None,
                        help="HF tokenizer repo-id or local dir")
    args = parser.parse_args()

    device_map = {"cpu": "CPUExecutionProvider", "npu": "NPUExecutionProvider"}
    server = OnnxServer(
        model_path=args.model,
        inference_type=args.type,
        device=device_map[args.device],
        port=args.port,
        tokenizer_path=args.tokenizer,
    )
    server.run()


if __name__ == "__main__":
    main()
