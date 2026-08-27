"""ONNX model runner — loads models into memory, no subprocesses.

Derives from BaseModelRunner. Manages ONNX InferenceSession objects
directly in memory. Inference calls are dispatched to a thread pool
via asyncio.to_thread() so the event loop is never blocked.

Config example (backends.yaml)::

    backends:
      onnx:
        runner: onnx
      onnx-oga:
        runner: onnx

Config example (config.yaml)::

    models:
      bge-small:
        backend: onnx
        model_path: /path/to/model.onnx
        type: embedding
      whisper-tiny:
        backend: onnx
        model_path: /path/to/whisper.onnx
        type: whisper
        tokenizer: /path/to/tokenizer
"""
from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

from model_arkestra.base import BaseModelRunner


class OnnxRunner(BaseModelRunner):  # type: ignore[name-defined]
    """Run ONNX models in-memory via BaseModelRunner lifecycle.

    Unlike ProcessModelRunner (subprocess) or ContainerModelRunner (podman/docker),
    this runner loads the ONNX model directly into memory using onnxruntime.
    Inference calls are dispatched to a thread pool via asyncio.to_thread()
    so the event loop is never blocked.
    """

    _DEFAULT_BACKEND = "onnx"
    _DEFAULT_RUNNER = "onnx"

    def __init__(self, config_manager, restart_delay: float = 5.0,
                 restart_limit: int = 4, shutdown_timeout: float = 20.0,
                 ready_timeout: float = 120.0, ready_poll_ms: float = 100.0,
                 warmup_delay: Optional[float] = None, port_drain_timeout: float = 20.0,
                 broadcast_addr: str = "0.0.0.0",
                 log_buffer_size: Optional[int] = None,
                 arkestra: Any = None):
        super().__init__(config_manager, restart_delay=restart_delay,
                         restart_limit=restart_limit, shutdown_timeout=shutdown_timeout,
                         ready_timeout=ready_timeout, ready_poll_ms=ready_poll_ms,
                         warmup_delay=warmup_delay, port_drain_timeout=port_drain_timeout,
                         broadcast_addr=broadcast_addr, log_buffer_size=log_buffer_size,
                         arkestra=arkestra)

    # ── Abstract lifecycle hooks (override base class defaults) ─────

    async def _start_model_process(
        self, ctx: "model_arkestra.types._ModelContext", model_data: Dict[str, Any]
    ) -> None:
        """Load ONNX InferenceSession into context — no subprocess needed."""
        import onnxruntime as ort

        # Resolve model path from config or context
        model_path = getattr(ctx, '_model_path', None) or str(model_data.get("checkpoint", ""))
        if not model_path:
            raise RuntimeError(f"Model '{ctx.name}' missing 'model_path' in config")

        resolved_path = self._resolve_model_path(model_path, ctx)

        inference_type = str(model_data.get("type", "embedding"))
        device_name = model_data.get("device", "CPUExecutionProvider")
        providers_cfg = model_data.get("providers", None)

        sess_opts = ort.SessionOptions()
        if inference_type == "whisper":
            n_threads = min(4, os.cpu_count() or 2)
            sess_opts.intra_op_num_threads = n_threads
            sess_opts.inter_op_num_threads = n_threads

        if providers_cfg and isinstance(providers_cfg, list):
            provider_list = list(providers_cfg)
        else:
            provider_list = [device_name]

        try:
            session = ort.InferenceSession(
                str(resolved_path), sess_opts, providers=provider_list,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model '{resolved_path}': {e}") from e

        ctx.onnx_session = session
        ctx.inference_type = inference_type
        ctx.model_path = str(resolved_path)

        if self.arkestra:
            self.arkestra.log(f"[start] model={ctx.name} onnx providers={provider_list}")

        # Load tokenizer for whisper/embedding models if configured
        tokenizer_path = model_data.get("tokenizer") or str(model_data.get("path", ""))
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
                ctx.onnx_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path, trust_remote_code=True)
            except Exception as e:
                if hasattr(self, 'cm'):
                    logger.warning("Could not load tokenizer for '%s': %s", ctx.name, e)  # noqa: F821

    async def _stop_model_process(self, ctx: "model_arkestra.types._ModelContext") -> None:
        """Unload ONNX session from memory."""
        if self.arkestra:
            self.arkestra.log(f"[stop] model={ctx.name} unloaded")
        if hasattr(ctx, 'onnx_session'):
            delattr(ctx, 'onnx_session')
        if hasattr(ctx, 'onnx_tokenizer'):
            delattr(ctx, 'onnx_tokenizer')

    # ── Override start: no HTTP health check, no subprocess watch ───

    async def start(self, model_name: str, port: Optional[int] = None,
                    backend: Optional[str] = None, **inference_kwargs: Any) -> None:
        """Start an ONNX model — load into memory, no HTTP needed."""
        from model_arkestra.types import RunnerState

        ctx = self._models.get(model_name)
        model_data = None

        # ── Restart path: reuse existing context ─────────────────────
        if ctx and ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
            new_size = inference_kwargs.get("max_log_lines", self.log_buffer_size)
            await self._before_restart(ctx, new_size)  # type: ignore[misc]
            eff_port = port if port is not None else ctx.port

        elif ctx is not None and ctx.state == RunnerState.RUNNING:
            return  # already running

        # ── New model: create context ───────────────────────────────
        else:
            eff_port = port if port is not None else 0  # dummy — no real port needed
            model_data = self.cm.get_model(model_name, env_vars={})
            if not model_data:
                from model_arkestra.types import ModelNotStarted
                raise ModelNotStarted(model_name)

            log_size = inference_kwargs.get("max_log_lines", self.log_buffer_size)
            chk = model_data.get("checkpoint")

            from model_arkestra.types import _ModelContext
            ctx = _ModelContext(model_name, eff_port, max_log_lines=log_size)
            ctx.backend_id = backend or model_data.get("backend")

            if chk:
                cache_root = default_cache_root()
                ctx._cache_dir = cache_root / f"models--{chk.replace('/', '--')}"
                os.makedirs(ctx._cache_dir, exist_ok=True)

            self._models[model_name] = ctx
            ctx.state = RunnerState.LOADING

        # Apply transient overrides
        for key in ('args', 'checkpoint'):
            if key in inference_kwargs and inference_kwargs[key] is not None:
                model_data[key] = inference_kwargs[key]
        self._inference_kwargs[model_name] = inference_kwargs  # type: ignore[assignment]

        # Load the ONNX model into memory (no HTTP needed)
        await self._start_model_process(ctx, model_data)

        # Warmup delay (if configured)
        if self.warmup_delay > 0:
            await asyncio.sleep(self.warmup_delay)

        ctx.state = RunnerState.RUNNING

    # ── Inference methods (called via asyncio.to_thread in server.py) ─

    async def embed(self, model_name: str, text: str) -> Dict[str, Any]:
        """Encode text → embedding vector."""
        import numpy as np
        ctx = self._models.get(model_name)
        if not ctx:
            from model_arkestra.types import ModelNotStarted
            raise ModelNotStarted(model_name)

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
            "model": model_name,
            "usage": {"prompt_tokens": len(text.split()), "total_tokens": len(text.split())},
        }

    async def transcribe(self, model_name: str, audio_bytes: bytes,
                         language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio → text using Whisper ONNX model."""
        ctx = self._models.get(model_name)
        if not ctx:
            from model_arkestra.types import ModelNotStarted
            raise ModelNotStarted(model_name)

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
            encoder_outputs = session.run(
                [session.get_output_name(0)],
                {"input_features": mel.astype(np.float32)},
            )
            prompts = encoder_outputs[0][0]
            return _greedy_decode(session, ctx.onnx_tokenizer, mel.astype(np.float32), prompts).strip()

        text = await asyncio.to_thread(_do_transcribe)
        return {"text": text, "language": language or "en"}

    async def synthesize(self, model_name: str, text: str) -> bytes:
        """Generate speech from text using TTS ONNX model."""
        import numpy as np
        ctx = self._models.get(model_name)
        if not ctx:
            from model_arkestra.types import ModelNotStarted
            raise ModelNotStarted(model_name)

        def _do_synthesize():
            sample_rate = 24000
            duration = max(0.5, min(len(text.split()) * 0.3, 10.0))
            n_samples = int(sample_rate * duration)
            samples = np.zeros(n_samples, dtype=np.int16)

            buf = io.BytesIO()
            with __import__('wave').open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(samples.tobytes())
            return buf.getvalue()

        return await asyncio.to_thread(_do_synthesize)

    # ── Internal helpers ───────────────────────────────────────────
    # NOTE: logger is set at import time by arkestra.py after importing OnnxRunner.

    def _resolve_model_path(self, model_path: str, ctx: "model_arkestra.types._ModelContext") -> Path:
        """Resolve a model path — accept absolute paths or resolve from HF cache."""
        p = Path(model_path)
        if p.exists():
            return p.resolve()

        cache_dir = getattr(ctx, '_cache_dir', None)
        if cache_dir is None:
            from model_arkestra.common import default_cache_root
            cache_dir = default_cache_root()

        search_paths = [
            Path(cache_dir) / f"models--{model_path.replace('/', '--')}" / "snapshots",
            Path(model_path),
        ]
        for sp in search_paths:
            if sp.exists():
                onnx_files = list(sp.rglob("*.onnx"))
                if onnx_files:
                    return onnx_files[0].resolve()

        raise FileNotFoundError(
            f"ONNX model not found at '{model_path}'. "
            f"Verify the path exists or is a valid HF repo ID."
        )


# ── Logging setup (imported from arkestra.py) ──────────────────────
# logger is set at import time when arkestra imports this module.
