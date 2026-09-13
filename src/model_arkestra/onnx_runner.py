"""ONNX model runner — loads models into memory, no subprocesses.

Derives from BaseRunner. Manages ONNX InferenceSession objects
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
import wave
from pathlib import Path
from typing import Any, Dict, Optional

from model_arkestra.base import BaseRunner
from model_arkestra.common import default_cache_root, resolve_model_ref


class OnnxRunner(BaseRunner):
    """Run ONNX models in-memory via BaseRunner lifecycle.

    Unlike ProcessRunner (subprocess) or ContainerRunner (podman/docker),
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
        self, ctx: "model_arkestra.types._Model", model_data: Dict[str, Any]
    ) -> None:
        """Load ONNX InferenceSession into context — no subprocess needed."""
        import onnxruntime as ort

        # Resolve model path from config or context
        model_path = getattr(ctx, '_model_path', None)
        if not model_path:
            default_section = (self.cm.data.get("default") or {})
            resolved = resolve_model_ref(
                raw=model_data.get("model"),
                default_section=default_section,
                model_repos=self.cm.data.get("model-repos"),
            )
            if resolved.repo == "hf":
                model_path = f"hf:{resolved.ref}"
            elif resolved.repo == "lcl":
                model_path = resolved.ref.removeprefix("lcl:")
        if not model_path:
            raise RuntimeError(
                f"Model '{ctx.name}' missing 'model' field or '_model_path'"
            )

        resolved_path = self._resolve_model_path(model_path, ctx)

        inference_type = "embed"
        tags = model_data.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        if "tts" in tags:
            inference_type = "tts"
        elif "asr" in tags or "whisper" in tags:
            inference_type = "whisper"

        # ── TTS models: use Kokoro class (manages its own ONNX session) ─
        if inference_type == "tts":
            voices_path = model_data.get("voices_path", "")
            resolved_voices = self._resolve_model_path(voices_path, ctx, file_pattern="*.bin")

            try:
                from kokoro_onnx import Kokoro as KokoroTTS
                ctx.kokero_model = KokoroTTS(str(resolved_path), str(resolved_voices))
                ctx.g2p_lang = model_data.get("g2p_lang", "en-us")
                if self.arkestra:
                    self.arkestra.log(f"[start] tts '{ctx.name}' loaded voice={ctx.g2p_lang}")
            except ImportError:
                if self.arkestra:
                    self.arkestra.log(
                        f"TTS model '{ctx.name}' requires the `onnx` extra. "
                        f"Install with: pip install model-arkestra[onnx]",
                        level="ERROR",
                    )
                raise RuntimeError(f"kokero-onnx not installed — TTS unavailable for '{ctx.name}'")
            return  # Kokoro manages its own session below

        # ── Streaming ASR models: sherpa-ai paraformer (VAD bundled) ──
        if inference_type == "sherpa-asr":
            ctx.inference_type = "sherpa-asr"
            if self.arkestra:
                self.arkestra.log(f"[start] sherpa-asr '{ctx.name}' loaded")
            return

        # ── Streaming TTS models: Piper (generates WAV per request) ───
        if inference_type == "piper":
            from piper import PiperVoice
            ctx.piper_voice = PiperVoice.load(str(resolved_path))
            ctx.inference_type = "piper"
            if self.arkestra:
                self.arkestra.log(f"[start] piper tts '{ctx.name}' loaded")
            return

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

        # Load tokenizer for whisper/embedding models (TTS handled above)
        if inference_type != "tts":
            tokenizer_path = model_data.get("tokenizer") or str(model_data.get("path", ""))
            if tokenizer_path:
                try:
                    from transformers import AutoTokenizer
                    ctx.onnx_tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_path, trust_remote_code=True)
                except Exception as e:
                    if hasattr(self, 'cm'):
                        logger.warning("Could not load tokenizer for '%s': %s", ctx.name, e)  # noqa: F821

    async def _stop_model_process(self, ctx: "model_arkestra.types._Model") -> None:
        """Unload ONNX session from memory."""
        if self.arkestra:
            self.arkestra.log(f"[stop] model={ctx.name} unloaded")
        for attr in ('onnx_session', 'onnx_tokenizer', 'kokero_model'):
            if hasattr(ctx, attr):
                delattr(ctx, attr)

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
            # _before_restart aborts early when already stopped; move to LOADING
            if ctx.state in (RunnerState.STOPPED, RunnerState.STOPPING):
                ctx.set_state("load")
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

            # Resolve cache directory via unified resolver
            default_section = (self.cm.data.get("default") or {})
            resolved = resolve_model_ref(
                raw=model_data.get("model"),
                default_section=default_section,
                model_repos=self.cm.data.get("model-repos"),
            )

            from model_arkestra.types import _Model
            ctx = _Model(model_name, eff_port, max_log_lines=log_size)
            ctx.backend_id = backend or model_data.get("backend")

            if resolved.cache_path:
                cache_root = default_cache_root()
                ctx._cache_dir = cache_root / f"models--{resolved.cache_path}"
                os.makedirs(ctx._cache_dir, exist_ok=True)

            self._models[model_name] = ctx
            if self.arkestra is not None and hasattr(self.arkestra, "_registry"):
                self.arkestra._registry.register(ctx)
            ctx.set_state("load")

        # Apply transient overrides
        for key in ('args',):
            if key in inference_kwargs and inference_kwargs[key] is not None:
                model_data[key] = inference_kwargs[key]
        self._inference_kwargs[model_name] = inference_kwargs  # type: ignore[assignment]

        # Load the ONNX model into memory (no HTTP needed)
        await self._start_model_process(ctx, model_data)

        # Warmup delay (if configured)
        if self.warmup_delay > 0:
            await asyncio.sleep(self.warmup_delay)

        ctx.set_state("ready")

    # ── Internal helpers ───────────────────────────────────────────
    # NOTE: logger is set at import time by arkestra.py after importing OnnxRunner.

    def _resolve_model_path(self, model_path: str, ctx: "model_arkestra.types._Model",
                            file_pattern: str = "*.onnx") -> Path:
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
                matches = list(sp.rglob(file_pattern))
                if matches:
                    return matches[0].resolve()

        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            f"Verify the path exists or is a valid HF repo ID."
        )


# ── Logging setup (imported from arkestra.py) ──────────────────────
# logger is set at import time when arkestra imports this module.
