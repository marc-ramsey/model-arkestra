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
import json
import logging
import os
import struct
import wave
from typing import Any, Dict, List, Optional, Tuple

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


def _create_mel_fb(n_mels: int, n_freq_bins: int, sr: int, f_max: float) -> np.ndarray:
    """Build a mel filterbank matrix matching Whisper's conventions.

    Args:
        n_mels: number of mel bands (80 for Whisper)
        n_freq_bins: FFT frequency bins (n_fft//2 + 1)
        sr: sample rate (16000 for Whisper)
        f_max: maximum frequency in Hz (8000 for Whisper)

    Returns:
        Mel filterbank matrix of shape [n_mels, n_freq_bins]
    """
    mel_low = 0.0
    mel_high = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_edges = np.linspace(mel_low, mel_high, n_mels + 2)

    # Convert mel to Hz
    hertz = 700.0 * (10.0 ** (mel_edges / 2595.0) - 1.0)

    # Convert Hz to FFT bin indices
    bins = np.floor(n_freq_bins * hertz / (sr / 2.0)).astype(int)

    fb = np.zeros((n_mels, n_freq_bins), dtype=np.float32)
    for m in range(n_mels):
        b_lo = max(bins[m], 0)
        b_hi = min(bins[m + 1], n_freq_bins)
        if b_hi <= b_lo:
            continue
        weights = np.linspace(0, 1, b_hi - b_lo)
        fb[m, b_lo:b_hi] = weights
    return fb


def _extract_mel_spectrogram(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Extract mel spectrogram from raw audio using scipy. Matches Whisper's preprocessing.

    Args:
        waveform: float32 audio samples [N]
        sr: sample rate (must be 16000 for Whisper)

    Returns:
        Mel spectrogram shape [1, 80, N_fft * hop_length] → [1, 80, 3000]
    """
    from scipy.signal import get_window, stft

    # Parameters matching Whisper's feature extraction
    n_fft = 400           # 23ms window at 16kHz
    hop_length = 160      # 10ms stride at 16kHz
    n_mels = 80
    sr_whisper = 16000

    # Resample if needed
    if sr != sr_whisper:
        from scipy.signal import resample
        num_samples = int(len(waveform) * sr_whisper / sr)
        waveform = resample(waveform, num_samples)
        sr = sr_whisper

    # Compute STFT
    window = get_window("hann", n_fft)
    f_arr, t_arr, stft_result = stft(waveform, fs=sr, window=window, nperseg=n_fft,
                                     noverlap=n_fft - hop_length, boundary=None)
    magnitude = np.abs(stft_result)  # shape: [freq_bins, time_frames]

    # Mel filterbank — use scipy's built-in function for correctness
    f_max = 8000.0

    mel_fb = _create_mel_fb(n_mels, stft_result.shape[0], sr, f_max)
    # mel_fb shape: [n_mels, n_freq_bins]
    
    # Apply filterbank: magnitude is [freq_bins, time_frames]
    # We want: [time_frames, n_mels] = [freq_bins, time_frames].T @ [freq_bins, n_mels]
    # Apply filterbank: magnitude is [freq_bins, time_frames]
    # We want: [time_frames, n_mels] = [freq_bins, time_frames].T @ [freq_bins, n_mels]
    mel_spec = (magnitude.T @ mel_fb.T).astype(np.float32)  # [time_frames, n_mels]

    # Convert to log scale and normalize to [0, 1] range (Whisper preprocessing)
    spec_db = np.log10(mel_spec + 1e-10)
    spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-10)

    # Pad or trim time dimension to exactly 3000 frames
    if spec_db.shape[0] < 3000:
        spec_db = np.pad(spec_db, ((0, 3000 - spec_db.shape[0]), (0, 0)))
    else:
        spec_db = spec_db[:3000, :]

    # Transpose to [1, n_mels, time_frames] = [1, 80, 3000]
    return spec_db.T.astype(np.float32)[np.newaxis, ...]


def _load_whisper_tokenizer(model_path: str) -> Dict[int, str]:
    """Load Whisper token ID → text mapping from model directory or fallback.

    Tries tokenizer.json in the model's parent dir and grandparent,
    then tiktoken, then returns empty dict as ultimate fallback.
    """
    # Try loading from model directory and its parents
    for search_dir in [os.path.dirname(str(model_path)), os.path.dirname(os.path.dirname(str(model_path)))]:
        if not search_dir:
            continue
        try:
            token_map_path = os.path.join(search_dir, 'tokenizer.json')
            with open(token_map_path) as f:
                raw = json.load(f)
                # SentencePiece format: vocab maps token_str -> int_id, invert it
                vocab = (raw.get("model", {}) or {}).get("vocab", {})
                if isinstance(vocab, dict):
                    return {int(v): k for k, v in vocab.items()}
        except Exception:
            pass

    # Fallback: tiktoken
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        tokens = {i: enc.decode([i]) for i in range(enc.n_vocab)}
        return {k: v for k, v in tokens.items()
                if not (k >= 50257 and k < 51864) or k == 50257}
    except ImportError:
        return {}


def _greedy_decode(
    session: Any,
    model_path: str,
    token_map: Dict[int, str],
    mel: np.ndarray,
    encoder_outputs: np.ndarray,
) -> str:
    """Greedy autoregressive decoding for Whisper decoder.

    Args:
        session: ONNX inference session (encoder model).
        model_path: Path to the encoder ONNX model.
        token_map: Token ID → text mapping.
        mel: Mel spectrogram [1, 80, 3000].
        encoder_outputs: Encoder hidden states [1, seq_len, d_model].

    Returns:
        Decoded transcription string.
    """
    import onnxruntime as ort

    # Find the decoder model — try multiple paths
    base = os.path.dirname(str(model_path))
    candidates = [base]
    
    # If we're in blobs/, also check snapshots/*/onnx/
    if 'blobs' in base:
        for root, dirs, files in os.walk(os.path.dirname(base)):
            onnx_dir = os.path.join(root, 'onnx')
            if os.path.isdir(onnx_dir) and 'blobs' not in onnx_dir:
                candidates.append(onnx_dir)
                break

    decoder_merged = None
    decoder_past = None
    for candidate in candidates:
        merged = os.path.join(candidate, 'decoder_model_merged_bnb4.onnx')
        past = os.path.join(candidate, 'decoder_with_past_model_bnb4.onnx')
        if not decoder_merged and os.path.isfile(merged):
            decoder_merged = merged
        if not decoder_past and os.path.isfile(past):
            decoder_past = past

    if os.path.isfile(decoder_merged):
        return _decode_autoregressive(
            session, model_path, token_map, mel, encoder_outputs,
            decoder_merged=decoder_merged,
        )
    elif os.path.isfile(decoder_past):
        return _decode_with_cache(
            session, model_path, token_map, mel, encoder_outputs,
            decoder_past=decoder_past,
        )
    else:
        raise RuntimeError(f"Whisper decoder model not found. Tried: {decoder_merged}, {decoder_past}")


def _decode_autoregressive(
    session: Any,
    model_path: str,
    token_map: Dict[int, str],
    mel: np.ndarray,
    encoder_outputs: np.ndarray,
    decoder_merged: str,
) -> str:
    """Decode using the merged decoder (encoder_hidden_states + input_ids)."""
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    dec_sess = ort.InferenceSession(decoder_merged, sess_options=sess_opts)
    inp_names = {i.name for i in dec_sess.get_inputs()}
    out_names = [o.name for o in dec_sess.get_outputs()]

    hidden_states = encoder_outputs.astype(np.float32)
    initial_token: np.ndarray = np.array([[50258]], dtype=np.int64)  # start_of_transcript

    # Count decoder layers from output names (present.{i}.decoder.key)
    present_keys = [int(n.split('.')[1]) for n in out_names if 'present.' in n and '.decoder.key' in n]
    num_layers = max(present_keys) + 1 if present_keys else 8
    zero_cache: List[np.ndarray] = [
        np.zeros((1, 6, 0, 64), dtype=np.float32) for _ in range(num_layers)
    ]

    enc_cache: List[np.ndarray] = [
        np.zeros((1, 6, 0, 64), dtype=np.float32) for _ in range(num_layers)
    ]

    tokens: List[int] = [50258]

    # Output layout (per layer i):
    #   outputs[1+4*i] = present.{i}.decoder.key
    #   outputs[2+4*i] = present.{i}.decoder.value
    #   outputs[3+4*i] = present.{i}.encoder.key
    #   outputs[4+4*i] = present.{i}.encoder.value
    def _dec_key(idx: int) -> int: return 1 + idx * 4
    has_enc_outputs = any('encoder.key' in o for o in out_names)

    for step in range(128):
        inputs: Dict[str, Any] = {
            "input_ids": initial_token,
            "encoder_hidden_states": hidden_states,
        }
        # Add past key/values to cache
        for i in range(num_layers):
            dec_key_name = f"past_key_values.{i}.decoder.key"
            dec_val_name = f"past_key_values.{i}.decoder.value"
            enc_key_name = f"past_key_values.{i}.encoder.key"
            enc_val_name = f"past_key_values.{i}.encoder.value"
            if dec_key_name in inp_names:
                inputs[dec_key_name] = zero_cache[i]
            if dec_val_name in inp_names:
                inputs[dec_val_name] = zero_cache[i]
            if enc_key_name in inp_names:
                inputs[enc_key_name] = enc_cache[i]
            if enc_val_name in inp_names:
                inputs[enc_val_name] = enc_cache[i]
        # First step: compute KV from encoder_hidden_states. Subsequent steps: reuse cached KV
        if "use_cache_branch" in inp_names:
            inputs["use_cache_branch"] = np.array([step > 0], dtype=np.bool_)

        outputs = dec_sess.run(out_names, inputs)
        logits = outputs[0].astype(np.float32)
        next_id = int(np.argmax(logits[0, -1]))
        tokens.append(next_id)
        if next_id == 50257:  # EOS
            break

        # Update caches from outputs (decoder KV at 1+4*i, encoder KV at 3+4*i)
        for i in range(num_layers):
            zero_cache[i] = outputs[_dec_key(i)].astype(np.float32)
            if has_enc_outputs:
                enc_cache[i] = outputs[_dec_key(i) + 2].astype(np.float32)

        initial_token = np.array([[next_id]], dtype=np.int64)

    return _tokens_to_text(tokens, token_map)


def _decode_with_cache(
    session: Any,
    model_path: str,
    token_map: Dict[int, str],
    mel: np.ndarray,
    encoder_outputs: np.ndarray,
    decoder_past: str,
) -> str:
    """Decode using the KV-cache decoder (encoder outputs injected as first-step cache)."""
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    dec_sess = ort.InferenceSession(decoder_past, sess_options=sess_opts)
    inp_names = [i.name for i in dec_sess.get_inputs()]
    out_names = [o.name for o in dec_sess.get_outputs()]

    # Build initial key/values from encoder hidden states
    hs = encoder_outputs.astype(np.float32)  # [1, seq_len, d_model]
    zero_cache: List[np.ndarray] = [
        np.zeros((1, 0, hs.shape[-1]), dtype=np.float32) for _ in range(8)
    ]

    initial_token = np.array([[50258]], dtype=np.int32)
    tokens: List[int] = [50258]

    # First step: pass encoder hidden states as "past key/values"
    inputs: Dict[str, Any] = {"input_ids": initial_token}
    for i in range(8):
        enc_key_name = f"past_key_values.{i}.encoder.key"
        enc_val_name = f"past_key_values.{i}.encoder.value"
        dec_key_name = f"past_key_values.{i}.decoder.key"
        dec_val_name = f"past_key_values.{i}.decoder.value"
        if enc_key_name in inp_names and enc_val_name in inp_names:
            inputs[enc_key_name] = hs
            inputs[enc_val_name] = hs
        if dec_key_name in inp_names and dec_val_name in inp_names:
            inputs[dec_key_name] = zero_cache[i]
            inputs[dec_val_name] = zero_cache[i]

    outputs = dec_sess.run(out_names, inputs)
    logits = outputs[0].astype(np.float32)
    next_id = int(np.argmax(logits[0, -1]))
    tokens.append(next_id)

    # Subsequent steps: reuse decoder key/values from output
    past_cache = [np.zeros((1, 0, hs.shape[-1]), dtype=np.float32) for _ in range(8)]
    cur_token = np.array([[next_id]], dtype=np.int32)

    for _ in range(127):
        if next_id == 50257:  # EOS
            break
        inputs = {"input_ids": cur_token}
        outputs_names = [o.name for o in dec_sess.get_outputs()]
        # Map output past keys to input keys
        for i in range(8):
            inp_key = f"past_key_values.{i}.decoder.key"
            inp_val = f"past_key_values.{i}.decoder.value"
            out_key = f"present.{i}.decoder.key"
            out_val = f"present.{i}.decoder.value"
            if inp_key in inp_names and out_key in outputs_names:
                inputs[inp_key] = past_cache[i]
                inputs[inp_val] = past_cache[i]
        
        raw_outputs = dec_sess.run(outputs_names, inputs)
        # First output is logits, rest are present key/values
        next_id = int(np.argmax(raw_outputs[0].astype(np.float32)[0, -1]))
        tokens.append(next_id)
        if next_id == 50257:
            break
        # Update cache from outputs (skip logits at index 0)
        for i in range(8):
            out_key = f"present.{i}.decoder.key"
            if i < len(raw_outputs) - 1:
                past_cache[i] = raw_outputs[2 + i * 2]
        cur_token = np.array([[next_id]], dtype=np.int32)

    return _tokens_to_text(tokens, token_map)


def _tokens_to_text(tokens: List[int], token_map: Dict[int, str]) -> str:
    """Convert token IDs to text, filtering Whisper special tokens."""
    result = []
    for tid in tokens:
        # Skip EOS (50257), startoftranscript (50258), task/lang tokens (50259-50358)
        # and timestamps (50360-51864)
        if 50257 <= tid < 51864:
            continue
        result.append(token_map.get(tid, ''))
    return "".join(result).strip()


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

    # ── Whisper tokenizer ────────────────────────────────────────────

    WHISPER_TOKEN_MAP: Optional[Dict[int, str]] = None

    def _load_whisper_tokenizer(self) -> Dict[int, str]:
        """Load or build Whisper token ID → text mapping."""
        if self.WHISPER_TOKEN_MAP is not None:
            return self.WHISPER_TOKEN_MAP

        # Try loading from model directory
        tokenizer_path = getattr(self, '_whisper_tokenizer_path', None)
        if tokenizer_path:
            try:
                import json
                token_map_path = str(tokenizer_path).replace('model.onnx', 'tokenizer.json')
                with open(token_map_path) as f:
                    raw = json.load(f)
                    # Convert to int→str map
                    self.WHISPER_TOKEN_MAP = {
                        int(k): v for k, v in (raw.get("model", {}) or {}).items()
                        if isinstance(k, str) and k.isdigit()
                    }
                    if self.WHISPER_TOKEN_MAP:
                        return self.WHISPER_TOKEN_MAP
            except Exception:
                pass

        # Fallback: build from OpenAI's Whisper tokenizer bytes
        try:
            from transformers import AutoTokenizer
            if self._tokenizer:
                tokenizer = self._tokenizer
                vocab = tokenizer.get_vocab()
                if isinstance(vocab, dict):
                    self.WHISPER_TOKEN_MAP = {
                        int(k) if isinstance(k, str) and k.isdigit() else int(k): v
                        for k, v in vocab.items()
                    }
                    return self.WHISPER_TOKEN_MAP
        except Exception:
            pass

        # Ultimate fallback: OpenAI whisper tokenizer bytes (embedded)
        import base64
        # This is the compressed vocabulary from whisper.tokenizer.DATA
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            self.WHISPER_TOKEN_MAP = {
                i: enc.decode([i]) for i in range(enc.n_vocab)
            }
            # Filter to Whisper-specific token ranges
            whisper_tokens = {k: v for k, v in self.WHISPER_TOKEN_MAP.items()
                            if not (k >= 50257 and k < 51864) or k == 50257}  # keep task/lang tokens
        except ImportError:
            # Minimal fallback: return empty dict — decoding will produce garbage but won't crash
            self.WHISPER_TOKEN_MAP = {}

        return self.WHISPER_TOKEN_MAP or {}

    def _decode_tokens(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode Whisper token IDs to text."""
        token_map = self._load_whisper_tokenizer()
        if not token_map:
            return "[no tokenizer available]"

        special_ids = {1, 2, 50257, 50359}  # start_of_transcript, previous_text_end, etc.
        if skip_special:
            special_ids.update(range(50360, 50400))  # lang tokens
            special_ids.add(50363)  # no_timestamps

        chars: List[str] = []
        for tid in token_ids:
            if skip_special and tid in special_ids:
                continue
            text = token_map.get(tid, f"\uFFFD")  # U+FFFD = replacement char
            chars.append(text)

        return "".join(chars).strip()

    def _greedy_decode(self, encoder_outputs: np.ndarray,
                       initial_token: int) -> List[int]:
        """Greedy autoregressive decoding for Whisper decoder model.

        Args:
            encoder_outputs: hidden states from encoder [1, seq_len, d_model]
            initial_token: first token ID (start_of_transcript)

        Returns:
            List of token IDs including initial token
        """
        # Import onnxruntime here to avoid circular dependency issues
        import onnxruntime as ort

        decoder_path = getattr(self, '_decoder_model_path', None)
        if not decoder_path or not os.path.isfile(decoder_path):
            raise RuntimeError(
                "Whisper decoder model not found. Provide path via "
                "--decoder-path or use a single-file Whisper ONNX export."
            )

        # Load decoder session (reuse if available)
        if not hasattr(self, '_decoder_session') or self._decoder_session is None:
            sess_opts = ort.SessionOptions()
            self._decoder_session = ort.InferenceSession(
                decoder_path,
                sess_options=sess_opts,
                providers=[self.device],
            )

        dec_input_names = [inp.name for inp in self._decoder_session.get_inputs()]
        dec_output_names = [out.name for out in self._decoder_session.get_outputs()]

        # Find input/output names (whisper decoder convention)
        hidden_name = "past_key_values.0.key" if any(
            "past_key" in n for n in dec_input_names) else None
        token_name = None
        encoder_out_name = None
        for n in dec_input_names:
            if "input_tokens" in n or (token_name is None and "input" in n):
                token_name = n
            elif "encoder_outputs" in n or "hidden_states" in n:
                encoder_out_name = n

        token_name = token_name or dec_input_names[0]
        encoder_out_name = encoder_out_name or dec_input_names[1] if len(dec_input_names) > 1 else None

        # Build past_key_values from encoder outputs (for single-layer case)
        past_init = {}
        # Standard whisper decoder has 32 layers; create dummy past keys
        num_layers = 32
        d_model = encoder_outputs.shape[-1] if len(encoder_outputs.shape) == 3 else 512
        head_dim = d_model // 8  # 8 heads typical

        for layer_idx in range(num_layers):
            past_init[f"past_key_values.{layer_idx}.key"] = np.zeros(
                (1, 8, 0, head_dim), dtype=np.float32)
            past_init[f"past_key_values.{layer_idx}.value"] = np.zeros(
                (1, 8, 0, head_dim), dtype=np.float32)

        token_ids = [initial_token]
        max_tokens = 512  # safety limit
        end_token = 50257  # English end-of-text token

        for _ in range(max_tokens):
            current_tokens = np.array([token_ids], dtype=np.int64)

            feed_dict: Dict[str, np.ndarray] = {token_name: current_tokens}
            if encoder_out_name:
                feed_dict[encoder_out_name] = encoder_outputs.astype(np.float32)
            feed_dict.update(past_init)

            outputs = self._decoder_session.run(dec_output_names, feed_dict)
            logits = outputs[0][0, -1, :]  # last token's logits

            # Greedy: pick highest probability
            next_id = int(np.argmax(logits))
            if next_id == end_token:
                break
            token_ids.append(next_id)

        return token_ids

    # ── TTS — placeholder ────────────────────────────────────────────

    async def _synthesize(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate audio from text. Placeholder for ONNX Kokoro TTS models."""
        await self._ensure_loaded()
        # Kokoro requires phoneme tokenization (not yet implemented).
        # This placeholder returns silence so the endpoint works.
        sample_rate = int(kwargs.get("sample_rate", 24000))
        duration_seconds = max(1, len(text.split()) * 0.3)  # rough estimate
        num_samples = int(sample_rate * duration_seconds)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        return {"audio": audio_data, "sample_rate": sample_rate}

    # ── HTTP handlers ────────────────────────────────────────────────

    async def _handle_health(self, request: web.Request) -> web.Response:
        try:
            await self._ensure_loaded()
            return web.json_response({"status": "ok"})
        except Exception as e:
            return web.json_response({"status": f"error: {e}"})

    async def _handle_transcribe(self, request: web.Request) -> web.Response:
        try:
            # Accept multipart form with audio file OR JSON with base64-encoded audio
            if request.content_type == "multipart/form-data":
                data = await request.post()
                audio_file = data.get("file")
                audio_bytes = await audio_file.read() if audio_file else b""
                language = (data.get("language") or {}).get_body() if data.get("language") else None
            else:
                data = await request.json()
                # Support base64-encoded audio or file upload as blob
                b64_audio = data.get("audio_b64") or data.get("audio", "")
                if isinstance(b64_audio, str):
                    import base64
                    try:
                        audio_bytes = base64.b64decode(b64_audio)
                    except Exception:
                        audio_bytes = b""
                elif isinstance(b64_audio, bytes):
                    audio_bytes = b64_audio
                else:
                    audio_bytes = b""
                language = data.get("language")

            if not audio_bytes or len(audio_bytes) < 44:  # minimum WAV header
                return web.json_response(
                    {"error": "No audio provided. Send a WAV file or base64-encoded audio."},
                    status=400)

            result = await self._transcribe(audio_bytes, language=language,
                                            **self.extra_kwargs)
            return web.json_response(result)
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
