# ONNX Inference Server (`onnx_server.py`)

A lightweight, separate HTTP server that runs ONNX Runtime models for auxiliary workloads — embedding generation, speech-to-text (Whisper), and text-to-speech (Kokoro). It operates as an independent process, preserving GPU VRAM for main LLM inference.

## Quick Start

```bash
pip install 'model-arkestra[onnx]'

python -m model_arkestra.onnx_server \
    --model /path/to/model.onnx \
    --type embedding \
    --port 8090 \
    --tokenizer Xenova/bge-small-en-v1.5
```

Then send requests to `POST http://localhost:8090/v1/embeddings`.

### CLI Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model` | Yes | — | Path to ONNX model file (`.onnx`) |
| `--type` | Yes | — | Model type: `embedding`, `whisper`, or `tts` |
| `--device` | No | `cpu` | Execution device: `cpu` or `npu` |
| `--port` | No | `8090` | HTTP port to listen on |
| `--tokenizer` | No | — | HuggingFace repo ID or local dir for text tokenization (needed for embedding models) |

## Supported Model Types

### Embedding (`--type embedding`)

Standard BERT-style encoders with mean-pooling. Supports any model that outputs a `last_hidden_state` tensor with optional `attention_mask`.

**Tested models:**
- [`Xenova/bge-small-en-v1.5`](https://huggingface.co/Xenova/bge-small-en-v1.5) — 384-dim, BGE architecture
- Any onnx-community variant of MiniLM, BGE, or similar

**Request format (OpenAI-compatible):**
```json
{ "input": "hello world" }
```

**Response:**
```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.1, -0.2, ...]}],
  "model": "model.onnx",
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

### Speech-to-Text (`--type whisper`)

ONNX Whisper models for transcription. Placeholder — requires model-specific tokenization.

**Request:** multipart form with `file` (audio) or JSON with `audio` (bytes) and `language`.

### Text-to-Speech (`--type tts`)

ONNX-based TTS (e.g., Kokoro). Placeholder — returns WAV audio on `POST /v1/audio/speech`.

**Request:**
```json
{ "input": "Hello world", "voice": "af", "sample_rate": 24000 }
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` when model is loaded |
| `POST` | `/v1/embeddings` | Embedding inference (embedding type only) |
| `POST` | `/v1/audio/transcriptions` | ASR transcription (whisper type only) |
| `POST` | `/v1/audio/speech` | TTS synthesis (tts type only, returns WAV bytes) |

## NPU Acceleration

Set `--device npu` to use the AMD XDNA NPU via ONNX Runtime's Vitis AI Execution Provider:

```bash
python -m model_arkestra.onnx_server \
    --model /path/to/model.onnx \
    --type embedding \
    --device npu \
    --port 8091
```

This requires:
- Kernel ≥6.14 with `amdxdna` driver loaded
- AMD NPU firmware (provided by `linux-firmware-amd-misc`)
- `onnxruntime-vitisai` from the AMD XRT stack (`ppa:lemonade-team/stable`)
- `ulimit -l unlimited` for NPU memory locking

On systems without NPU support, fall back to CPU with `--device cpu`.

## Architecture

The ONNX server runs as a **separate process** from the main Arkestra server. It uses `aiohttp` (not FastAPI) for minimal overhead and crash isolation:

```
┌─────────────────┐         ┌──────────────────────┐
│  Arkestra Server │         │ ONNX Inference Server│
│  (FastAPI/uvicorn)│◄──►    │ (aiohttp)            │
│                  │         │                      │
│  /v1/chat/*      │         │ /health              │
│  /v1/embeddings  │         │ /v1/embeddings       │
│  /v1/audio/*     │         │ /v1/audio/transcripts│
└─────────────────┘         └──────────┬───────────┘
                                       │
                                   ONNX Runtime
                                  CPUExecProvider
                               (or NPUExecProvider)
```

Models load lazily on the first request. Session options are fixed at construction — no runtime reconfiguration.

## Dependencies

| Package | Required For | Install |
|---|---|---|
| `onnxruntime` | All ONNX models | Included in `[onnx]` extra |
| `transformers` | Tokenization (embedding) | Included in `[onnx]` extra |
| `scipy` | Audio resampling (whisper) | Included in `[onnx]` extra |
| `aiohttp` | HTTP server | Included with arkestra |

Install extras: `pip install 'model-arkestra[onnx]'`

## See Also

- [Server Documentation](./server.md) — main OpenAI-compatible API server
- [Architecture](./architecture.md) — runner routing and port allocation
- [Configuration Format](./config.md) — how to define auxiliary models in YAML
