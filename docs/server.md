# OpenAI-Compatible Server (`server.py`)

Model Arkestra ships a production-ready OpenAI v1-compatible API server that sits on top of any started model. It exposes `POST /v1/chat/completions`, `GET /v1/models`, and `GET /health` — all compatible with OpenAI client libraries, LangChain's `OpenAIChat` adapter, and any tool that talks to the OpenAI endpoint.

## Quick Start — CLI

The simplest way to start is as a standalone process:

```bash
python -m model_arkestra.server --config config.yaml --port 8080
```

This launches a FastAPI server backed by ModelArkestra on port 8080. Models load lazily — the first request to `/v1/chat/completions` for a given model triggers its startup.

### CLI Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--config` / `-c` | *(optional)* | `~/.config/arkestra/config.yaml` | Path to YAML config file |
| `--port` / `-p` | — | `8080` | HTTP port to listen on |
| `--host` / `-H` | — | `0.0.0.0` | Bind address — use `127.0.0.1` for localhost-only |
| `--ready-timeout` / `-t` | — | `120` | Seconds to wait for models during startup |
| `--alias` / `-a` | — | — | OpenAI model alias mapping (`KEY=VALUE`). Repeat for multiple, e.g. `-a gpt-4=qwen3 -a claude=gemma` |
| `--api-key` | — | — | Require this Bearer token on every request (basic auth bypass) |
| `--cors` | — | `false` | Enable full CORS via CORSMiddleware — handles preflight OPTIONS, all standard Access-Control headers. Accepts any origins (`"*"`) by default. |
| `--ssl-certfile` | — | — | Path to TLS certificate file (PEM) |
| `--ssl-keyfile` | — | — | Path to TLS private key file (PEM) |
| `--log-level` | — | `info` | Uvicorn log level |
| `--workers` / `-w` | — | `1` | Number of worker processes |
| `--broadcast-addr` | — | auto | Address models bind to (`0.0.0.0` or `127.0.0.1`). No CLI value defers to config ``runners.broadcast_addr``, then global default ``0.0.0.0``. |

## Usage — Embed Into an Existing App

For deeper integration, create an `ArkestraServer`, extract its FastAPI app, and mount it alongside your own routes:

```python
from model_arkestra.server import ArkestraServer

server = ArkestraServer(
    "config.yaml",
    port=8080,
    openai_aliases={"gpt-4": "qwen3.5-4b", "claude": "gemma-4-e2b"},
)
app = server.get_app()

# Use `app` with uvicorn, or extend it further
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Enable CORS for browser clients

Pass ``allow_origins`` to install FastAPI's CORSMiddleware — this handles preflight `OPTIONS` requests and all standard Access-Control headers automatically:

```python
server = ArkestraServer(
    "config.yaml",
    port=8080,
    allow_origins=["*"],  # open to any origin; use specific origins for production
)
app = server.get_app()
uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Mount alongside other FastAPI routes

```python
from fastapi import FastAPI
from model_arkestra.server import ArkestraServer

# Build your own app first
app = FastAPI(title="My App")

@app.get("/custom")
def custom_endpoint():
    return {"hello": "world"}

# Mount the Arkestra app routes onto it
server = ArkestraServer("config.yaml")
arkestra_app = server.get_app()
app.mount("", arkestra_app)  # all /v1/*, /health routes merge in

uvicorn.run(app, port=8080)
```

### Embed into an existing FastAPI app's lifespan

When you need Arkestra as a component of a larger service, create the server but let the parent FastAPI app control its lifecycle through a lifespan manager:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from model_arkestra.server import ArkestraServer

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start Arkestra — models load lazily on first inference request
    server = ArkestraServer("config.yaml")
    app.mount("", server.get_app())
    yield
    await server.shutdown()

app = FastAPI(title="My Service", lifespan=lifespan)

@app.get("/custom")
def custom_endpoint():
    return {"hello": "world"}
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completion — blocking or streaming (SSE) |
| `GET` | `/v1/models` | List all tracked models in OpenAI format |
| `GET` | `/health` | Health check — returns running model count |
| `GET` | `/v1/health` | Alias for `/health` (OpenAI compat) |
| `POST` | `/v1/embeddings` | Text embeddings — ONNX models with `capabilities: [embed]` |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text (Whisper) — ONNX models with `capabilities: [stt]` |
| `POST` | `/v1/audio/speech` | Text-to-speech (Kokoro) — ONNX models with `capabilities: [tts]` |

### POST /v1/chat/completions

Accepts the standard OpenAI chat completion request body:

```json
{
  "model": "qwen3.5-4b",
  "messages": [
    {"role": "system", "content": "You are a helpful tutor."},
    {"role": "user", "content": "What is quantum entanglement?"}
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "stream": false
}
```

#### Non-streaming response

```json
{
  "id": "chatcmpl-default",
  "object": "chat.completion",
  "model": "qwen3.5-4b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Quantum entanglement..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 84, "total_tokens": 96}
}
```

#### Streaming response (SSE `text/event-stream`) — one chunk per token

```jsonlines
data: {"id":"cmpl-stream-default","object":"chat.completion.chunk",
       "model":"qwen3.5-4b","choices":[{"index":0,"delta":{"content":"Quantum"}}]}
data: {"id":"cmpl-stream-default","object":"chat.completion.chunk",
       "model":"qwen3.5-4b","choices":[{"index":0,"delta":{"content":" entanglement..."}}]}
data: {"id":"cmpl-stream-default","object":"chat.completion.chunk",
       "model":"qwen3.5-4b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

### GET /v1/models

Returns all configured models in OpenAI-compatible format:

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3.5-4b",
      "object": "model",
      "created": 1700000000,
      "owned_by": "local",
      "status": "running"
    }
  ]
}
```

| Field | Source |
|---|---|---|
| `id` | Model name |
| `object` | Always `"model"` |
| `created` | Unix timestamp |
| `owned_by` | From config (default `"local"`) |
| `status` | Lowercase state string: `running`, `stopped`, `uncached`, `error`, etc. |

See [Model Introspection](./usage.md#model-introspection) for equivalent Python API methods.

### POST /v1/embeddings

Text embeddings via ONNX inference server (models with `capabilities: [embed]`):

```json
{ "input": "hello world" }
```

Response:

```json
{
  "object": "list",
  "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, -0.2, ...] }],
  "model": "model.onnx",
  "usage": { "prompt_tokens": 0, "total_tokens": 0 }
}
```

The `embedding` array length depends on the model (e.g., BGE-small → 384-dim). Models are loaded lazily — the first request triggers ONNX session warmup.

Requires `onnxruntime>=1.28` and optionally `transformers>=4.40` for tokenization (`pip install 'model-arkestra[onnx]'`).

## Model Resolution — Aliases

The `openai_aliases` parameter maps OpenAI-style model IDs to your internal model names:

```python
server = ArkestraServer(
    "config.yaml",
    openai_aliases={"gpt-4": "qwen3.5-4b", "claude-3-opus": "llama3-70b"},
)
```

A request with `model: "gpt-4"` resolves to `"qwen3.5-4b"` automatically. If no alias matches, the model ID is passed through as-is (the runner uses it as the model name).

### Remote Model Proxying

When a configured model uses `runner: remote`, all inference requests are transparently proxied to the target worker:

```yaml
models:
  gpu-worker/qwen3-4b:
    checkpoint: unsloth/Qwen3-4B-GGUF:Q4_K_M
    backend: gpu-worker
backends:
  gpu-worker:
    runner: remote
    base_url: "http://192.168.1.42:18000"
```

A request to `/v1/chat/completions` with `model: "gpu-worker/qwen3-4b"` is forwarded to `http://192.168.1.42:18000/v1/chat/completions`. Streaming responses are relayed as SSE chunks in real time. Non-streaming requests wait for the worker's response and return it as-is.

## Request & Response Models

All Pydantic models are available for direct import:

```python
from model_arkestra.server import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionStreamResponse, Message, UsageInfo, ModelInfo,
)
```

| Model | Purpose |
|---|---|
| `ChatCompletionRequest` | Input shape — mirrors OpenAI's `/v1/chat/completions` body |
| `Message` | Single message with `role` and optional `content` (str or list) |
| `ChatCompletionResponse` | Non-streaming response — id, model, choices array, usage stats |
| `ChatCompletionStreamResponse` | Per-chunk SSE response during streaming |
| `ModelInfo` | Single model entry in `/v1/models` list |
| `ListModelsResponse` | Wrapper for `/v1/models` with `object: "list"` |

## Building Container Images

Two shell scripts build the container images from the project root:

### ROCm image

```bash
./scripts/build-ark-llama-rocm-container.sh           # uses podman (default)
./scripts/build-ark-llama-rocm-container.sh docker     # explicit runtime
```

Builds from `tests/files/Containerfile.rocm` and tags it as `localhost/ark-llama:rocm`.

### Vulkan (RADV) image

```bash
./scripts/build-ark-llama-vulkan-radv-container.sh     # uses podman (default)
./scripts/build-ark-llama-vulkan-radv-container.sh docker  # explicit runtime
```

Builds from `tests/files/Containerfile.vulkan-radv` and tags it as `localhost/ark-llama:vulkan-radv`.

Both scripts accept an optional first argument to select the container runtime (`podman` or `docker`). They exit with a clear error if the requested runtime isn't found on PATH.

## Related Documentation

- [Admin API](./admin.md) — admin panel that extends the same FastAPI app
- [Usage Guide](./usage.md) — Python API for starting/controlling models
- [Configuration Format](./config.md) — how config drives model behavior
- [HTTP Client](./http-client.md) — standalone client utility for external integrations
