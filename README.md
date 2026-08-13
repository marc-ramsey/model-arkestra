Model Arkestra is a lightweight Python orchestrator for running local LLM inference engines — primarily [llama.cpp](https://github.com/ggerganov/llama.cpp) — across your choice of backends, from bare-metal subprocesses to isolated containers (Podman / Docker). It exists so you can deploy and manage models on your own hardware with **safety and stability**, without the overhead of a full-blown proxy or cluster manager.

This is **not** a replacement for [Lemonade](https://github.com/ollama/lemonade) or [llama-swap](https://github.com/sgl-project/llama-swap). No model registries, no auto-scaling, no Kubernetes babysitting. If you just want models up and running on your own GPU — with clean lifecycle management, graceful shutdowns, and restart resilience out of the box — Arkestra is a straight line between config file and inference.

> *Author's note: Consider this an experiment. A first attempt at entirely LLM-coded and (soon) documented project, built using [unsloth/unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q4_K_XL](https://huggingface.co/unsloth/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) on a [Corsair AI Workstation 300](https://www.corsair.com/us/en/cp/category/builds/corsair-ai-workstation-300/) with 128GB integrated memory. This was not vibe coding — I watched the reasoning carefully and intervened numerous times when things started going off track. Development started with a lemonade-inspired config.yaml, with a prompt to write python code to read and set up an equivalent python object. With with several more short prompts that became [llm_config_manager](https://github.com/marc-ramsey/llm-config-manager), from which BaseModelRunner and it's subclasses were derived. The only manual editing was on config.yaml as more details were added, then some cleanup edits on README.md. Most needed interventions were along the lines of "stop looping," "overthinking the problem," or "focus on the problem, nothing else," with occasional "NO, do it this way…," which can be reduced through suitable agent loops (next project). Total time from start to completion: roughly 50 hours. I could have written the actual code faster myself with search/summaries from a model, this time, but there was once a short period when I code write assembly language code faster than C. This feels about the same.*

## Architecture

ModelArkestra handles port allocation and backend→runner routing.

<table style="border-collapse: collapse; width: 100%; table-layout: fixed; border: 1px solid black; font-family: sans-serif; text-align: center; margin: 0;">
  <!-- Top Section (Full Width) -->
  <tr>
    <td colspan="3" style="border: 1px solid black; padding: 10px;"><b style="font-style: italic;">ModelArkestra</b></td>
  </tr>
  <tr>
    <td colspan="3" style="border: 1px solid black; padding: 4px; font-size: 0.9em;">── port allocation, backend→runner routing</td>
  </tr>
  <tr>
    <td colspan="3" style="border: 1px solid black; padding: 4px; font-size: 0.9em;">__build__runner_class_map() → config-driven</td>
  </tr>
  <tr>
    <td colspan="3" style="border: 1px solid black; padding: 4px; font-size: 0.9em;">runners: section maps type → class</td>
  </tr>
  <tr>
    <td colspan="3" style="border: 1px solid black; padding: 4px; font-size: 0.9em;">__get___runner__instance(type, model) → lazy factory</td>
  </tr>

  <!-- Bottom Section -->
  <tr>
    <td rowspan="3" style="border: 1px solid black; padding: 15px 5px; vertical-align: middle;">
      <b style="font-style: italic;">ProcessModelRunner</b><br><br>subprocesses
    </td>
    <td colspan="2" style="border: 1px solid black; padding: 8px;">
      <b style="font-style: italic;">ContainerModelRunner</b><br>(abstract base)
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;"><b style="font-style: italic;">PodmanModelRunner</b></td>
    <td style="border: 1px solid black; padding: 8px;"><b style="font-style: italic;">DockerModelRunner</b></td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;">OCI containers</td>
    <td style="border: 1px solid black; padding: 8px;">OCI containers</td>
  </tr>
</table>

### Runner routing — explicit args override config

Runner type resolution follows a strict precedence: **explicit arguments take priority over all configuration values**. An explicit runner type that does not exist in the registry is rejected immediately.

#### Explicit override (highest priority)

When `runner=` is supplied, that value is used directly:

```
runner="podman" → PodmanModelRunner   (verified against registry)
runner="process" → ProcessModelRunner  (verified against registry)
```

#### Automatic resolution (config chain)

When no explicit argument is given, the runner type is resolved through config:

```
backends section exists?
  └─> model.backend → backends.<id>.runner → runners:<type> (or "process")
backends section absent?
  └─> runners:<default> (or "process")
```

### Port allocation — a global counter controlled by `ModelArkestra`. The starting port and range are driven entirely by config:

| Config key | Purpose                                                                                |
|---|---|
| `models-start-port` | First port in the range (default 18000)                                                |
| `model-ports` | Size of the pool — valid ports are `start_port` through `start_port + model-ports - 1` |

When `ModelArkestra.start()` is called without an explicit `port`, it allocates the next available number from this range sequentially. Once all ports in the pool are exhausted, `RuntimeError("Port range exceeded: …")` is raised immediately.

A **direct** runner instance (e.g., `ProcessModelRunner(cm)`) does **not** use a global counter — it picks `models-start-port` as the default for its first model, and subsequent calls reuse existing ports via `_dispatch()` / in-place restart.

Stopped models **retain their port assignments** in both the orchestrator and direct runner instances. Calling `start()` again on a stopped model restarts it on the same port (in-place). New models are assigned the next unused port from the pool when using `ModelArkestra`.

```
Config: start_port=18000, model-ports=32  →  valid range 18000–18031

Model started      Port assigned
─────────────────  ───────────
"alpha"            18000    ← auto-allocated from pool
"beta"             18001    ← next in pool
"gamma" (port=9000) 9000   ← explicit override, bypasses pool
"delta"            18002    ← next in pool
"alpha" stopped → start("alpha")  18000  ← restart-in-place, same port
"epsilon"          18003    ← next free port in pool
```

Explicit ports bypass the pool entirely — no range validation is performed. The caller bears that responsibility.

---

## Usage

### Basic Initialization

```python
# Orchestration layer (recommended)
from model_arkestra.arkestra import ModelArkestra

runner = ModelArkestra("config.yaml", ready_timeout=120.0)

async with runner:
    # ... use runner.start(), stop(), ainvoke(), etc ...
# → shutdown() called automatically on exit

# Direct runners for standalone use (they do support context managers)
from model_arkestra.process import ProcessModelRunner

cm = ConfigManager("config.yaml")  # optional: pass ConfigManager directly
runner = ProcessModelRunner(cm, shutdown_timeout=20.0, ready_timeout=120.0)
```

### Starting a Model

```python
# Auto-assigns port and picks runner from config chain
await runner.start("gpt-oss-20b")

# Model with backend: rocm → runner picks podman from config chain (no runner= needed)
await runner.start("qwen3-4b:rocm")

# Explicit backend override (also routes correctly via config chain)
await runner.start("qwen3-4b", backend="rocm")

# Explicit port assignment (caller responsibility to avoid conflicts)
await runner.start("qwen3.6-35B-think", port=18005)

# Explicit runner= override forces a specific runner type directly
await runner.start("qwen3-4b", runner="podman")

# Multiple models can run concurrently, each on its own port
print(runner.running_models)
# → {"gpt-oss-20b", "qwen3.6-35B-think", "gemma-4-26b-instruct"}
```

### Restarting a Model

Stops the running instance and starts a fresh one on the **same port**. Optional keyword args override backend or runner for the new instance.

Calling `start()` again on a stopped model restarts it in-place on the same port (no need to call `stop()` first). Calling `start()` on an already-running model is idempotent — it checks `/health` and returns immediately if the server responds with 200.

```python
# Identical restart — same config, same port
await runner.restart("gpt-oss-20b")

# Switch backend (runner resolves to podman from config)
await runner.restart("qwen3-4b", backend="rocm")

# Force a specific container runtime regardless of config
await runner.restart("qwen3-4b", runner="docker")

# Restart a stopped model via start() (same-port, same-backend)
await runner.stop("gpt-oss-20b")
await runner.start("gpt-oss-20b")  # reuses the original port

# Calling start() on an already-running model is idempotent
await runner.start("qwen3-4b")  # checks /health, returns if alive
```

### Sending a Prompt (Blocking)

Single prompt:
```python
result = await runner.ainvoke("gpt-oss-20b", "Explain quantum entanglement")
print(result)
# → "Quantum entanglement is a phenomenon in which..."
```

Full conversation history (multi-turn):
```python
result = await runner.ainvoke(
    "gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a physics tutor."},
        {"role": "user", "content": "What is quantum entanglement?"},
        {"role": "assistant", "content": "It's when particles become correlated..."},
        {"role": "user", "content": "Can you explain it simply?"},
    ],
)
```

### Streaming Response (Token-by-Token)

```python
async for event in runner.astream("gpt-oss-20b", {"prompt": "Write a haiku"}):
    if "token" in event:
        print(event["token"], end="", flush=True)  # Hell -> o -> Wo -> r -> l -> d
    elif "usage" in event:
        print()
        print(f"Done ({event['usage']['total_tokens']} tokens, {event['usage']['time_seconds']:.1f}s)")

# Output:
# Hello World
# Done (3 tokens, 0.2s)
```

### Graceful Shutdown

```python
# Stop a single model
await runner.stop("gemma-4-26b-instruct")

# Stop all running models — processes/containers are killed but model entries remain
# in STOPPED state so start() can restart them in-place on the same port.
await runner.stop_all()

# Full teardown — stops all models, clears internal state, resets port allocator
# For container runners, also force-removes all stopped containers.
await runner.shutdown()

# Or use async context manager (auto-calls shutdown() on exit)
async with ModelArkestra("config.yaml") as runner:
    await runner.start("qwen3-4b")
    # ... do work ...
# → shutdown() called here, port counter resets to start value
```

---

## API Reference — `ModelArkestra`

The centralized entry point. Does not launch any processes at this point — models must be started explicitly via `start()`.

| Parameter | Type | Default | Description                                                                                                                                          |
|---|---|---|---|
| `config_path` | `str` | *(required)* | Path to the YAML config file.                                                                                                                        |
| `start_port` | `int` | `18000` | Fallback starting port — only used when `models-start-port` is absent from config. Port allocation reads the actual values from config at init time. |
| `**runner_kwargs` | — | — | Passed through to each runner instance (e.g. `ready_timeout`, `warmup_delay`).                                                                       |

### Backward-compat shims

The properties `.process_runner`, `.podman_runner`, and `.docker_runner` still exist for backward compatibility. Each delegates to the unified lazy factory (`_get_runner_instance`) so they behave identically to the config-driven path.

### Methods

#### `async start(model_name: str, port: int | None = None, backend: str | None = None, runner: str | None = None) -> None`

Starts the server process or container for *model_name*. Runner type is resolved automatically:
- If `runner=` is provided, that runner type is used directly.
- Otherwise: `backends.<id>.runner` (from config) → `runners:<type>` in config → falls back to `"process"`.

If the model already exists and is in a STOPPED state, it restarts in-place on the same port. If it is already RUNNING, `/health` is polled — 200 returns immediately, anything else continues startup. Polls `/health` until ready or timeout. Raises `ServerReadyTimeout` on health-check failure (not on intermediate 502/504; those raise `RunnerError`).

#### `async stop(model_name: str) -> None`

Stops the model matching *model_name*. Sends the appropriate shutdown signal (see [Lifecycle](#process-lifecycle)).

#### `async restart(model_name: str, *, backend: str | None = None, runner: str | None = None) -> None`

Stops the running instance and starts a new one on the **same port**.  Optional keyword args override backend or runner for the restarted instance.

#### `async stop_all() -> None`

Stops all model processes. Model entries remain in `_models` with state `STOPPED` — calling `start()` again on a stopped model restarts it in-place on the same port. For container runners, containers are **not** removed (only force-removed during `shutdown()`).

#### `async shutdown() -> None`

Full teardown: stops all models, clears all runner and model entries, and resets the port allocator to `models-start-port`. For container runners, also sends `rm -f` on every stopped container. Use this for final cleanup (e.g. context manager exit, test teardown).

#### `async ainvoke(model_name: str, prompt: str = "", backend: str | None = None, messages: list[dict[str, Any]] | None = None) -> str`

Sends a blocking completion request and returns the full response string. Retries up to 12 times on connection errors or 503 status (with 2.5s backoff). Raises `RunnerError` for 502/504 (upstream failures), connection unreachable, or max retries exceeded.

Accepts either a single `prompt` string **or** a full `messages` list (OpenAI-compatible format) for multi-turn conversations:

```python
# Single prompt (legacy)
result = await runner.ainvoke("qwen3-4b", prompt="Hello")

# Full conversation history — includes system message, user turns, assistant response
result = await runner.ainvoke(
    "qwen3-4b",
    messages=[
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2 + 2 = 4"},
        {"role": "user", "content": "And 3 times that?"},
    ],
    temperature=0.7,          # forwarded to the runner
)

# Single prompt with stop tokens
result = await runner.ainvoke("qwen3-4b", "Write a poem.", stop=["\n\n"])
```

Additional keyword arguments (e.g. `temperature`, `stop`, `top_p`) are forwarded through to the underlying runner's invoke method.

When both `prompt` and `messages` are provided, `messages` takes precedence.

#### `async astream(model_name: str, payload: Dict[str, Any], backend: str | None = None) -> AsyncIterator[Dict[str, Any]]`

Sends a streaming completion request and yields events for each SSE update. Returns an async generator — callers iterate with `async for`. The payload dict must contain `"prompt"` (string) or `"messages"` (list of message dicts).

Each yielded dict is one of two forms:

**Token event** (one per SSE message during generation):
```json
{ "token": "Hell" }
```

**Usage summary** (yielded once at the end, replacing `[DONE]`):
```json
{
  "usage": {
    "model": "...",
    "prompt_tokens": N,
    "completion_tokens": N,
    "total_tokens": N,
    "time_seconds": T,
    "tokens_per_second": R
  }
}
```

Token chunks may be partial — e.g., the first event might be `"token": "Hell"`, the next `"token": "o World"` rather than full words. Callers should concatenate tokens and/or check for a trailing space if word boundaries matter.

#### `async request(model_name: str, path: str, **kwargs) -> dict | bytes`

Low-level HTTP forwarder for custom server endpoints not covered by `ainvoke()` or `astream()`. Forwards *kwargs* as the request payload and returns the parsed JSON response (or raw bytes if the endpoint does not return JSON). Only sends POST requests with 15s timeout. Raises `RunnerError` on responses with status ≥ 400.

#### `running_models: set[str]`

Read-only property returning the names of all models currently in `"running"` state, aggregated across all runner instances.

---

## Model Introspection — runtime state

These three methods expose live model state tracked at runtime (port, backend, runner type, health). They operate on models that have been started — before any `start()` call the lists are empty.

#### `_get_model_contexts() -> list[_ModelContext]`

Internal aggregation: returns every tracked `_ModelContext` across all runners. Each context carries `name`, `port`, `state` (`RunnerState` enum), `backend_id`, `runner_type`, `restart_count`, and `last_error`. Callers who need detailed runtime info should use this.

#### `get_model_list() -> list[str]`

Convenience wrapper around `_get_model_contexts()` — returns just the model names as strings. Useful when you want a dynamic list of currently-tracked models (including stopped or errored ones) rather than static YAML config names.

```python
runner = ModelArkestra("config.yaml")
await runner.start("qwen3-4b")
print(runner.get_model_list())  # → ["qwen3-4b"]
```

#### `get_v1_models() -> dict`

OpenAI-compatible `/v1/models` response. Returns a dict with `"object": "list"` and a `"data"` array — one entry per tracked model containing:

| Field | Source                                                            |
|---|---|
| `id` | model name                                                        |
| `object` | `"model"`                                                         |
| `created` | Unix timestamp                                                    |
| `owned_by` | from config (`model_cfg.get("owned_by")`, default `"local"`)      |
| `status` | lowercase state string (e.g. `"running"`, `"stopped"`, `"error"`) |
| `port` | allocated port number                                             |
| `runner_type` | runner type string (`"process"`, `"podman"`, `"docker"`)          |
| `backend_id` | resolved backend id (e.g. `"rocm"`)                               |

```python
models = runner.get_v1_models()
print(models)  # → {"object": "list", "data": [{"id": "qwen3-4b", "status": "running", ...}]}
```

---

## LangChain LCEL Integration

ModelArkestra ships with a LangChain adapter that wraps any started model to implement the standard LangChain chat model interface. This enables drop-in compatibility with LangGraph, LangServe, and other LangChain ecosystem tools.

```python
from model_arkestra.arkestra import ModelArkestra
from model_arkestra.langchain_adapter import LangChainModelAdapter

async with ModelArkestra("config.yaml") as runner:
    await runner.start("qwen3-4b")

    adapter = LangChainModelAdapter(runner, "qwen3-4b")

    # ── Blocking invocation ──────────────────────────────
    result = await adapter.ainvoke("What is quantum entanglement?")
    print(result.content)  # → "Quantum entanglement is a phenomenon..."

    # ── Token-by-token streaming ─────────────────────────
    async for chunk in adapter.astream("Write a haiku about code"):
        print(chunk.content, end="", flush=True)
    # → partial tokens accumulating (Hello World!)

    # ── Typed event stream (LangGraph-compatible) ────────
    async for event in adapter.astream_events("Explain photosynthesis"):
        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)
        elif event["event"] == "on_chat_model_end":
            print("\n[done]")
```

### Input types

The adapter accepts all LangChain `LanguageModelInput` variants:

| Input type | Example                                                                                |
|---|---|
| `str` | `"Hello world"`                                                                        |
| OpenAI-style dicts | `{"role": "user", "content": "Hi"}`                                                    |
| List of dicts | `[{"role": "system", "content": "Be nice"}, {"role": "user", "content": "Say hello"}]` |
| LangChain `BaseMessage` list | `[HumanMessage(content="Hi"), AIMessage(content="Hello!")]`                            |
| `PromptValue` | A LangChain prompt template's `.invoke()` output                                       |

The adapter normalizes all inputs to the OpenAI-compatible message format (`[{"role": "...", "content": "..."}, ...]`) and passes the full conversation history to the underlying runner.

### Supported parameters

Both `ainvoke` and `astream` accept:

| Parameter | Type | Description                                                                               |
|---|---|---|
| `input` | `LanguageModelInput` | User input (see above)                                                                    |
| `config` | `RunnableConfig` | LangChain runnable config (passed through, reserved for future use)                       |
| `stop` | `list[str]` | Stop sequences sent to the server                                                         |
| `**kwargs` | — | Additional parameters forwarded to the model (e.g., `temperature`, `max_tokens`, `top_p`) |

### With LangChain Expression Language / LangGraph

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{messages}"),
])

# The adapter works as an LCEL runnable:
chain = prompt | adapter  # or use .bind(stop=["\n"])

result = await chain.ainvoke({"messages": [("user", "What's the weather?")]})
```

---

## OpenAI-Compatible Server (`server.py`)

ModelArkestra ships a production-ready OpenAI v1-compatible API server that sits on top of any started model. It exposes `POST /v1/chat/completions`, `GET /v1/models`, and `GET /health` — all compatible with OpenAI client libraries, LangChain's `OpenAIChat` adapter, and any tool that talks to the OpenAI endpoint.

### Quick Start — CLI

The simplest way to start is as a standalone process:

```bash
python -m model_arkestra.server --config config.yaml --port 8080
```

This launches a FastAPI server backed by ModelArkestra on port 8080. Models load lazily — the first request to `/v1/chat/completions` for a given model triggers its startup.

#### CLI Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--config` / `-c` | *(required)* | — | Path to YAML config file |
| `--port` / `-p` | — | `8080` | HTTP port to listen on |
| `--host` / `-H` | — | `0.0.0.0` | Bind address — use `127.0.0.1` for localhost-only |
| `--ready-timeout` / `-t` | — | `60` | Seconds to wait for models during startup |
| `--alias` / `-a` | — | — | OpenAI model alias mapping (`KEY=VALUE`). Repeat for multiple, e.g. `-a gpt-4=qwen3 -a claude=gemma` |
| `--api-key` | — | — | Require this Bearer token on every request (basic auth bypass) |
| `--cors` | — | `false` | Enable CORS headers |
| `--ssl-certfile` | — | — | Path to TLS certificate file (PEM) |
| `--ssl-keyfile` | — | — | Path to TLS private key file (PEM) |
| `--log-level` | — | `info` | Uvicorn log level |
| `--workers` / `-w` | — | `1` | Number of worker processes |
| `--broadcast-addr` | — | auto | Address models bind to (`0.0.0.0` or `127.0.0.1`) |

### Usage — Embed into an Existing App

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

To run alongside other FastAPI routes:

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

To run as a background thread in an existing process:

```python
server = ArkestraServer("config.yaml")
await server.start_background()  # uvicorn runs in daemon thread
# ... your app logic ...
await server.shutdown()  # stops everything on exit
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completion — blocking or streaming (SSE) |
| `GET` | `/v1/models` | List all tracked models in OpenAI format |
| `GET` | `/health` | Health check — returns running model count |
| `GET` | `/v1/health` | Alias for `/health` (OpenAI compat) |

#### POST /v1/chat/completions

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

Non-streaming response:

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

Streaming response (SSE `text/event-stream`) — one chunk per token:

```jsonlines
data: {"id":"cmpl-stream-default","object":"chat.completion.chunk",
       "model":"qwen3.5-4b","choices":[{"index":0,"delta":{"content":"Quantum"}}]}
data: {"id":"cmpl-stream-default","object":"chat.completion.chunk",
       "model":"qwen3.5-4b","choices":[{"index":0,"delta":{"content":" entanglement..."}}]}
data: {"id":"cmpl-stream-default","object":"chat.completion.chunk",
       "model":"qwen3.5-4b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

#### GET /v1/models

Returns all tracked models in OpenAI format:

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3.5-4b",
      "object": "model",
      "owned_by": "local",
      "status": "running",
      "port": 18000,
      "runner_type": "process",
      "backend_id": null
    }
  ]
}
```

### Model Resolution — Aliases

The `openai_aliases` parameter maps OpenAI-style model IDs to your internal model names:

```python
server = ArkestraServer(
    "config.yaml",
    openai_aliases={"gpt-4": "qwen3.5-4b", "claude-3-opus": "llama3-70b"},
)
```

A request with `model: "gpt-4"` resolves to `"qwen3.5-4b"` automatically. If no alias matches, the model ID is passed through as-is (the runner uses it as the model name).

### Request & Response Models

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

---

## API Reference — `ProcessModelRunner`

For direct use when you only need subprocess-based execution. Bound to a `ConfigManager` instance; does not launch any processes until `start()` is called.

| Parameter | Type | Default | Description                                                                            |
|---|---|---|---|
| `config_manager` | `ConfigManager` | *(required)* | The configuration source for model definitions and global settings.                    |
| `shutdown_timeout` | `float` | `20.0` | Seconds to wait after SIGHUP before escalating to SIGKILL during shutdown.             |
| `ready_timeout` | `float` | `120.0` | Maximum seconds to wait for the server's `/health` endpoint to respond after starting. |
| `ready_poll_ms` | `float` | `100.0` | Milliseconds between consecutive health-check polls during startup.                    |
| `warmup_delay` | `float` | `20.0` | Seconds to wait after `/health` returns OK before marking the model as `"running"`.    |

> `restart_delay` and `restart_limit` control automatic restart behavior inherited from `BaseModelRunner`: when a managed process or container exits unexpectedly, the runner polls and attempts up to `restart_limit` restarts spaced by `restart_delay` seconds. `port_drain_timeout` is used by container runners to wait for port listeners to drain; it has no effect on direct subprocess execution.

Each runner manages exactly one model. Its `stop()` takes no arguments and shuts down that model; `stop_all()` delegates to `stop()`. The remaining methods (`start`, `ainvoke`, `astream`, `request`, `running_models`) follow the same contract as described in the `ModelArkestra` section above.

---

## Process Lifecycle

```
                ┌───────────┐
  start(model) ─│           │──► /health OK + warmup delay ─► state = "running"
                ▼           │
              "loading"     │
                              │ process/container exits unexpectedly
                              │ retry_count < restart_limit?
                              ├─ Yes → sleep(restart_delay) → restart
                              │         (reuses same port)
                              └─ No  → state = "error"
```

### Crash detection by backend

| Runner | Behavior                                                                                                                                                                                                  |
|---|---|
| **Process** | Polls subprocess exit code via `process.wait()`. On unexpected exit (non-zero), automatically attempts up to `restart_limit` restarts spaced by `restart_delay`. The watcher task runs in the background. |
| **Podman / Docker** | Polls container status every 2 seconds; on unexpected exit (`exited`, `dead`), automatically attempts up to `restart_limit` restarts spaced by `restart_delay`. The watcher task runs in the background.  |

### Shutdown sequencing (`stop` / `stop_all`)

1. State transitions to **STOPPING** immediately — prevents any watcher from attempting a restart.
2. Signal is sent:
   - **Process**: SIGHUP to process group → waits up to `shutdown_timeout` (default 20s) → escalates to SIGKILL if still running.
   - **Podman / Docker**: Container stop signal (`podman stop --time <port_drain_timeout>` or `docker stop --time <port_drain_timeout>`) with graceful shutdown timeout (`port_drain_timeout` default 20s).
3. After termination, watcher tasks are cancelled and state transitions to **STOPPED**. Port is released (with `port_drain_timeout` wait for container runners to let the port listener drain).

### Full teardown (`shutdown`)

Identical to stop sequencing with two additions:
1. All watcher tasks are cancelled and awaited.
2. `_models` and `_watchers` dictionaries are cleared — models cannot be restarted after shutdown.
3. **For container runners**: After stopping all containers, force-removes them via `podman rm -f` / `docker rm -f`.

---

## Error Hierarchy

All exceptions inherit from `RunnerError`, enabling callers to catch all runner-related failures with a single clause:

```python
try:
    result = await runner.ainvoke("my-model", prompt="...")
except RunnerError as e:
    logger.error(f"Runner failed: {e}")
```

| Exception | When raised                                                                                                                                                                                                                                                   |
|---|---|
| `ServerReadyTimeout` | The server did not become ready (health check returned no 200 within `ready_timeout`). Raised only on total health-check exhaustion.                                                                                                                          |
| `ModelNotStarted` | A request was made for a model name that doesn't exist in the config, or the model hasn't been started yet (`start()` not called). Raised by `_dispatch` when no tracked context exists, and during `BaseModelRunner.start()` when model config lookup fails. |
| `ModelShutdown` | A request was made to a model that has been explicitly stopped via `stop()` or `stop_all()` (state is STOPPED or STOPPING). Raised by `_dispatch`.                                                                                                            |
| `MaxRestartsExceeded` | The runner crashed and exhausted all automatic restart attempts (`restart_limit`). State transitions to ERROR; raised by `_dispatch` on any subsequent request.                                                                                               |
| `RunnerError` | Generic base class for all runner failures, including HTTP errors (502, 503, 504) from the server, connection failures, and general request errors.                                                                                                           |

---

## Configuration Format (YAML)

The runner reads from the same YAML configuration used by `ConfigManager`. Three sections are relevant:

### Top-level settings

| Key | Type | Default | Description                                                                                                                                                                              |
|---|---|---|---|
| `models-start-port` | `int` | `18000` | First port in the auto-allocated range.                                                                                                                                                  |
| `model-ports` | `int` | `32` | Number of ports available — valid range is `models-start-port` through `models-start-port + model-ports - 1`. Allocation raises `RuntimeError("Port range exceeded: …")` when exhausted. |

### `env:` section — process environment variables

Environment variables defined here are merged into the subprocess/container environment at startup, in addition to the host's current environment. This is how paths like `LLAMA_CACHE` and `HF_HUB_CACHE` propagate to the server process.

```yaml
env:
  LLAMA_CACHE: /home/lemonade/hub
  HF_HUB_CACHE: /home/lemonade/hub
```

### `macros:` section — argument templates

Macros are expanded at boot time. At `start()` runtime the backend `args` template is resolved again with concrete `${PORT}` and `${CHECKPOINT}` values.

```yaml
macros:
  llama-args: --port ${PORT} --jinja -fa on -ngl 999
  ctx-default: 131072
```

> **Note on macro syntax:** Config files use the `${MACRO}` pattern — values are resolved by `llm_config_manager` from the `macros:` section. Shell-style `${VAR:-default}` is **not** supported.

### `backends:` section — executable registry

Each backend entry specifies an argument template and which runner type should handle it.

```yaml
backends:
  vulkan-radv:
    args: ${llama-args}
    runner: process     ← maps this backend to ProcessModelRunner
  rocm:
    args: ${llama-args}
    runner: podman      ← maps this backend to PodmanModelRunner
  default: vulkan-radv         ← global default backend (also has runner: process)
```

When a model has `backend: rocm`, the routing chain resolves: `rocm` → `runner: podman` → `runners.podman` → `PodmanModelRunner`.

**Backend configuration keys:**

| Key | Type | Description                                                                                                                                                                                                                                                     |
|---|---|---|
| `args` | str | Argument template for the llama-server. May reference macros via `${name}`.                                                                                                                                                                                     |
| `runner` | str | Runner type string (e.g. `"process"`, `"podman"`, `"docker"`). Resolved against `runners:` config or built-in registry.                                                                                                                                         |
| `binary_dir` | str | Absolute path to host directory containing the llama-server binary. Direct-process runners use it directly; container runners resolve it indirectly via `resolve_binary_from_backend()` in `common.py` (which also checks `version` and image name heuristics). |
| `binary` | str | Binary name (default: `"llama-server"`). Used with `binary_dir` to form the full path.                                                                                                                                                                          |
| `image` | str | Container image tag for Podman/Docker runners.                                                                                                                                                                                                                  |
| `devices` | list[str] | Device passthrough entries for container runs (e.g. `"/dev/dri/card1:rwm"`).                                                                                                                                                                                    |
| `env_container` | dict | Environment variables passed into the container. Merged on top of global `env:`.                                                                                                                                                                                |
| `version` | str | ROCm/Vulkan version string — used to resolve binary dir from known build directory map (`_ROCM_BUILD_MAP`).                                                                                                                                                     |

### `runners:` section — runner class registry

This section maps runner type strings to concrete class names. Built-in types (`process`, `podman`, `docker`) are auto-registered; config entries can override them or add new ones (as long as the class is importable from this module).

```yaml
runners:
  podman: PodmanModelRunner
  docker: DockerModelRunner
  default: ProcessModelRunner
```

The `default` key specifies the fallback runner type when neither `backends.<id>.runner` nor a model's own `backend` field provides one.

### `models:` section — model definitions

Each model entry uses `checkpoint` for the model path, `args` for flags, and optionally `backend` to override which backend registry entry is used. **There is no `cmd` field** — commands are assembled at start time via the backend registry.

> HuggingFace models: use `-hf ${CHECKPOINT}` in the backend `args` template (see `sample-config.yaml`) so llama.cpp downloads the model instead of treating it as a local file path.

```yaml
models:
  "gpt-oss-20b":
    checkpoint: unsloth/gpt-oss-20b-GGUF:UD-Q4_K_XL
    args: |
      --temp 1.0 --top-k 0 --ctx-size ${ctx-default}

  "qwen3-4b":
    checkpoint: unsloth/Qwen3-4B-GGUF:Q4_K_M
    backend: rocm          ← optional per-model override
    args: |
      --temp 0.7 --top-k 20 --ctx-size ${ctx-default}
```

---

## HTTP Client (`http_client.py`)

The package ships a lightweight `ModelHttpClient` class in `model_arkestra.http_client` that encapsulates aiohttp usage patterns:

| Method | Description                                                                                     |
|---|---|
| `get_json(url)` | GET and return parsed JSON body.                                                                |
| `post_json(url, json_body)` | POST with JSON body and return parsed response.                                                 |
| `post_raw(url, json_body)` | POST and return an async context manager for raw response streaming (SSE, large binaries).      |
| `stream_sse(url, json_body)` | Iterate SSE `data:` lines from a POST endpoint — yields raw strings without the `data:` prefix. |

Sessions are scoped to each call; no manual session management needed.

```python
from model_arkestra.http_client import ModelHttpClient

async with ModelHttpClient(timeout=60) as client:
    data = await client.get_json("http://127.0.0.1:8080/health")
    async for line in client.stream_sse(url, {"prompt": "hi"}):
        print(line)
```

> **Note:** `ModelHttpClient` is a standalone utility — the runner classes use aiohttp directly internally and do not depend on this wrapper. It exists primarily for testing and external integrations.

## Shared Utilities (`common.py`)

Functions used by the runner classes to build commands from configuration data:

| Function | Description |
|---|---|
| `build_model_args(cm, model_name, env_vars=None, override_backend=None) → (list[str], str)` | Builds command arguments for a model using its backend config — resolves the effective backend, fills `${PORT}` and `${CHECKPOINT}` placeholders in the backend args template, and concatenates backend args with model-specific args. Returns `(arg_list, cmd_str)`. |
| `_resolve_backend(cm, model, model_name, override_backend) → str` | Determines which backend ID to use for a model. Resolution order: `override_backend` → `model["backend"]` → `backends.default`. Used internally by `build_model_args`. |

These replace the former `ConfigManager.assemble_command()` and `ConfigManager._resolve_backend_for_model()` methods, keeping command assembly logic inside arkestra where it belongs.

## Running Tests

The project uses `pytest` with a modular fixture system (`tests/conftest.py`) that provides shared model runners, port cleanup, and per-test isolation. All tests are collected from `tests/`, `tests/unit/`, and any module-scoped test files.

```bash
# Run all tests (fast + slow)
python -m pytest -v --tb=short

# Run only fast tests (skips models that download/run LLMs)
python -m pytest -v --tb=short -m "not slow"

# Run only slow integration tests (requires a working GPU/backend or container runtime)
python -m pytest -v --tb=short -m slow
```

**Fixture summary:**

| Fixture | Scope | Purpose                                                                                        |
|---|---|---|
| `mr` | module | Shared `ModelArkestra` instance — port allocation and runner maps are shared across the module |
| `_cleanup_after_test` | function (autouse) | Stops all models after each test so every method sees a clean slate                            |
| `_cleanup_ports` | module (autouse) | Safety net that kills any lingering processes on configured ports before/after each module     |
| `podman_cleanup` | function | Tracks podman containers/tasks/ports with guaranteed teardown — opt-in per-test fixture        |

---

## Import Path

The runner and its backends live in the `model_arkestra` package:

```python
from llm_config_manager.config_manager import ConfigManager    # data layer
from model_arkestra.arkestra import ModelArkestra              # orchestration (recommended)
from model_arkestra.base import BaseModelRunner                # abstract base class
from model_arkestra.process import ProcessModelRunner          # process runner
from model_arkestra.podman import PodmanModelRunner            # podman runner
from model_arkestra.docker import DockerModelRunner            # docker runner
from model_arkestra.container_runner import ContainerModelRunner  # container base class
from model_arkestra.http_client import ModelHttpClient         # lightweight HTTP client
from model_arkestra.langchain_adapter import LangChainModelAdapter  # LangChain LCEL wrapper
from model_arkestra.server import ArkestraServer             # OpenAI v1-compatible API server

# Convenience re-exports from __init__.py:
from model_arkestra import RunnerState, RunnerError, ServerReadyTimeout
from model_arkestra import ModelNotStarted, MaxRestartsExceeded, ModelShutdown
```
