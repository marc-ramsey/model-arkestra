# Architecture

Model Arkestra handles port allocation and backend→runner routing.

> **Lifecycle vs inference.** Since the core restructure, a `Model` object owns
> its state and its *provider* (inference). Runners/backends own only launch,
> stop, and crash-watching. See [Inference Providers & Capabilities](#inference-providers--capabilities) below.

## Component Diagram

```
┌─────────────────────────────────────────────────┐
│                  ModelArkestra                   │
│  ── port allocation, backend→runner routing      │
│                                                  │
│  __build__runner_class_map() → config-driven     │
│  runners: section maps type → class              │
│  get_runner_instance(type, model) → lazy factory│
└──────┬──────────────────┬───────────────────────┘
       │                  │
       ▼                  ▼
┌──────────────┐  ┌────────────────────────┐
│ProcessRunner │  │ ContainerRunner   │
│              │  │     (abstract base)     │
│subprocesses  │  ├────────────┬───────────┤
│              │  │PodmanRunner│ DockerRunner│
└──────┬───────┘  │            │           │
       │          │OCI containers│OCI cont.│
       │          └────────────┴───────────┘
       ▼
┌─────────────────────────────────────────────┐
│  llama.cpp / Container Process              │
│  (llama-server on ROCm/Vulkan/CUDA)         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  ONNX Inference Server (optional)           │
│  ── separate process, distinct port range    │
│                                              │
│  /v1/embeddings   → BERT-style encoders     │
│  /v1/audio/*      → Whisper / Kokoro        │
│                                              │
│  --device cpu   → CPUExecutionProvider       │
│  --device npu   → NPUExecutionProvider (AMD) │
└─────────────────────────────────────────────┘
```

## Runner Routing — Explicit Args Override Config

Runner type resolution follows a strict precedence: **explicit arguments take priority over all configuration values**. An explicit runner type that does not exist in the registry is rejected immediately.

### Explicit override (highest priority)

When `runner=` is supplied, that value is used directly:

```
runner="podman" → PodmanRunner   (verified against registry)
runner="docker" → DockerRunner   (verified against registry)
runner="process" → ProcessRunner  (verified against registry)
runner="onnx" → OnnxRunner             (ONNX inference server)
```

When `backend=` is supplied without `runner=`, the backend's ``runner:`` field (or its engine mapping) determines the runner type.

### Automatic resolution (config chain)

When no explicit argument is given, the runner type is resolved through config:

```
backends section exists?
  └─> model.backend → backends.<id>.runner → runners:<type> (or "process")
backends section absent?
  └─> runners:<default> (or "process")
```

See [Configuration Format](../config.md) for how `backends:` and `runners:` sections work.

## Naming Conventions

All names are deterministic and derived from configuration — no random or time-based suffixes. This makes debugging, log searching, and container management straightforward.

### Container Image Tags

| Name | Pattern | Source |
|---|---|---|
| ROCm image | `ark-llama:rocm` | `backends.rocm.image` in backends.yaml |
| Vulkan image | `ark-llama:vulkan-radv` | `backends.vulkan-radv.image` in backends.yaml |

The `backends.default.image` value is the fallback when no backend specifies an ``image:``.

### Container Names (Podman / Docker)

Container runtime names follow this pattern:

```
llm-{model-name}-{port}
```

- `llm` — fixed prefix identifying Arkestra-managed containers
- `{model-name}` — the model name from config (underscores → hyphens, dots → hyphens)
- `{port}` — the port assigned to that model instance

Example: a model named `qwen3.5-4b` on port 18001 becomes `llm-qwen3-5-4b-18001`.

Implemented by `safe_container_name()` in `model_arkestra/common.py`. The same name is used for both Podman and Docker runners via `_build_podman_cmd()` / `_build_docker_cmd()`.

### Backend IDs

Backend identifiers serve as the single naming hub — they connect config, images, and Containerfiles:

| Context | Reference |
|---|---|
| `backends:` key in YAML | `rocm`, `vulkan-radv` |
| Image tag (per-backend) | Same IDs; each defines its ``image`` tag in backends.yaml |
| Runner type dispatch | `runner: podman` + backend ID → resolves to correct ContainerRunner subclass |

The backend ID never changes across the system — it is the anchor that ties config, images, and runtime execution together.

## Port Allocation — A Global Counter Controlled by ModelArkestra

The starting port and range are driven entirely by config:

| Config key | Purpose |
|---|---|
| `model-start-port` (in `default:`) | First port in the range (default 18000) |
| `model-ports` (in `default:`) | Size of the pool — valid ports are ``start_port`` through ``start_port + model-ports - 1`` |

When `ModelArkestra.start()` is called without an explicit `port`, it allocates the next available number from this range sequentially. Once all ports in the pool are exhausted, `RuntimeError("Port range exceeded: …")` is raised immediately.

A **direct** runner instance (e.g. ``ProcessRunner(cm)``) does **not** use a global counter — it picks ``model-start-port`` as the default for its first model, and subsequent calls reuse existing ports via `_dispatch()` / in-place restart.

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

See [API Reference — ModelArkestra](../api/model-arkestra.md) for full method signatures including `start()`.

## Process Lifecycle

For crash detection, shutdown sequencing, and teardown behavior, see [Lifecycle](../lifecycle.md).

### Checkpoint Pulling

`arkestra-admin pull <model>` (or `POST /admin/pull`) downloads a model's weights. The server never blocks: it spawns the download as a background task and returns at once; the CLI polls `GET /admin/pull-status/{model}` and renders progress until done.

The download is driven by `model_arkestra/hf_gguf.py`, which ports llama.cpp's file-selection rules (`common/download.cpp`) to resolve a `repo:quant` ref to the *exact* GGUF file set the engine will load — primary model, optional `mmproj` (vision) and `mtp` (speculative) sidecars, plus split shards. Only those files are fetched into the standard HF cache layout that `llama-server -hf repo:quant` reads, so a pull never drags in an entire repository.

## Argument Passing and Resolution

CLI arguments flow through a two-phase pipeline:

1. **Merge**: ``build_model_args()`` merges model-level ``args:`` with runtime inference kwargs into a flat dict.
2. **Convert**: The engine layer (e.g. ``LlamaCppEngine.build_cli_args(merged, port)``) converts the dict to CLI tokens.

Values are stored as flat YAML dicts internally and converted to CLI flags by the engine layer (e.g. ``LlamaCppEngine.build_cli_args(merged, port)``).

### Merge Phase

The merged dict contains only two sources — model args from config plus runtime inference kwargs (last-wins):

1. Model-level ``args:`` dict — explicit per-model overrides
2. Runtime ``inference_kwargs`` passed to ``start()`` — transient, single invocation only

### Engine → Target Resolution

Each backend name (e.g., `rocm`, `vulkan-radv`) implies an inference engine (llama.cpp). The engine is resolved from the backend name at runtime — users do not specify it separately. This allows each engine to maintain its own target registry and default chain without cluttering user-facing config.

### Inference Param Filtering (`LlamaCppEngine`)

For llama.cpp backends, inference kwargs are filtered through `LlamaCppEngine.LLAMA_INFER_ARGS` before being passed to ``LlamaCppEngine.build_cli_args()``. Only keys in this whitelist (e.g. `temp`, `top-p`, `reasoning-budget`) become CLI flags; anything else is silently dropped to prevent crashes from bogus POST body fields.

### CLI Reconstruction (`LlamaCppEngine.build_cli_args`)

``LlamaCppEngine.build_cli_args(merged, port)`` in `model_arkestra.llama_cpp` converts a flat dict to CLI flags. It handles model/repo/mmproj/port injection plus all remaining keys as kebab-case flags:

| YAML Entry | Reconstructed Flag |
|---|---|
| `temp: 0.7` | `--temp 0.7` (value appended) |
| `jinja: true` / `jinja: false` | `--jinja` (boolean True → presence-only, False → omitted) |

Keys use kebab-case in YAML, matching CLI flag names directly.

Infrastructure flags (`--port`, `--model`) are handled by ``LlamaCppEngine.build_cli_args()``. Internally everything stays structured as dicts until CLI conversion time.

## Inference Providers & Capabilities

A `Model` is the single owner of per-model state (state, port, log ring) **and**
of inference. It holds exactly **one provider**, chosen by the model's runner
kind. Runners never touch inference; they only launch/stop/watch the process or
container.

``Model.provider`` builds the provider on first use and caches it:

```
runner_type == "remote"    → RemoteProvider   (proxied to a cluster worker)
runner_type == "onnx"      → OnnxProvider     (in-process ONNX session, no transport)
anything else              → LlamaProvider    (local llama-server via its HTTP API)
```

The provider class is **engine/location-based**, not mechanism-based. ONNX runs
in-process with no transport; Llama and Remote use network transports. The name
reflects *what provides the inference*, not *how it is reached*.

### Why one provider per model, not one per modality

A single provider owns all of a model's modalities and gates each by capability.
This keeps the ownership surface tiny (one object on `Model`) while still letting
each modality be added independently: a new modality is a new method on the
provider plus a new capability in its set. Capability derivation stays trivial —
a provider advertises what it can do, and `Model.capabilities` mirrors it.

### The Provider interface

All providers implement a common contract (`model_arkestra.providers.base`):

| Method | Purpose |
|---|---|
| `probe()` | Health/availability check |
| `invoke_full(prompt, **kw)` | Blocking completion (full response dict) |
| `stream(payload)` | Async iterator of stream chunks |
| `embed(text)` | Embedding for one text |
| `transcribe(audio_bytes, language)` | Speech → text |
| `synthesize(text, voice, speed)` | Text → speech bytes |
| `request(path, **kw)` | Raw passthrough to the underlying endpoint |

Methods a provider does not support raise `NotSupported`. The facade checks
capability before dispatching, so unsupported calls fail fast with a clear error.

### Capability derivation

Capabilities are **derived**, not stored, from the model's config:

| Source | Capabilities |
|---|---|
| ONNX `type: embedding` | `{embed}` |
| ONNX `type: whisper` / `asr` | `{asr}` |
| ONNX `type: tts` | `{tts}` |
| llama-cpp (default) | `{chat, embed}` |
| llama-cpp + explicit `capabilities: [embed]` | `{embed}` (the one override) |
| remote cluster model | mirrors the worker (empty locally until probed) |

The single override exists because some llama models are embedding-only and must
not advertise `chat`. Everything else is computed.

### How image/video slot in later

- **Vision chat** (image in → text out): *no new provider.* It is a chat call
  with an image content part; the backend just needs `mmproj`. Capability stays `chat`.
- **Image generation** (text → image): a new `generate_image()` method on the
  relevant provider, capability `image-gen`, likely a new backend. Derivation via
  `type: image-gen` or a backend that advertises it.
- **Video**: same pattern — a new method + capability `video`.

Each modality = one provider method + one capability (+ optionally a backend).
Adding video never forces a rewrite of chat/embed/asr.

### Facade routing

The public facade methods are thin routers to the active provider, gated by
capability:

```
arkestra.ainvoke / .astream      → LlamaProvider.invoke_full / .stream  (chat)
arkestra.embed                   → provider.embed                       (embed)
arkestra.transcribe              → provider.transcribe                  (asr)
arkestra.synthesize              → provider.synthesize                  (tts)
```

`ainvoke`/`astream` remain stable shims for the LangChain adapter. New modalities
get a facade method + a CLI subcommand for free.

### CLI mapping

Each CLI subcommand binds one capability, gated before dispatch:

```
arkestra chat <model>            → provider.invoke_full / .stream
arkestra embed <model> "text"    → provider.embed
arkestra transcribe <model> f.wav→ provider.transcribe
arkestra tts <model> "text"      → provider.synthesize
```

## Related

- [Log Ring Buffer](log-ring-buffer.md) — fixed-capacity ring buffer for model log capture
