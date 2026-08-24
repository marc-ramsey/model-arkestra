# Architecture

Model Arkestra handles port allocation and backend→runner routing.

## Component Diagram

```
┌─────────────────────────────────────────────────┐
│                  ModelArkestra                   │
│  ── port allocation, backend→runner routing      │
│                                                  │
│  __build__runner_class_map() → config-driven     │
│  runners: section maps type → class              │
│  _get_runner_instance(type, model) → lazy factory│
└──────┬──────────────────┬───────────────────────┘
       │                  │
       ▼                  ▼
┌──────────────┐  ┌────────────────────────┐
│ProcessModel  │  │ ContainerModelRunner   │
│    Runner    │  │     (abstract base)     │
│              │  ├────────────┬───────────┤
│subprocesses  │  │PodmanModel │ DockerModel│
│              │  │   Runner   │  Runner   │
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
runner="podman" → PodmanModelRunner   (verified against registry)
runner="process" → ProcessModelRunner  (verified against registry)
```

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
| ROCm image | `ark-llama:rocm` | `images.rocm.image` in config.yaml |
| Vulkan image | `ark-llama:vulkan-radv` | `images.vulkan-radv.image` in config.yaml |

The `default-image` top-level key is the fallback when no backend specifies an `image:`. Defined and documented in [Configuration Format](../config.md).

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
| Image registry key (`images:`) | Same IDs; each defines its `image` tag and `containerfile` |
| Runner type dispatch | `runner: podman` + backend ID → resolves to correct ContainerModelRunner subclass |

The backend ID never changes across the system — it is the anchor that ties config, images, and runtime execution together.

## Port Allocation — A Global Counter Controlled by ModelArkestra

The starting port and range are driven entirely by config:

| Config key | Purpose |
|---|---|
| `models-start-port` | First port in the range (default 18000) |
| `model-ports` | Size of the pool — valid ports are `start_port` through `start_port + model-ports - 1` |

When `ModelArkestra.start()` is called without an explicit `port`, it allocates the next available number from this range sequentially. Once all ports in the pool are exhausted, `RuntimeError("Port range exceeded: …")` is raised immediately.

A **direct** runner instance (e.g. `ProcessModelRunner(cm)`) does **not** use a global counter — it picks `models-start-port` as the default for its first model, and subsequent calls reuse existing ports via `_dispatch()` / in-place restart.

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

## Argument Passing and Resolution

CLI arguments flow through a defaults cascade before reaching subprocess construction. Values are stored as flat YAML dicts internally and reconstructed to CLI flags via ``_dict_to_cli()`` inside `build_model_args()`.

### Defaults Cascade

Each layer fills in values, with later layers overriding earlier ones for overlapping keys:

1. `**inference_kwargs` — transient runtime values (single invocation), merged into model args dict
2. Model-level `args:` dict — explicit per-model overrides
3. `defaults:` top-level config section — global shared defaults
4. Backend `args:` dict — per-backend fallback values
5. Engine + runner type defaults — inference engine and execution container baselines
6. Hardcoded fallbacks on the base runner class

### Engine → Target Resolution

Each backend name (e.g., `rocm`, `vulkan-radv`) implies an inference engine (llama.cpp). The engine is resolved from the backend name at runtime — users do not specify it separately. This allows each engine to maintain its own target registry and default chain without cluttering user-facing config.

### Inference Param Filtering (`LlamaCppEngine`)

For llama.cpp backends, inference kwargs are filtered through `LlamaCppEngine.LLAMA_INFER_ARGS` before reaching `_dict_to_cli()`. Only keys in this whitelist (e.g. `temp`, `top-p`, `reasoning-budget`) become CLI flags; anything else is silently dropped to prevent crashes from bogus POST body fields.

### CLI Reconstruction (`_dict_to_cli()`)

The `_dict_to_cli()` helper inside `build_model_args()` converts a merged args dict into a subprocess-compatible command (llama.cpp only — container runners have their own path):

| YAML Entry | Reconstructed Flag |
|---|---|
| `temp: 0.7` | `--temp 0.7` (value appended) |
| `jinja: true` / `jinja: false` | `--jinja` (boolean True → presence-only, False → omitted) |

Keys use kebab-case in YAML, matching CLI flag names directly.

Infrastructure flags (`--port`, `--model`) are added by the runner from context metadata. The conversion happens inside ``build_model_args()`` which is called once per model start — this is the only place where arguments become strings. Internally everything stays structured as dicts. For llama.cpp, a single whitelist (`LLAMA_INFER_ARGS`) gates what inference kwargs reach CLI construction.

## Related

- [Log Ring Buffer](log-ring-buffer.md) — fixed-capacity ring buffer for model log capture
