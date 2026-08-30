# Configuration Format (YAML)

ModelArkestra uses **two configuration files** that work together:

| File | Purpose | User Editability |
|------|---------|-----------------|
| `~/.config/arkestra/config.yaml` | Model definitions, ports, global defaults | Full — edit freely |
| `~/.config/arkestra/backends.yaml` | Backend types, binary sources, download channels | **Advanced** — see below |

Both files are scaffolded together by `model-arkestra init`. They live in the XDG-compliant directory `~/.config/arkestra/`, resolvable via `resolve_config_path()` and `resolve_backends_path()`.

```
~/.config/arkestra/
├── config.yaml        ← your models + global settings
└── backends.yaml      ← backend definitions + download sources
```

## Quick Start: The Init Command

```bash
model-arkestra init --force          # writes both files from scratch
model-arkestra list-backends         # shows available backends + binary status
model-arkestra download-backend rocm  # downloads the ROCm binary
model-arkestra start                 # validates backends, starts server
```

**Detection flow:** `init` probes your hardware (GPU vendor, CPU arch), then writes a `config.yaml` with `backends.default:` pointing to the best backend for your machine. The full backend definitions live in `backends.yaml`.

## Configuration Commands

| Command | Purpose |
|---------|---------|
| `model-arkestra init [--force]` | Scaffold both config files; sets default backend from detection |
| `model-arkestra detect` | Read-only hardware report — no file changes |
| `model-arkestra list-backends` | Table of backends: type, description, cached binary status |
| `model-arkestra add-backend -l /path/to/binary [-n name] [-d desc]` | Add a custom local llama-server binary |
| `model-arkestra remove-backend <name>` | Delete a backend from backends.yaml |
| `model-arkestra download-backend <name> [--version TAG]` | Download a pre-built binary for a backend |
| `model-arkestra download-all` | Auto-detect + download primary + fallback backends |

---

## `config.yaml` — Model Configuration

This file defines models, ports, and global settings. Backend selection is minimal — just pick the default:

```yaml
models-start-port: 18000
model-ports: 32
warmup-time: 10

app-log-lines: 2000

env:
  HF_HUB_CACHE: ~/.cache/huggingface

backends:
  default: rocm        # picks a backend from backends.yaml

models:
  qwen3.8-27b:
    repo: hugging-face
    model: unsloth/Qwen3.8-27B-GGUF:Q4_K_M
    args:
      temp: 0.7
      top-p: 0.80
      top-k: 20
      presence-penalty: 1.5
      chat-template-kwargs: '{"reasoning_effort":"medium"}'

# ── Auxiliary models (ONNX) run on a separate inference server ────

  bge-embeddings:
    model_path: /path/to/model.onnx
    type: embedding
    tokenizer: Xenova/bge-small-en-v1.5
    capabilities: [embed]
    port: 8090   # separate from LLM port range

  qwen3.6-27b-mtp:
    repo: hugging-face
    model: unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL
    args:
      temp: 0.6
      top-p: 0.95
      top-k: 20
      chat-template-kwargs: '{"enable_thinking":true}'
```

### Auxiliary Models (ONNX)

Models marked with `capabilities` run on a separate ONNX inference server, preserving GPU VRAM for LLM inference.

| Capability | Value | Endpoint | Model Type |
|---|---|---|---|
| `embed` | `["embed"]` | `/v1/embeddings` | Embedding encoder (BERT-style) |
| `stt` | `["stt"]` | `/v1/audio/transcriptions` | Whisper ASR |
| `tts` | `["tts"]` | `/v1/audio/speech` | Kokoro TTS |

ONNX model keys:
- `model_path: /path/to/model.onnx` — path to ONNX model file (required)
- `type: embedding|whisper|tts` — inference type (required)
- `tokenizer: Xenova/bge-small-en-v1.5` — HF tokenizer repo or local dir (embedding only)
- `port:` — HTTP port for the ONNX server instance (separate from LLM port range)

See [ONNX Server](./onnx-server.md) for full documentation.

### Top-Level Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `models-start-port` | `int` | `18000` | First port in the auto-allocated range. |
| `model-ports` | `int` | `32` | Number of ports available — valid range is `models-start-port` through `models-start-port + model-ports - 1`. |
| `warmup-time` | `float` | `10.0` | Seconds to wait after `/health` returns OK before marking the model as `"running"`. |
| `app-log-lines` | `int` | `2000` | Number of server-level log entries retained in the global ring buffer (Admin API → `/admin/logs`). Each entry ~200 bytes. |
| `env` | `dict` | — | Environment variables merged into model subprocesses at startup (e.g., `HF_HUB_CACHE`). |
| `container_type` | `str` | `"process"` | Default container runner when a backend uses `runner: container`. Valid values: `"podman"`, `"docker"`. Set to `"process"` to disable containers by default.

### `backends.default:` Key

This single key selects which backend from `backends.yaml` is used as the default for all models that don't specify a `backend` override. It does **not** define the backend — that lives in `backends.yaml`.

```yaml
backends:
  default: rocm   # references an entry in backends.yaml
```

### `models:` Section — Model Definitions

Each model uses `repo:` + `model:` for the HF reference, `args:` for CLI flags, and optionally a `backend` override.

#### `args:` Field Format

The `args:` key is a **flat YAML dict** where each key is a CLI flag name (kebab-case):

| Style | YAML Entry | Flag Output |
|---|---|---|
| Standard value | `temp: 0.7` | `--temp 0.7` |
| Boolean `true` | `jinja: true` | `--jinja` (presence-only) |
| Boolean `false` | `no-mmap: false` | omitted entirely |
| HuggingFace repo | `hf: user/repo:Q4_K_M` | `-hf user/repo:Q4_K_M` |

> **Note:** YAML interprets bare `on` and `off` as booleans. Always quote them (`"on"`, `"off"`) for flags that accept strings.

```yaml
models:
  qwen3-4b:
    repo: hugging-face
    model: unsloth/Qwen3-4B-GGUF:Q4_K_M
    backend: rocm            # optional override (defaults to backends.default)

    args:
      temp: 0.7
      top-p: 20
      ctx-size: 16384
      jinja: true

    max_log_lines: 500        # per-model log buffer size
    capabilities: ["chat"]     # non-arg — shown in admin UI, not passed to subprocess
```

---

## `backends.yaml` — Backend Definitions & Download Sources

> ⚠️ **Advanced:** This file is auto-generated on `init`. Edit with caution if you're not familiar with backend configuration. Use `model-arkestra add-backend` for custom binaries.

```yaml
backends:

  vulkan-radv:
    description: "Vulkan with RADV driver — works on AMD, NVIDIA, Intel"
    runner: ProcessModelRunner
    source_ref: official-vulkan-radv
    args:
      ngl: 999
      ctx-size: ${ctx-size}

  rocm:
    description: "ROCm — best for AMD iGPU and discrete GPUs"
    runner: ProcessModelRunner
    source_ref: lemonade-rocm-nightly
    args:
      ngl: 999
      ctx-size: ${ctx-size}

  cuda:
    description: "NVIDIA CUDA — for NVIDIA discrete GPUs"
    runner: ProcessModelRunner
    source_ref: official-cuda
    args:
      ngl: 999
      ctx-size: ${ctx-size}

  cpu:
    description: "CPU-only mode — uses all available cores"
    runner: ProcessModelRunner
    source_ref: ggml-org-cpu
    args:
      threads: ${nproc}
      no-mmap: true


# ── Download sources (referenced by backends above) ───────────────

sources:

  official-vulkan-radv:
    type: github-release
    repo: ggml-org/llama.cpp
    release_type: latest
    asset_pattern: "llama-server-*-bin-*-vulkan*"
    sha256_asset: "*.sha256"

  lemonade-rocm-nightly:
    type: github-release
    repo: lemonade-sdk/llamacpp-rocm
    release_type: latest
    asset_pattern: "*-linux-x86_64.tar.gz"
    sha256_asset: "*.sha256"

  official-cuda:
    type: github-release
    repo: ggml-org/llama.cpp
    release_type: latest
    asset_pattern: "llama-server-*-bin-*-cuda*"
    sha256_asset: "*.sha256"

  ggml-org-cpu:
    type: github-release
    repo: ggml-org/llama.cpp
    release_type: latest
    asset_pattern: "llama-server-*-bin-*-static*"
    sha256_asset: ""


defaults:
  release_type: latest
  verify_checksum: true
  cache_ttl_hours: 24
```

### Backend Entry Keys

| Key | Type | Description |
|---|---|---|
| `description` | str | Human-readable description shown in `list-backends`. |
| `runner` | str | Runner type: `"process"`, `"podman"`, `"docker"`, `"container"`, or `"remote"` (**legacy** — use `clusters:` top-level key for federation). Use `"container"` to defer to the top-level `container_type:` config value. |
| `base_url` | str | (Legacy `runner: remote` only) URL of the target arkestra worker. Prefer the `clusters:` top-level key instead. |
| `admin_key` | str | (Remote optional) API key forwarded as `x-admin-key` header to workers requiring authentication. If the target worker also proxies, forward its `admin_key` value here.
| `source_ref` | str | Name of a source entry from the `sources:` section below. |
| `args` | dict | Default CLI arguments merged into model args during startup. |
| `hf_flag` | str | (Optional) Override for the HuggingFace flag format — e.g., `"--hf"` instead of default `"-hf"`. Used when container images or binaries use a different flag convention. |
| `entrypoint` | str | (Container only) Override the container's ENTRYPOINT — e.g., `/llama.cpp/llama-server`. Prevents image defaults (like `tini`) from intercepting CLI args. |
| `binary_dir` | str | (Written at runtime) Absolute directory containing the downloaded binary, populated by `download-backend`. |
| `binary` | str | (Written at runtime) Binary filename, populated by `download-backend`. |

### Source Entry Keys

| Key | Type | Description |
|---|---|---|
| `type` | str | Source type: `"github-release"`, `"oci-image"`, or `"local-file"`. |
| `repo` | str | (For github-release) Owner/repo on GitHub (e.g., `"ggml-org/llama.cpp"`). |
| `release_type` | str | `"latest"` for newest tag via API, or a pinned version like `"v2.95"`. |
| `asset_pattern` | str | Glob pattern to match the desired download asset. |
| `sha256_asset` | str | Pattern for checksum sidecar file; empty string skips verification. |
| `registry` | str | (For oci-image) Container registry hostname. |
| `tag` | str | (For oci-image) Image tag to pull. |
| `path` | str | (For local-file) Absolute path to a pre-built binary. |

### Custom Backend Entries (User-Added)

Add custom backends via the CLI:

```bash
model-arkestra add-backend --local /opt/my-builds/llama-server \
                           --name my-avx512 \
                           --description "Custom AVX512 build"
```

This appends to the `backends:` section of backends.yaml:

```yaml
my-avx512:
  description: "Custom AVX512 build"
  runner: ProcessModelRunner
  args:
    ngl: 999
    ctx-size: ${ctx-size}
```

Then select it in config.yaml:

```yaml
backends:
  default: my-avx512
```

### Federated Clusters

The `clusters:` top-level key defines managed arkestra instances. Model names prefixed `<cluster>/<model-id>` route to the matching cluster:

```yaml
clusters:
  local:                          # auto-created from server host/port
    base-url: "http://127.0.0.1:18000"
  gpu-server:
    base-url: "http://192.168.1.42:18000"
  cpu-worker:
    base-url: "http://192.168.1.43:8080"

models:
  gpu-server/gemma-4b:
    repo: hugging-face
    model: unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
    backend: rocm
  cpu-worker/whisper-large:
    repo: hugging-face
    model: distil-whisper/distil-small.en
    backend: cpu
```

**How it works:**
- The master **never downloads, spawns, or allocates ports** for remote-cluster models.
- All requests proxy through the cluster's `base-url`.
- Model names use `<cluster>/<model-id>` convention (e.g., `gpu-server/qwen3`) to identify routing.
- Local cluster uses port pool for subprocesses; remote clusters proxy all traffic.

> **Legacy compatibility**: Models prefixed `<worker>/<model-id>` with a backend entry
> having `runner: remote` + `base_url:` continue to work without a `clusters:` block.

---

## Backend Resolution & Defaults Chain

Arguments follow a six-layer resolution cascade — each layer fills values missing from the one above:

1. `**overrides` passed to `start()` — transient, single invocation only
2. Model-level `args:` dict — explicit per-model overrides
3. `defaults:` section at top level of config.yaml — global shared defaults
4. Backend entry `args:` from backends.yaml — inherited by all models using it
5. Source-level defaults (cache TTL, checksum verification)
6. Hardcoded fallbacks

Backend selection resolution:

1. Model's `backend:` field (if specified)
2. `backends.default:` key in config.yaml
3. `"vulkan-radv"` hardwired fallback

Runner resolution (for ProcessModelRunner, always process):

1. Backend entry's `runner:` field
2. Hardwired `"ProcessModelRunner"` for process mode

---

## Related Documentation

- [Usage Guide](./usage.md) — how to load and use config files
- [Architecture](./architecture.md) — runner routing based on backend config
- [Admin API](./admin.md) — runtime config management via HTTP endpoints
- [Lifecycle](./lifecycle.md) — state transitions for model runners
