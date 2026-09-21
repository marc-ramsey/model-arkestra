# Configuration Format (YAML)

ModelArkestra uses **one user file and one shipped file**:

| File | Purpose | Editability |
|------|---------|-------------|
| `~/.config/arkestra/config.yaml` | Models, checkpoints, defaults, backend selection + overrides | Full — the only user-managed file |
| `<package>/data/backends.yaml` | Stock backend definitions (run config + binary source) | Read-only — ships with the package; update via package upgrade |

`arkestra init` scaffolds `config.yaml`. Backend definitions and engine arg
schemas (`<package>/data/schemas.yaml`) are loaded from package data at startup
and deep-merged **under** config.yaml: any key you set in config.yaml wins.

```
~/.config/arkestra/
└── config.yaml        ← everything user-owned

<package>/data/       ← read-only, version-matched to the code
├── backends.yaml      ← stock backend definitions (run config + binary source)
└── schemas.yaml       ← engine arg schemas (admin UI field types)
```

## Quick Start: The Init Command

```bash
arkestra init --force          # writes config.yaml from scratch
arkestra start                 # validates backends, starts server
```

**Detection flow:** `init` probes your hardware (GPU vendor, CPU arch), then writes a `config.yaml` with `backends.default:` pointing at the best stock backend for your machine.

## Configuration Commands

| Command | Purpose |
|---------|---------|
| `arkestra init [--force]` | Scaffold config.yaml; sets default backend from detection |
| `arkestra detect` | Read-only hardware report — no file changes |
| `arkestra list-backends` | Table of backends: type, description, cached binary status |
| `arkestra-bin list` | Binary slots: backend, tag, pin, sha256 |
| `arkestra-bin fetch [backend ...]` | Download + install latest (or pinned) release into a slot |
| `arkestra-bin pin <backend> <tag>` / `unpin <backend>` | Pin a remote backend to a release tag / follow latest again |

---

## `config.yaml` — Model Configuration

This file defines models, ports, and global settings. Backend selection is minimal — just pick the default:

```yaml
default:
  model-start-port: 18000     # first port in the auto-allocated range
  model-ports: 32              # number of ports available
  warmup-time: 10.0            # seconds after /health before "running"
  stream-sock-timeout: 120     # seconds to wait for the next SSE chunk (default 120)
  app-log-lines: 2000          # global log ring buffer size
  model-repo: unsloth           # default HuggingFace repo owner
  model-quant: Q4_K_M          # default quantizer suffix

default-env:
  admin_key: whatever
  hf-hub-cache: ~/.cache/huggingface

backends:
  default: rocm        # picks a stock backend (shipped definition)

models:
  qwen3.8-27b:
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
| `asr` | `["asr"]` | `/v1/audio/transcriptions` | Whisper ASR |
| `tts` | `["tts"]` | `/v1/audio/speech` | Kokoro TTS |

ONNX model keys:
- `model_path: /path/to/model.onnx` — path to ONNX model file (required)
- `type: embedding|whisper|tts` — inference type (required)
- `tokenizer: Xenova/bge-small-en-v1.5` — HF tokenizer repo or local dir (embedding only)
- `port:` — HTTP port for the ONNX server instance (separate from LLM port range)

See [ONNX Server](./onnx-server.md) for full documentation.

### `default-env:` — Runtime Environment Variables

All env-var-backed settings are declared here in **kebab-case**. Values are merged with the actual process environment at startup into a computed `_env` section (never persisted to disk).

Resolution: constructor arg → `_env[key]`.

| Key | Process Env Var | Description |
|---|---|---|
| `admin_key` | `ADMIN_KEY` | Admin panel API key — gates `/admin/*` paths. |
| `api_key` | `API_KEY` | Public API key — gates `/api/*` paths. |
| `hf-hub-cache` | `HF_HUB_CACHE` | HuggingFace model cache directory. |

Keys can also be overridden at runtime via the actual process environment variable — values in `_env` reflect the merged state of YAML defaults and process env.

#### Example:
```yaml
default-env:
  admin_key: "supersecret"
  hf-hub-cache: /data/hf-cache
```

## Configuration Keys

### `default:` — Operational Defaults

| Key | Type | Default | Description |
|---|---|---|---|
| `url` (in `default:`) | `str` | `http://127.0.0.1:8080` | Public address `scheme://host:port/prefix`. The path component is the URL prefix. Env: `ARKESTRA_URL`, CLI: `--url`. |
| `bind` (in `default:`) | `str` | `127.0.0.1` | Server bind address — `127.0.0.1` for local only, `0.0.0.0` to expose on the LAN. CLI: `--bind`. |
| `model-start-port` (in `default:`) | `int` | `18000` | First port in the auto-allocated range. |
| `model-ports` (in `default:`) | `int` | `32` | Number of ports available — valid range is `model-start-port` through `model-start-port + model-ports - 1`. |
| `warmup-time` (in `default:`) | `float` | `10.0` | Seconds to wait after `/health` returns OK before marking the model as `"running"`. |
| `stream-sock-timeout` (in `default:`) | `float` | `120.0` | Max seconds of silence between stream chunks before the request is aborted. Raise for large models whose prefill exceeds the default. |
| `app-log-lines` (in `default:`) | `int` | `2000` | Number of server-level log entries retained in the global ring buffer. |
| `container-type` | `str` | `"process"` | Default container runner when a backend uses `runner: container`. Valid values: `"podman"`, `"docker"`. Set to `"process"` to disable containers by default.

### `backends.default:` Key

This single key selects which stock backend (shipped definition) is the default for all models that don't specify a `backend` override. It does **not** define the backend.

```yaml
backends:
  default: rocm   # references a shipped stock backend
```

### `models:` Section — Model Definitions

Each model uses a single `model:` field for the reference, `args:` for CLI flags, and optionally a `backend` override.

#### Model Field Syntax

```
[<ark-path>:][<repo>:]<model>[:<quant>]
```

| Input | Resolves to |
|---|---|
| `unsloth/Qwen3.5:Q4_K_M` | Full explicit — no expansion |
| `fb:gemma-4e2b` | Alias `fb` → expands from `model-repos:` registry |
| `Qwen3.5:Q4_K_M` | Bare name + quant — applies default repo & quant |
| `gemma-4-e2b` | Bare name — applies default repo, appends default quant |
| `lcl:/data/model.gguf` | Local path (lcl: prefix) |
| `/data/model.gguf` | Auto-tagged as local path |

#### Repo Type Resolution Chain

The source type (`hf` or `lcl`) is determined by:
1. Parse raw string — `/path` → `lcl`, else tentatively `hf`
2. Model-level `repo:` field (if specified)
3. Backend-level `repo:` in args dict
4. Default `model-repo` from `default:` section
5. Hardwired fallback: `"hf"`

#### Alias Registry (`model-repos:`)

Short aliases expand to full repo paths:

```yaml
model-repos:
  fb:   # alias key
    name: foo-bar   # expands to "foo-bar"
```

Usage: `fb:gemma-4e2b` → resolves to `foo-bar/gemma-4e2b:<default_quant>`

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
    model: unsloth/Qwen3-4B-GGUF:Q4_K_M
    backend: rocm            # optional override (defaults to backends.default)

    args:
      temp: 0.7
      top-p: 0.95
      ctx-size: 16384
      jinja: true

    max_log_lines: 2000        # per-model log buffer size
    capabilities: ["chat"]     # non-arg — shown in admin UI, not passed to subprocess
```

---

## Backend Definitions & Overrides

Backend definitions ship read-only in `<package>/data/backends.yaml`. They are
merged **under** your config.yaml at startup — you never edit the shipped file.

A backend answers two questions:

- **how to run** — `runner`, `args`, `env`, `engine`
- **how to provision the binary** — the `source:` entry (see below)

The binary slot path is *derived*, never written:

| `source.type` | Slot path |
|---|---|
| `remote` | `~/.local/arkestra/bin/<backend-id>/` |
| `local` | `source.path` itself — your build dir, verified, never fetched |

### Shipped backends

| Backend | Source | Notes |
|---|---|---|
| `vulkan-radv` | ggml-org/llama.cpp (vulkan) | Generic fallback — AMD, NVIDIA, Intel |
| `rocm-gfx1151` | lemonade-sdk/llamacpp-rocm | Strix Halo / gfx1151 |
| `rocm-gfx120x`, `rocm-gfx1150`, `rocm-gfx110x`, `rocm-gfx90a`, `rocm-gfx103x` | lemonade-sdk/llamacpp-rocm | One per gfx target |
| `cuda` | ggml-org/llama.cpp (cuda) | NVIDIA discrete GPUs |
| `cpu` | ggml-org/llama.cpp (static) | CPU-only, all cores |
| `onnx` | — (in-process) | ONNX runner; no binary, no slot |

**Naming convention:** shipped ROCm backends are named `rocm-gfx<target>`, one
per single-target artifact. The ID is otherwise an opaque label — a local or
multi-target build gets whatever name you give it (e.g. `rocm-custom`).

### Two things you can do from config.yaml

**1. Select the default backend** (a shipped or declared name):

```yaml
backends:
  default: rocm-gfx1151
```

**2. Declare a custom backend, or override any key of a stock backend.**
Overrides deep-merge over the shipped definition; your value wins per key.
Unlisted keys keep their shipped values. A full sub-dict declares a new backend.

```yaml
backends:
  default: rocm-custom
  # sparse override — only ngl changes, everything else ships as-is
  rocm-gfx1151:
    args:
      ngl: 512
  # custom backend backed by your own build
  rocm-custom:
    description: "Local ROCm build (Strix Halo)"
    source: {type: local, path: /home/me/local/rocm-bin/gfx1151}
    runner: process
    engine: llama-cpp
    args:
      ngl: 999
      ctx-size: ${default/ctx-size}
```

Declaring a full backend sub-dict in config.yaml logs a one-line warning at
startup reminding you it is treated as an override of the shipped layer.

### `source:` entry keys

| Key | Type | Description |
|---|---|---|
| `type` | str | `"remote"` (GitHub release, fetched by arkestra-bin) or `"local"` (your build dir — verified at init, never fetched). |
| `repo` | str | (remote) Owner/repo on GitHub, e.g. `"ggml-org/llama.cpp"`. |
| `asset` | str | (remote) Asset filename template; `{tag}` is substituted with the resolved release tag. |
| `path` | str | (local) Absolute path to the build directory containing `llama-server`. |

### arkestra-bin — binary installation

`arkestra-bin` (console command, on PATH) installs prebuilt llama.cpp binaries
into stable slots and manages pins:

```bash
arkestra-bin list                  # slots: backend, tag, pin, sha256
arkestra-bin fetch [backend ...]   # download + install latest (or pinned) tag
arkestra-bin pin <backend> <tag>   # pin a backend to a release tag
arkestra-bin unpin <backend>       # follow latest again
```

- Swaps are atomic: extract to `.new`, `mv` over the live slot. A running
  `llama-server` keeps its open inode; the next launch picks up the new build.
- State (installed tag, sha256, pin) lives in
  `~/.local/arkestra/bin-state/state.json`, keyed by backend ID.
- For `local` sources, `fetch` only verifies `path` exists and records the sha.

**At server init**, arkestra checks every slot referenced by a configured
backend and auto-fetches missing `remote` slots (default on). Upgrading an
installed slot to a newer release is explicit: `arkestra-bin fetch <backend>`. A model whose slot is still missing fails to start with the exact
`arkestra-bin fetch <backend>` command.

**Gfx guard:** if a `rocm-gfx<target>` backend's single target differs from the
gfx target `rocm-smi` reports, init logs a WARNING. Multi-target IDs pass if
the detected target is any member of the set. `local` and other non-matching
IDs are never checked.

### Backend Entry Keys

| Key | Type | Description |
|---|---|---|
| `description` | str | Human-readable description shown in `list-backends`. |
| `runner` | str | Runner type: ``"process"``, ``"podman"``, ``"docker"``, or ``"remote"``. Use ``"container"`` to defer to the top-level ``container-type:`` config value. |
| `base_url` | str | (Legacy `runner: remote` only) URL of the target arkestra worker. Prefer the `clusters:` top-level key instead. |
| `admin_key` | str | (Remote optional) API key forwarded as `x-admin-key` header to workers requiring authentication. If the target worker also proxies, forward its `admin_key` value here.
| `source` | dict | Binary provisioning entry — keys in the `source:` table above. Remote backends fetch into `~/.local/arkestra/bin/<backend-id>/`; local backends use `path` directly. |
| `args` | dict | Default CLI arguments merged into model args during startup. |
| `hf_flag` | str | (Optional) Override for the HuggingFace flag format — e.g., `"--hf"` instead of default `"-hf"`. Used when container images or binaries use a different flag convention. |
| `entrypoint` | str | (Container only) Override the container's ENTRYPOINT — e.g., `/llama.cpp/llama-server`. Prevents image defaults (like `tini`) from intercepting CLI args. |

### Custom Backends (Local Builds)

Declare them directly in the `backends:` section of config.yaml (your file) —
full example in [Two things you can do from config.yaml](#two-things-you-can-do-from-configyaml).
`source: {type: local, path: ...}` points at your build directory; arkestra
verifies it at init and records the binary sha, but never touches the files.
Select the backend with `backends.default:` or a per-model `backend:` override.

### Federated Clusters

The `clusters:` top-level key defines managed arkestra instances. Each entry is just `{name, url}` in the same URL form as the server's public address (`ARKESTRA_URL` / `default.url`). The **local** cluster is auto-created from that same canonical URL, so it always agrees with the server's bind/prefix. Model names prefixed `<cluster>/<model-id>` route to the matching cluster:

```yaml
clusters:
  gpu-server:
    url: "http://192.168.1.42:18000/base"
  cpu-worker:
    url: "http://192.168.1.43:8080"

models:
  gpu-server/gemma-4b:
    model: unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
    backend: rocm
  cpu-worker/whisper-large:
    model: distil-whisper/distil-small.en
    backend: cpu
```

**How it works:**
- The master **never downloads, spawns, or allocates ports** for remote-cluster models.
- All requests proxy through the cluster's `url`.
- Model names use `<cluster>/<model-id>` convention (e.g., `gpu-server/qwen3`) to identify routing.
- Local cluster uses port pool for subprocesses; remote clusters proxy all traffic.

> **Legacy compatibility**: Models prefixed `<worker>/<model-id>` with a backend entry
> having `runner: remote` + `base_url:` continue to work without a `clusters:` block.

---

## Backend Resolution & Defaults Chain

### Argument Merge Pipeline

Arguments are resolved per-key through a unified chain, then converted to CLI:

**Phase 1 — Per-key resolution (first match wins):**
1. Model-level field in config.yaml (e.g. ``models.<m>.ngl``)
2. Backend ``args:`` dict (e.g. ``backends.<b>.args.ngl``) — from config.yaml overrides or the shipped definition; GPU/offload defaults a backend sets for all its models
3. ``default:`` section in config.yaml
4. Runtime ``inference_kwargs`` passed to ``start()`` — transient, last-wins override

Only keys present in the engine's schema whitelist (shipped ``schemas.yaml`` →
``model-args.<engine>``) are emitted; infra/junk keys are dropped.

**Placeholders:** string values containing ``${...}`` are expanded against the
config ``macros:`` section plus the runtime-injected ``NPROC`` (CPU count).
A placeholder that cannot be resolved is treated as absent, so resolution falls
through to the next level in the chain. Example: ``ctx-size: ${default/ctx-size}``
reads the ``ctx-size`` key from the ``default:`` section.

**Phase 2 — CLI Conversion:**
``LlamaCppEngine.build_cli_args(merged, port)`` converts the merged dict to CLI
tokens. Infrastructure flags (``--port``, ``--model``) are injected by the engine.

### Backend Selection Resolution:

1. Model's ``backend:`` field (if specified)
2. ``backends.default:`` key in config.yaml (or a runtime-detected fallback override when the default backend's runtime is missing)
3. GPU-detection recommendation, else ``"cpu"``

### Runner Resolution:

1. Model's ``runner:`` field (per-model launch-mode override, e.g. force a container)
2. Backend entry's ``runner:`` field (``process``, ``podman``, ``docker``, ``onnx``, ``remote``)
3. ``default.container-type`` / ``runners.default``, else hardwired ``"process"``

The runner→class mapping is fixed in code; the config carries only the short
runner id, not a class name.

### Binary Resolution:

1. Backend's ``source:`` → slot path (``~/.local/arkestra/bin/<backend-id>/`` for remote, ``path`` for local)
2. `arkestra init`: one-time fetch of missing referenced slots (opt-out: `default/auto-fetch-binaries: false`)
3. Server start: local verify only; slot missing → hard error naming the exact ``arkestra bin fetch <backend>`` command

---

## Related Documentation

- [Usage Guide](./usage.md) — how to load and use config files
- [Architecture](./architecture.md) — runner routing based on backend config
- [Admin API](./admin.md) — runtime config management via HTTP endpoints
- [Lifecycle](./lifecycle.md) — state transitions for model runners
