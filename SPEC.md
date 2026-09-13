# ModelArkestra — Specification

A lightweight, config-driven manager for LLM inference engines. Models are
defined in YAML and started/stopped on demand. Each model gets an allocated
port and is launched through a pluggable runner (local process, OCI
container, remote worker, or in-process ONNX). An OpenAI-compatible HTTP
server and an admin dashboard sit on top of the same core.

## 1. Goals

- Run llama.cpp GGUF models (ROCm, Vulkan, CUDA, CPU) on demand; no
  permanently resident GPU processes.
- Drop-in OpenAI-compatible endpoints for any client.
- One console to administer local and remote clusters (federation).
- Auxiliary models (embeddings, Whisper ASR, Kokoro TTS) run in-process
  via ONNX Runtime — no ports, no subprocesses, VRAM preserved for LLMs.
- Deterministic naming (no random suffixes) for containers, images, logs.

## 2. Layers

```
┌────────────────────────────────────────────────────────────┐
│ CLI / HTTP surface                                         │
│  arkestra · arkestra-admin · arkestra-server · arkestra-onnx
│  ArkestraServer (FastAPI, OpenAI v1) + ArkestraAdmin (/admin/*)
├────────────────────────────────────────────────────────────┤
│ ModelArkestra  (arkestra.py)                               │
│  single entry point: config, port pool, cluster routing,   │
│  model context registry, pull/eject, global log ring       │
├────────────────────────────────────────────────────────────┤
│ BaseRunner  (base.py)                                      │
│  state machine, watchers, crash-restart, health polling,   │
│  SSE streaming, request retry                              │
├──────────────────────┬───────────────┬───────────┬─────────┤
│ ProcessRunner        │ ContainerRunner│ RemoteRunner│ OnnxRunner
│ (process.py)         │ podman/docker  │ (remote.py) │ (onnx_runner.py)
└──────────────────────┴───────────────┴───────────┴─────────┘
```

Each layer talks only to the layer below it. The HTTP layer never touches
subprocesses; the core never touches HTTP framing.

## 3. Configuration

Two XDG files, scaffolded by `arkestra init` (hardware auto-detected):

| File | Content |
|---|---|
| `~/.config/arkestra/config.yaml` | Models, port pool, defaults, clusters, env keys |
| `~/.config/arkestra/backends.yaml` | Backend definitions (rocm/vulkan-radv/cuda/cpu/custom) + download sources |

backends.yaml is the base; config.yaml overlays it (deep merge). Resolution
precedence, all keys:

- **Backend**: `model.backend` → `backends.default` → hardwired fallback.
- **Runner**: `model.runner` → `backend.runner` → `default/container-type` →
  `"process"`. The sentinel `container` resolves to `default/container-type`.
- **Model ref**: `[alias:][repo:]model[:quant]`, with `lcl:` / absolute paths
  for local files; aliases expand from `model-repos:`; missing repo/quant
  filled from `default:` (`model-repo`, `model-quant`).
- **Args**: model `args:` dict + runtime inference kwargs (last wins),
  merged flat, then converted to CLI tokens by the engine.
- **Env keys** (`default-env:` + process env): explicit arg → `os.environ` →
  YAML default. Computed at startup, never persisted.

### Model entry

```yaml
models:
  qwen3-4b:
    model: unsloth/Qwen3-4B-GGUF:Q4_K_M   # or lcl:/path.gguf, alias:short
    backend: rocm                          # optional
    args: { temp: 0.7, ctx-size: 16384, jinja: true }
    capabilities: [chat]                   # informational, not passed to engine
  bge-embeddings:                          # ONNX auxiliary model
    model_path: /path/model.onnx
    type: embedding                        # | whisper | tts
    tokenizer: Xenova/bge-small-en-v1.5
```

Tags (`asr`, `tts`, `embed`) or `runner: onnx` route a model to the ONNX
runner instead of a subprocess.

## 4. Port Allocation

- Pool: `default/model-start-port` (18000) through
  `start + model-ports - 1` (32).
- Sequential counter; exhaustion raises `RuntimeError`.
- Explicit `port=` bypasses the pool — caller owns validation.
- Stopped models retain their port; restart is in-place, same port.
- `start()` on an already-RUNNING model is a health-checked no-op.
- ONNX and remote models allocate no local port.

## 5. Lifecycle

States: `UNCACHED → DOWNLOADING → STOPPED → LOADING → RUNNING`,
with `STOPPING` (stop requested, no restarts) and `ERROR` (terminal).

```
 start ─► LOADING ─► /health OK + warmup ─► RUNNING
             │                          │
             │ crash (watcher)          │ health poll → loading/error
             ├─ restarts < limit → LOADING (same port)
             └─ restarts ≥ limit → ERROR

 pull ─► DOWNLOADING ─► STOPPED | ERROR     (cancel → UNCACHED)
```

- **Watchers**: per-model background task awaits process exit
  (process runner) or polls container status (containers); unexpected exit
  triggers restart after `restart_delay` (5s), max `restart_limit` (4).
- **Health watcher**: polls `/health` on every RUNNING model; maps
  `loading`/`error` statuses to states.
- **Start**: resolve backend/runner → allocate port → build CLI args →
  spawn → poll `/health` until ok/loaded (timeout 120s) → warmup delay
  (10s) → RUNNING. Failure stops the model and raises `ServerReadyTimeout`.
- **Stop**: STOPPING first (blocks restarts) → graceful signal → escalate.
  Process: SIGHUP to process group, 20s, then SIGKILL. Container:
  `stop --time 20`. Stopped models stay tracked.
- **Shutdown**: stop all, cancel watchers, clear registry, reset port
  counter. Only operation that discards contexts.
- **Pull**: `snapshot_download` in a thread; progress streams to the model
  log ring; cancelable; partial-cache cleanup on cancel/failure.
- **Eject**: stop + delete checkpoint cache; refuses if another running
  model shares the same cache directory.

## 6. Runners

| Runner | Launch | Crash detection | Notes |
|---|---|---|---|
| Process | `asyncio` subprocess of `llama-server` | `process.wait()` | env = process + global + device-profile + backend `env_container` |
| Podman/Docker | detached OCI container, `llm-{model}-{port}` | status poll 2s | image per backend; `entrypoint:` override supported |
| Remote | none — HTTP proxy to another Arkestra | none | no local port; tolerates raw llama-server workers (no admin routes) |
| ONNX | in-process `InferenceSession` | n/a | chat/embed/asr/tts handlers, no HTTP of its own |

One runner instance per model (`type:model` key in the runner registry);
shared per-type instances for stateless paths (e.g. remote).

### Argument pipeline

1. **Merge** — `build_model_args()`: backend `args:` + model `args:` +
   runtime inference kwargs (last wins).
2. **Convert** — `LlamaCppEngine.build_cli_args(merged, port)`:
   - `model` + `repo=hf` → `-hf <ref> --alias <ref>`; else `--model <ref>`
   - kebab-case keys → flags; single-dash set for short llama flags
   - `bool true` → presence-only; `false` → omitted
   - `--port` appended; `max-tokens` aliased to `n-predict`
   - request-time sampling params filtered through a whitelist; unknown
     keys are dropped, never forwarded.

## 7. HTTP Server (ArkestraServer)

FastAPI + uvicorn. Public URL (`--url` / `ARKESTRA_URL` / `default.url`)
carries host, port, and path prefix; `--bind` is separate (default
`127.0.0.1`, `0.0.0.0` for LAN). Optional TLS, CORS, `extra_headers`.

| Path | Purpose |
|---|---|
| `POST /v1/chat/completions` | chat; `stream: true` → SSE; auto-starts the model if stopped |
| `GET /v1/models` | list with runtime status |
| `POST /v1/embeddings` | routed to ONNX (or remote) model |
| `POST /v1/audio/transcriptions` / `/v1/audio/speech` | Whisper / Kokoro |
| `GET /health`, `/v1/health` | server + model health |
| `/`, `/index.html` | admin dashboard (self-contained HTML) |
| `/admin/*` | admin API — gated by `admin_key` |

Model resolution for inference: exact config name, `openai_aliases`
mapping, or tag lookup. Sampling params from the request are merged as
transient inference kwargs. Requests on down models retry on 503.

**Admin API** (all JSON): model list/start/stop/stop-all/restart/eject,
config CRUD (create/get/update/delete), model logs + global log (ring
buffer), checkpoint pull/cancel, image list/build/remove, cluster
list/add/delete, shutdown.

## 8. Federation

`clusters:` maps name → `{url, admin-key}`. Model names of the form
`<cluster>/<model-id>` route all traffic (lifecycle and inference) to that
worker; the local cluster is synthesized from the server's public URL. The
master never downloads, spawns, or allocates ports for remote models.
Legacy `runner: remote` backends with `base_url:` still resolve. Admin key
forwarded via `x-admin-key`.

## 9. CLI

| Command | Function |
|---|---|
| `arkestra init / detect / models / chat` | scaffold config, hardware report, list models, interactive chat (auto-start, `/`-commands) |
| `arkestra list-backends / add-backend / remove-backend / download-backend / download-all` | backend + binary management (GitHub releases, checksum-verified, cached) |
| `arkestra-admin models/start/stop/stop-all/restart/pull/cancel-pull/eject/logs/config …/clusters …/images …/shutdown` | full remote administration of the server |
| `arkestra-server` / `arkestra-onnx` | launch HTTP servers |

Connection resolution for all CLIs: `--url` → `ARKESTRA_URL` →
`config default.url` → `http://127.0.0.1:8080`.
Config path: `--config` → `ARKESTRA_CONFIG` → `ARKESTRA_DIR` → XDG default.

## 10. Binary & Model Provisioning

- `backends.yaml sources:` declare fetch strategy (`github-release`,
  `oci-image`, `local-file`) with asset globs and sha256 verification;
  `download-backend` writes `binary_dir`/`binary` back into the backend.
- GGUF checkpoints use the HuggingFace cache layout
  (`models--<org>--<repo>`) under `hf-hub-cache`; presence of a `.gguf`
  snapshot/blob decides STOPPED vs UNCACHED at init.
- GPU detection (`gpu_detect`) gates backend validity at startup (hard
  error if the configured backend's runtime is missing) and supplies
  device-profile env vars (e.g. `HSA_VISIBLE_DEVICES`) to spawned processes.

## 11. Logging

- Per-model ring buffer (`UnicodeRingBuffer`, default 500 lines) fed from
  process stdout/stderr (or container log stream, or pull progress).
- Global server ring buffer (`app-log-lines`, 2000) with uvicorn-style
  colored console output; both served via admin log endpoints.

## 12. Invariants

- Ports are assigned once per model and reused across restarts.
- Names are deterministic: container `llm-{sanitized-model}-{port}`,
  image tags from backend definitions, no timestamps or random IDs.
- Inference kwargs are transient; only `args:` in config persist.
- A model in STOPPED/ERROR/UNCACHED may start; LOADING/RUNNING/STOPPING/
  DOWNLOADING may stop (see `can_start` / `can_restart` / `can_stop`).
- The core is embeddable: `ModelArkestra` is a plain async context manager
  usable without the HTTP server (Python API: `start`, `ainvoke`,
  `astream`, `stop`, `embed`, `transcribe`, `synthesize`, `pull_model`,
  `eject`).
