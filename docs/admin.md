# Admin API (`admin.py`)

Model Arkestra ships an administrative panel that integrates into the same FastAPI app via `ArkestraServer`. It provides endpoints for monitoring and managing models, with optional API-key authentication on all admin paths.

## Initialization

Admin routes are installed automatically when calling `server.get_app()` — pass the ``admin_key`` argument to enable:

```python
server = ArkestraServer(
    "config.yaml",
    port=8080,
    admin_key="your-secret-key",  # gates all /admin/* paths
)
app = server.get_app()
```

The key resolves with priority: **constructor argument** > ``config.env.ADMIN_KEY`` from the YAML config file > disabled (no auth).

When `admin_key` is provided, every request to `/admin/*` must include the header:

```http
X-Admin-Key: your-secret-key
```

Missing or incorrect keys return `401 Unauthorized`. Public paths (`/`, `/index.html`) are unaffected.

## Endpoints

| Method | Path | Auth Required | Description |
|---|---|---|---|
| `GET` | `/` | No | Serves the admin dashboard HTML page |
| `GET` | `/index.html` | No | Same as `/` (explicit route) |
| `GET` | `/admin/models` | Yes | Full context listing for all configured models |
| `GET` | `/admin/config` | Yes | List model names in config |
| `POST` | `/admin/config` | Yes | Create a new model entry |
| `GET` | `/admin/config/{model}` | Yes | Retrieve a single model's configuration |
| `PUT` | `/admin/config/{model}` | Yes | Update an existing model's configuration (no restart) |
| `POST` | `/admin/start/{model}` | Yes | Start or restart a model (with optional transient overrides) |
| `POST` | `/admin/stop/{model}` | Yes | Stop a running model |
| `POST` | `/admin/eject/{model}` | Yes | Remove model from cache, clear contexts (no config change) |
| `GET` | `/admin/log/{model}?since=N&lines=M` | Yes | Delta or snapshot log query |
| `GET` | `/admin/logs?since=N&lines=M` | Yes | Server-level log entries (proxy traffic, lifecycle events) |
| `GET` | `/admin/images` | Yes | List configured container images with runner type and availability |
| `POST` | `/admin/images/build` | Yes | Build a single backend's image (body: `{"backend": "rocm"}`) |
| `DELETE` | `/admin/images/{image_tag}` | Yes | Remove an image from the local store |
| `POST` | `/admin/stop-all` | Yes | Stop all running models — models restart implicitly on next inference request |
| `POST` | `/admin/shutdown` | Yes | Full server teardown — stops uvicorn and all models

### GET /admin/models

Returns a list of all configured models with their full runtime context. Models that have been started get their real state from the runner context; unstarted models get a constructed entry:

```json
{
  "models": [
    {
      "id": "qwen3.5-4b",
      "status": "running",
      "port": 18000,
      "runner_type": "process",
      "backend_id": "rocm",
      "args": ["temp", 0.7, "top-k", 20, "ctx-size", 131072],
      "repo": "hugging-face",
      "model": "unsloth/Qwen3-4B-GGUF:Q4_K_M",
      "capabilities": ["chat"]
    },
    {
      "id": "gemma-4-e2b",
      "status": "uncached",
      "port": null,
      "runner_type": null,
      "backend_id": "vulkan-radv",
      "args": ["temp", 0.7, "top-p", 0.95, "ctx-size", 131072],
      "repo": "hugging-face",
      "model": "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M",
      "capabilities": []
    }
  ],
  "backends": {"vulkan-radv": {...}, "rocm": {...}},
  "runner_types": ["process", "podman", "docker"]
}
```

| Field | Source |
|---|---|
| `id` | Model name from config |
| `status` | One of: `running`, `loading`, `error`, `stopping`, `stopped`, or `uncached` |
| `port` | Allocated port (null if not running) |
| `runner_type` | Runner type string (null if not running) |
| `backend_id` | Resolved backend (from context or config fallback) |
| `args` | Model args from config |
| `repo` | Model repo identifier |
| `model` | Model path within repo |
| `capabilities` | Capability tags (default `["chat"]` if none specified) |
| `available_capabilities` | Resolved capability pool for the admin UI chips — follows the chain: per-model `capabilities` → backend-declared `backends.<id>.capabilities` → hardcoded fallback `["chat"]` |

**Status values:**
- `running`, `loading`, `error`, `stopping` — real states from active runner contexts
- `stopped` — model was previously started but is now stopped; its weights **are** in the HF cache
- `uncached` — model exists in config but is not currently downloaded

Top-level metadata (`backends`, `runner_types`) is static for the lifetime of the server.

### POST /admin/config

Create a new model entry in config. Returns `201 Created` on success.

```bash
curl -X POST 'http://localhost:8080/admin/config' \
     -H 'Content-Type: application/json' \
     -H 'X-Admin-Key: your-secret-key' \
     -d '{
       "name": "my-new-model",
       "repo": "hugging-face",
       "model": "unsloth/my-model-GGUF:Q4_K_M",
       "backend": "vulkan-radv",
       "args": {"temp": 0.7, "ctx-size": 131072}
     }'
```

| Field | Required | Description |
|---|---|---|
| `name` | No | Model name in config. Defaults to the last segment of `model` (before `:` and `/`). |
| `repo` | **Yes** | HuggingFace repo identifier. |
| `model` | **Yes** | HF model path within repo. |
| `args` | No | Command-line arguments for the model. |
| `backend` | No | Backend ID from config (e.g., `vulkan-radv`, `rocm`). |
| `capabilities` | No | Capability tags (default `["chat"]` if empty). |
| `tags` | No | Free-form tags. |

Returns `400 Bad Request` if `repo` or `model` is missing. Returns `409 Conflict` if the model name already exists.

### GET /admin/config/{model}

Retrieve a single model's configuration:

```bash
curl 'http://localhost:8080/admin/config/qwen3.5-4b' \
     -H 'X-Admin-Key: your-secret-key'
```

Returns:
```json
{
  "ok": true,
  "model": "qwen3.5-4b",
  "status": "running",
  "config": {
    "repo": "hugging-face",
    "model": "unsloth/Qwen3-4B-GGUF:Q4_K_M",
    "args": {"temp": 0.7, "top-k": 20, "ctx-size": 131072},
    "backend": "rocm"
  },
  "available_capabilities": ["chat"]
}
```

Returns `404` if the model is not found.

### PUT /admin/config/{model}

Update an existing model's configuration and write it to disk. **Does not restart** the model — use `POST /admin/start/{model}` separately.

```bash
curl -X PUT 'http://localhost:8080/admin/config/qwen3.5-4b' \
     -H 'Content-Type: application/json' \
     -H 'X-Admin-Key: your-secret-key' \
     -d '{"args": {"temp": 1.0, "ctx-size": 32768}, "capabilities": ["chat"]}'
```

Valid fields: `args`, `repo`, `model`, `backend`, `capabilities`, `runner`, `tags`.

Returns `404` if the model does not exist. Returns `500` on write failure (config is rolled back).

### POST /admin/start/{model}

Start or restart a model. Returns the port assigned.

For an already-running model with transient overrides, this will stop and restart the model:

```bash
# Start a stopped model
curl -X POST 'http://localhost:8080/admin/start/qwen3.5-4b' \
     -H 'X-Admin-Key: your-secret-key'

# Restart with transient overrides (no config change)
curl -X POST 'http://localhost:8080/admin/start/qwen3.5-4b' \
     -H 'Content-Type: application/json' \
     -H 'X-Admin-Key: your-secret-key' \
     -d '{"backend": "docker", "repo": "hugging-face", "model": "unsloth/Qwen3.5-4B-GGUF:Q5_K_M"}'
```

Transient overrides are **not** persisted to disk. They apply only to this invocation:

Infra keys (resolved before inference filtering):
- `args` — command-line arguments override
- `repo` — model repo override
- `model` — model path within repo override
- `backend` — backend ID override (runner resolves from config chain)
- `runner` — explicit runner type (`process`, `podman`, `docker`, or `remote` (legacy))
- `max_log_lines` — per-invocation log buffer size

Any other keys are treated as inference parameters for llama.cpp. Only those present in the engine's ``LLAMA_INFER_ARGS`` whitelist (e.g. `temp`, `top-p`, `reasoning-budget`) reach CLI construction; unknown keys are silently dropped to prevent subprocess crashes.

Returns `503` if the model fails to start within `ready_timeout`.

### POST /admin/stop/{model}

Stops the named model. Returns `202 Accepted` if the model is already stopped/stopping (no-op), or `200 OK` after a successful stop.

```json
// 200 — model was running, now stopped
{"ok": true, "model": "qwen3.5-4b", "previous_state": "running"}

// 202 — already stopped (no-op)
{"ok": true, "model": "qwen3.5-4b", "previous_state": "stopped"}
```

Returns `404` if the model is not found in any runner context.

### POST /admin/stop-all

Stops all running models at once. Models remain configured and will **restart implicitly** on their next inference request (same lazy-start behavior as a cold server).

```bash
curl -X POST 'http://localhost:8080/admin/stop-all' \
     -H 'X-Admin-Key: your-secret-key'
```

**When models are running:**
```json
{"ok": true, "message": "Stopped 2 model(s) — will restart implicitly on next request", "stopped": ["qwen3.5-4b", "gemma-4-e2b"]}
```

**When no models are running:**
```json
{"ok": true, "message": "No models running — nothing to stop", "stopped": []}
```

Always returns `200 OK`. The HTTP server stays alive; only model runners are stopped.

### POST /admin/shutdown

Full server teardown — stops the uvicorn HTTP listener and shuts down all model runners. This is an irreversible operation: after shutdown, model entries are cleared and cannot be restarted without restarting the entire server process.

```bash
curl -X POST 'http://localhost:8080/admin/shutdown' \
     -H 'X-Admin-Key: your-secret-key'
```

Returns immediately with `200 OK`:
```json
{"ok": true, "message": "Server shutting down"}
```

Then shuts down in background:
1. All model runners stopped (same sequencing as `stop_all`) — graceful signals → 20s timeout → SIGKILL
2. Watcher tasks cancelled and awaited
3. `_models` and `_watchers` dictionaries cleared
4. Container runners force-removed (`podman rm -f` / `docker rm -f`)
5. Uvicorn HTTP server stops — process exits

This endpoint always returns `200 OK`. The response is sent before shutdown begins.

### POST /admin/eject/{model}

Removes a model from cache without modifying config. Stops the model first (if running), deletes its cached files from the HF cache directory, and clears all runner context entries.

**Safety check:** before deleting, if another *running* model shares the same underlying cache directory (same resolved model path under `HF_HUB_CACHE`), the eject is rejected:
```json
{"detail": "Model 'qwen3.5-4b' is in use by other running runners: gemma-v2, llama-rdma"}
```

Returns `200 OK` with a detail report on success:
```json
{
  "ok": true,
  "model": "qwen3.5-4b",
  "cache_deleted": true,
  "cache_path": "/home/user/.cache/huggingface/hub/models--unsloth--Qwen3-4B-GGUF",
  "contexts_cleared": 1
}
```
If the model has no model configured, or the cache directory doesn't exist, `cache_deleted` is `false`. Returns `404` if the model doesn't exist in config. Returns `409 Conflict` when a shared-cache conflict prevents eject.

### GET /admin/log/{model}?since=N&lines=M

Return log lines for a running model. This is an **HTTP delta endpoint** — no streaming, no SSE. Clients poll on a schedule (typically 1–2s) to receive only new log lines since their last request.

**Snapshot mode** (no `since` parameter):
```bash
curl 'http://localhost:8080/admin/log/qwen3.5-4b' \
     -H 'X-Admin-Key: your-secret-key'
```
Returns the full current log buffer:
```json
{"lines": ["[INFO] Loading model...", "[INFO] Ready"]}
```

**Delta mode** (`since` parameter):
```bash
curl 'http://localhost:8080/admin/log/qwen3.5-4b?since=847&lines=50' \
     -H 'X-Admin-Key: your-secret-key'
```
Returns only log lines with sequence number greater than `847`:
```json
{
  "since": 912,
  "missed_lines": 0,
  "lines": [
    {"seq": 848, "text": "[INFO] KV cache init..."},
    {"seq": 849, "text": "[INFO] Loading model weights..."}
  ]
}
```

**Response headers** (present on every response):
| Header | Description |
|---|---|
| `X-Current-Max` | Latest log line sequence number on the server |
| `X-Missed-Lines` | Number of lines pruned from the buffer before or during the requested range. Zero if all requested lines are still in the buffer. |

**Missed-lines scenario:** when a client is disconnected too long, older lines may fall off the ring buffer (default capacity: 500 lines). If `since=N` but line N has already been evicted, the response includes `X-Missed-Lines: K` indicating how many lines were skipped. The returned JSON still includes any remaining lines newer than the gap.

**Delta protocol usage pattern:**
1. Client starts with `since=0` to get all available lines
2. Each response header `X-Current-Max` becomes the next request's `since`
3. On reconnect, client sends its last known `since` value
4. If `X-Missed-Lines > 0`, the client knows some log lines were lost

**Implementation notes:** Log lines are tagged with a per-model monotonic sequence number as they are appended to the ring buffer by subprocess watchers (`ProcessModelRunner`) or container log streaming (`podman logs -f` / `docker logs -f`). The buffer uses a fixed-size deque (default 500 lines, configurable via `max_log_lines` in config or startup override). Only lines within the current window are available — older entries are automatically evicted.

### GET /admin/logs?since=N&lines=M

Return **server-level** log entries for the entire ModelArkestra instance. This is a separate ring buffer from per-model logs (`/admin/log/{model}`) and captures proxy traffic, model lifecycle events, and server startup/shutdown.

**Request format:**
```bash
curl 'http://localhost:8080/admin/logs?since=0&lines=200' \
     -H 'X-Admin-Key: your-secret-key'
```

**Response (same shape as per-model log):**
```json
{
  "seq": 47,
  "missed_lines": 0,
  "lines": [
    {"seq": 1, "text": "[action=start server port=8080]"},
    {"seq": 2, "text": "[action=start model=qwen3 port=18001]"},
    {"seq": 3, "text": "[action=req model=qwen3 method=POST path=/v1/chat/completions status=200 latency_ms=420 tokens=240]"},
    {"seq": 4, "text": "[action=stream_start model=gemma messages=2]"},
    {"seq": 5, "text": "[action=stream_end model=gemma duration_ms=1850 tokens=342] status=ok"}
  ]
}
```

**Response headers:** Same as per-model log — `X-Current-Max` (latest seq), `X-Missed-Lines` (evicted entries before the `since` point).

**Log entry types and their metadata fields:**
| action | Fields logged |
|---|---|
| `start server` | port |
| `start model` | model, port |
| `stop model` | model |
| `shutdown` | — (server-level teardown) |
| `req` | model, method, path, status, latency_ms, tokens |
| `stream_start` | model, messages |
| `stream_end` | model, duration_ms, tokens, status (`ok`, `error`, or `no_tokens`) |

The buffer is configurable via `app-log-lines` in the YAML config (default: 2000 entries). In the admin dashboard, select **"Server logs"** from the log model dropdown to view these entries alongside per-model output.

### GET /admin/images

List all container images configured in `backends.yaml`, along with their runner type and availability status.

Returns a JSON array:
```json
[
  {
    "backend_id": "rocm",
    "runner": "podman",
    "runtime_detected": true,
    "image": "ark-llama:rocm",
    "containerfile": "Containerfile.rocm",
    "available": false
  }
]
```

| Field | Description |
|---|---|
| `backend_id` | Backend identifier from the `backends:` section of backends.yaml |
| `runner` | Resolved runner type (`podman`, `docker`, or `process`) |
| `runtime_detected` | Whether the container runtime for this runner is available on PATH |
| `image` | Full image tag configured for this backend |
| `containerfile` | Name of the Containerfile (resolved to `tests/files/<name>`) |
| `available` | Whether the image exists in the local container store (only checked when `runtime_detected` is true) |

### POST /admin/images/build

Build a single backend's container image. The `backend` key must be provided — no "build all" mode.

Request body:
```json
{"backend": "rocm"}
```

The endpoint resolves the configured runner type for the backend, detects which runtime (podman or docker) is available on PATH, then runs the appropriate build command. If the configured runtime isn't present, returns gracefully:
```json
{"skipped": true, "reason": "runner=podman but no 'podman' binary found on PATH", "image": "ark-llama:rocm"}
```

On attempt (whether successful or not):
```json
{
  "backend": "rocm",
  "image": "ark-llama:rocm",
  "success": false,
  "runtime": "podman",
  "output": "STEP 1/10: FROM ...\nError: ...",
  "error": "Failed to resolve the transaction:\nNo match for argument: hip-runtime-rocm"
}
```

Build runs synchronously with a 600s timeout. The full stdout/stderr from the container runtime is returned in `output`. Returns `400` if `backend` is missing from the body, or `404` if no Containerfile is found for the backend.

### DELETE /admin/images/{image_tag}

Remove an image tag from the local store. The tag must match one configured in a backend's `image:` entry in backends.yaml.

```json
DELETE /admin/images/ark-llama:rocm
```

Returns:
```json
{"removed": true, "image": "ark-llama:rocm", "error": null}
```

Resolves the backend this image belongs to, determines its runner type, and runs `podman rmi -f` or `docker rmi -f` accordingly. Returns `404` if the tag isn't configured in any backend. Returns `{"skipped": true}` with a reason when the runtime isn't available.

## Admin Dashboard (`static/index.html`)

Model Arkestra ships a single-file, zero-dependency admin dashboard — a vanilla JavaScript application served at `/` that provides a web UI for the Admin API. It requires no build step, no frameworks, and runs entirely in the browser.

### Deployment

Place `static/index.html` anywhere the server serves static files. The dashboard is served automatically when the admin routes are mounted:

```python
server = ArkestraServer(
    "config.yaml",
    port=8080,
    admin_key="my-secret",   # gates /admin/* endpoints
)
```

With `admin_key` set, visiting the server root (e.g. `http://localhost:8080/`) serves the dashboard HTML page.

### Configuration

At the very top of the script block is a single configurable constant:

```js
const ADMIN_KEY = 'whatever';   // must match admin_key in ArkestraServer
```

This is the only thing you need to change before deploying. The dashboard automatically attaches it as the `X-Admin-Key` header on every API call.

### Layout

The dashboard uses a **two-column resizable layout** starting at 40% / 60%:

| Column | Content |
|---|---|
| **Left** (default 40%) | **Models** accordion — searchable model list with status indicators. **Edit** accordion — form for viewing/modifying config (opens on model selection). |
| **Right** (default 60%) | Unified **accordion widget** containing two panes: **Logs** (terminal-style log viewer) and **Chat** (conversational inference UI). Open panes split available height evenly. |

- Drag the divider between columns to resize (20–80% range); preference persists in `localStorage`.
- Click any accordion header to toggle collapse/expand. The top header bar toggles all sections simultaneously.
- Model list search filters client-side from cached data — instant, no server round-trip.

### Features

**Models list** — Displays every configured model with:
- Status dot: 🟢 running, 🟡 loading/uncached/stopped, 🔴 error
- Backend ID and runner type as metadata
- Click a model to select it (highlights in accent color)
- Text filter input at top for quick lookup

**Edit form** — Opens automatically on model selection. Contains fields for:
- Checkpoint path, backend, runner type, args string
- Log buffer size (max log lines ring buffer)
- Capability chips (`chat`, `tts`) — click to toggle; default is `[chat]` for all models unless explicitly overridden with non-chat tags (opt-out model)

State management:

| Button | Behavior |
|---|---|
| **Reset** | Re-fetches config from server, repopulates form, clears dirty state |
| **Save ·** | Writes current values to disk via `PUT /admin/config/{model}`. Starts disabled; enables when form differs from cached snapshot. The trailing dot animates on hover as visual feedback. |
| **Cancel** | Closes the Edit accordion. Form values stay in memory (not reverted) so you can reopen later with edits intact. |
| **Start / Restart** | Sends `POST /admin/start/{model}` with current transient draft values as overrides — no save-to-disk step first. |

**Logs pane** — Terminal-style log viewer for a selected model or server:
- **Model logs**: Select a specific model from the dropdown to view its process stdout/stderr via `GET /admin/log/{model}`
- **Server logs**: Select "Server logs" from the dropdown to view proxy traffic and lifecycle events via `GET /admin/logs`
- Both share the same delta-polling pattern (1–2s interval) with `?since=N` cursor
- Smart auto-scroll: scrolls to bottom during active streaming *unless* you've scrolled up to read older logs
- Clear button empties the pane content
- Smart auto-scroll: scrolls to bottom during active streaming *unless* you've scrolled up to read older logs
- Clear button empties the pane content
- Missed-line notifications shown if a reconnect gap is detected

**Chat pane** — Conversational inference UI that talks directly to the model's port:
- Parameter panel: Temperature, Top P, Max Tokens — persisted per-model in `localStorage`
- Full conversation history maintained in-memory across turns (OpenAI-compatible message format)
- Token-by-token SSE streaming with animated cursor during generation
- Markdown rendering via `marked.js` from CDN — code blocks, bold, lists, inline code all rendered
- Send button disabled while a response is streaming; concurrent messages prevented

### Technical Details

- **File size**: ~60KB served, ~1400 lines of HTML/CSS/JS (as of v0.3)
- **Zero dependencies**: No build step, no frameworks. Only `marked.js` loaded from CDN for markdown rendering.
- **All data via fetch/SSE**: Model list refreshes automatically on page load; subsequent interactions use the same Admin API documented above.
- **Layout persistence**: Column width ratio saved to `localStorage('arkestra-col-width')`
- **Chat params**: Per-model chat parameters saved to `localStorage('arkestra-chat-params')`

## Related Documentation

- [Server Documentation](./server.md) — how ArkestraServer works, CLI options
- [Configuration Format](./config.md) — YAML config that admin endpoints modify
- [Usage Guide](./usage.md) — Python API equivalent of admin operations
- [Lifecycle](./lifecycle.md) — state transitions reflected in `/admin/models`

## `arkestra-admin` CLI Tool

A command-line interface for all admin endpoints, installed alongside `arkestra-cli` and `arkestra-server`. Reads `ADMIN_KEY` from config.yaml by default.

```bash
arkestra-admin --server http://localhost:8080 --api-key SECRET <command>
```

### Authentication Priority
1. `--api-key KEY` flag (highest)
2. `$ADMIN_KEY` environment variable
3. `ADMIN_KEY` from `config.yaml`'s `env:` section

### Commands

| Command | Description |
|---|---|
| `arkestra-admin models` | List all configured models with status, port, backend |
| `arkestra-admin start <name>` | Start a model (supports `--port`, `--backend`, `--runner`, `key=value` params) |
| `arkestra-admin stop <name>` | Stop a running model |
| `arkestra-admin stop-all` | Stop all running models |
| `arkestra-admin config list` | List model names in config |
| `arkestra-admin config get <name>` | Show one model's full config + runtime status |
| `arkestra-admin config set <name> key=value` | Update a model field (e.g., `backend=rocm`) |
| `arkestra-admin config create --model PATH` | Add a new model to config |
| `arkestra-admin config rm <name>` | Remove a model from config |
| `arkestra-admin logs <name\|all> [--lines 100]` | Tail model or global server logs |
| `arkestra-admin eject <name>` | Stop model and delete its cached files |
| `arkestra-admin images list` | Show OCI image availability per backend |
| `arkestra-admin images build <backend> [--tag TAG]` | Build an OCI container image |
| `arkestra-admin images rm <image_tag>` | Remove a container image |
| `arkestra-admin shutdown` | Gracefully stop the server |

### Examples

```bash
# List models (auto-reads ADMIN_KEY from config)
arkestra-admin --server http://127.0.0.1:8080 models

# Start with overrides
arkestra-admin start qwen3-4b --backend vulkan-radv temp=0.7 top-k=20

# Tail logs with custom count
arkestra-admin logs gemma-4-e2b --lines 50

# Build and check OCI images
arkestra-admin images build rocm-container --tag rocm-7.14
arkestra-admin images list

# Full server shutdown on remote host
arkestra-admin shutdown -x http://remote-host:8080 --api-key mysecret
```

### JSON Output

Add `--json` to any command for machine-readable output:

```bash
arkestra-admin --server http://localhost:8080 config get qwen3-4b --json
```
