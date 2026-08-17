# Admin API (`admin.py`)

Model Arkestra ships an administrative panel that integrates into the same FastAPI app via `ArkestraServer`. It provides endpoints for monitoring and managing models, with optional API-key authentication on all admin paths.

## Initialization

Admin routes are installed automatically when calling `server.get_app()` — pass the `admin_key` argument to enable:

```python
server = ArkestraServer(
    "config.yaml",
    port=8080,
    admin_key="your-secret-key",  # gates all /admin/* paths
)
app = server.get_app()
```

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
| `GET` | `/admin/log/{model}?lines=N&follow=true` | Yes | Log snapshot or SSE stream |
| `POST` | `/admin/restart` | Yes | Stop all running models — models restart implicitly on next inference request |
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
      "checkpoint": "unsloth/Qwen3-4B-GGUF:Q4_K_M",
      "capabilities": ["chat"]
    },
    {
      "id": "gemma-4-e2b",
      "status": "uncached",
      "port": null,
      "runner_type": null,
      "backend_id": "vulkan-radv",
      "args": ["temp", 0.7, "top-p", 0.95, "ctx-size", 131072],
      "checkpoint": "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M",
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
| `checkpoint` | Model checkpoint reference |
| `capabilities` | Capability tags (default `["chat"]` if none specified) |

**Status values:**
- `running`, `loading`, `error`, `stopping` — real states from active runner contexts
- `stopped` — model was previously started but is now stopped; its checkpoint **is** in the HF cache
- `uncached` — model exists in config but has never been downloaded (checkpoint not found in `HF_HUB_CACHE`)

Top-level metadata (`backends`, `runner_types`) is static for the lifetime of the server.

### POST /admin/config

Create a new model entry in config. Returns `201 Created` on success.

```bash
curl -X POST 'http://localhost:8080/admin/config' \
     -H 'Content-Type: application/json' \
     -H 'X-Admin-Key: your-secret-key' \
     -d '{
       "name": "my-new-model",
       "checkpoint": "unsloth/my-model-GGUF:Q4_K_M",
       "backend": "vulkan-radv",
       "args": {"temp": 0.7, "ctx-size": 131072}
     }'
```

| Field | Required | Description |
|---|---|---|
| `name` | No | Model name in config. Defaults to the last segment of checkpoint (before `:` and `/`). |
| `checkpoint` | **Yes** | HuggingFace checkpoint reference. |
| `args` | No | Command-line arguments for the model. |
| `backend` | No | Backend ID from config (e.g., `vulkan-radv`, `rocm`). |
| `capabilities` | No | Capability tags (default `["chat"]` if empty). |
| `tags` | No | Free-form tags. |

Returns `400 Bad Request` if `checkpoint` is missing. Returns `409 Conflict` if the model name already exists.

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
  "port": 18000,
  "config": {
    "checkpoint": "unsloth/Qwen3-4B-GGUF:Q4_K_M",
    "args": {"temp": 0.7, "top-k": 20, "ctx-size": 131072},
    "backend": "rocm"
  }
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

Valid fields: `args`, `checkpoint`, `backend`, `capabilities`, `runner`, `tags`.

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
     -d '{"backend": "docker", "checkpoint": "unsloth/Qwen3.5-4B-GGUF:Q5_K_M"}'
```

Transient overrides are **not** persisted to disk. They apply only to this invocation:

Infra keys (resolved before inference filtering):
- `args` — command-line arguments override
- `checkpoint` — model checkpoint reference override
- `backend` — backend ID override (runner resolves from config chain)
- `runner` — explicit runner type (`process`, `podman`, `docker`)
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

### POST /admin/restart

Stops all running models at once. Models remain configured and will **restart implicitly** on their next inference request (same lazy-start behavior as a cold server).

```bash
curl -X POST 'http://localhost:8080/admin/restart' \
     -H 'X-Admin-Key: your-secret-key'
```

**When models are running:**
```json
{"ok": true, "message": "Stopped 2 model(s) — will restart implicitly on next request", "stopped": ["qwen3.5-4b", "gemma-4-e2b"]}
```

**When no models are running:**
```json
{"ok": true, "message": "No models running — nothing to restart", "stopped": []}
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

Removes a model from cache without modifying config. Stops the model first (if running), deletes its checkpoint files from the HF cache directory, and clears all runner context entries.

```json
{"ok": true, "model": "qwen3.5-4b"}
```

The cache path is computed from `config.yaml`'s `env.HF_HUB_CACHE` (or `LLAMA_CACHE` fallback) using the standard HF Hub layout: `<cache>/models--{checkpoint.replace('/', '--')>`.

Always returns `200 OK`. Returns `404` if the model doesn't exist in config. If a context has already been cleared by eject, subsequent `stop()` calls will return `404`.

### GET /admin/log/{model}?lines=N&follow=true

Return log lines for a running model. Without `follow`, returns a JSON snapshot. With `follow=true`, switches to an SSE stream.

**Snapshot mode:**
```json
{"object": "log", "data": ["[INFO] Loading model...", "[INFO] Ready"]}
```

**SSE mode** — streams new lines as they are produced:
```
data: {"type":"snapshot","lines":["line1","line2"]}

data: {"type":"line","lines":["new line 3"]}

data: [DONE]
```

Returns `404` if the model is not found in config. Uses the log buffer populated by either the process watcher (ProcessRunner) or the Docker log capture subprocess (`docker logs -f`), both feeding the same deque.

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

**Logs pane** — Terminal-style log viewer for a selected model:
- Snapshot mode (default): loads last 200 lines via `GET /admin/log/{model}?lines=200`
- Follow mode: toggle "Follow" checkbox to enable SSE streaming — new lines appear in real-time
- Smart auto-scroll: scrolls to bottom during streaming *unless* you've scrolled up to read older logs (then it follows only when you return)
- Clear button empties the pane content
- Stream is automatically aborted on model change or accordion close

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
