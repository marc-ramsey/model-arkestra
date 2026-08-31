# Widget Engine — JSON-Driven Layout System

ModelArkestra admin UI is driven by a single JSON tree rendered into DOM at runtime. No class hierarchies. No inheritance. Data → render → DOM.

## Quick Start

```json
{ "widget":"SplitPane", "axis":"h", "ratio":40, "children":[
    { "widget":"AccordionContainer" },
    { "widget":"SplitPane", "axis":"v", "ratio":65, "children":[
        { "widget":"LogPane" },
        { "widget":"ChatPane" }
    ]}
]}
```

Render it:
```js
import render from '/static/widget.js'; // exports window.render
const tree = await fetch('/static/app.json').then(r => r.json());
document.getElementById('app-root').replaceWith(render(tree));
```

## Tree Structure

Every node has a `widget` field. The renderer matches widget name to DOM creation logic. Nodes nest children recursively — no fixed depth.

| Field | Purpose | Required |
|-------|---------|----------|
| `widget` | Widget factory name (SplitPane, AccordionContainer, LogPane, ChatPane) | Yes |
| `axis` | `'h'` or `'v'` split direction | SplitPane |
| `ratio` | Initial flex percentage before divider | SplitPane |
| `children` | Array of child nodes | Containers |

## Widget Types

### SplitPane

Flex container with draggable divider between siblings. Auto-persists ratio to localStorage on drag end. Overridden by stored value on next load.

```json
{ "widget":"SplitPane", "axis":"h", "ratio":40, "children": [...] }
```

No `ratio` = fixed layout (children sized by their own CSS).

### AccordionContainer

Collapsible section with a single content body. The model list is deferred on first click — the body starts empty and populates when a model row is clicked.

```json
{ "widget":"AccordionContainer" }
```

### LogPane

Read-only log display with model selector dropdown. Fetches log snapshots on load and polls for deltas. Shows lines in chronological order, auto-scrolls to bottom.

No child nodes needed — self-contained pane.

### ChatPane

Chat interface with message area, model selector, params panel, and input bar. Messages rendered as user/assistant bubbles. Supports SSE streaming from the model's port.

The params panel (Temp, Max Tokens, Top-P, Top-K) is hidden by default; toggled via the "Params" span in the header.

No child nodes needed — self-contained pane.

### ConfigPanel (Deferred)

Not defined in JSON. Created dynamically when a user clicks a model row via `renderModelRow(model)` from `app.js`. Displays editable fields from `/admin/config/{model}` with action buttons: reset, stop, start, save, eject.

## Event Wiring

### Click Actions (Delegation)

Buttons wired by ID convention — no explicit listener registration.

Convention: `id="btn-{action}-{context}"` → calls `actions.action(contextId)` from the action map in `app.js`.

Example button generated in ConfigPanel:
```html
<button id="btn-stop-gemma-4-e2b">Stop</button>
```

Action map (defined in `app.js`):
```js
const actions = {
    start(id)   { adminPost('/admin/start/'+id, {}); },
    stop(id)    { adminPost('/admin/stop/'+id, {}); },
};
```

The `wireEvents(actions)` call installs click delegation on `document.body`. Any element matching `btn-{action}-{id}` triggers the corresponding action handler.

### Field Change Events (Delegation)

Inputs wired by ID convention: `id="f-{context}-{name}"`. An `input` event triggers a custom `CustomEvent('field.change')` bubbling up from the element, with structured detail:

```js
{ detail: { context: 'gemma', name: 'checkpoint', value: 'v2.1' } }
```

Subscribers in `app.js` handle dirty detection and save state:
```js
document.addEventListener('field.change', (e) => {
    const d = e.detail;
    if (d.name === 'checkpoint') markDirty(d.context);
});
```

### Cross-Widget Communication

Use the shared `EventBus` for broader events. Defined at module scope and exported on `window`.

```js
EventBus.on('model.select', data => { /* handle */ });
EventBus.emit('model.select', { modelId: 'gemma' });
```

### Model Row Click

Clicking a `.model-row` triggers two flows:
1. Calls `actions.click(modelId)` if defined — loads config panel via deferred render
2. Starts log polling via `startLogPoll(modelId)` for the selected model's log stream

The model row click handler is installed by `wireEvents` via delegation matching `e.target.closest('.model-row')`.

## Actions and Handlers

All domain logic lives in the action map (`actions` object) defined in `app.js`. The widget renderer is agnostic — it only wires IDs to function lookups.

To add a new action:
1. Ensure buttons/fields follow ID convention (`btn-*`, `f-*`)
2. Add handler to `actions`:
```js
const actions = {
    eject(id)  { adminPost('/admin/eject/'+id, {}); },
};
```

No render() change needed — wiring is automatic by convention.

## Theming

JSON defines structure only. All visual appearance (colors, spacing, fonts, borders) comes from `style.css`. Swap CSS to change the look — JSON stays identical.

Key CSS classes:
- `.accordion.collapsed` → hides accordion body
- `.chat-params-panel.open` → shows/hides params panel
- `.btn-danger`, `.btn-success` → action button variants
- `.dirty` → marks unsaved config changes

## Patterns

### Dirty Detection

ConfigPanel fields snapshot initial values on render. Each `input` event compares the field's current value to its snapshot. If changed, a `.dirty` class is added and save button state updates. Implemented in `checkDirty(modelId, panel)`.

### Layout Persistence

SplitPane drag end listeners save ratios to localStorage keys `arkestra-layout-h` and `arkestra-layout-v`. On next render, stored values override JSON `ratio` for persistent user preferences.

### Deferred Loading

AccordionContainer body starts empty. Model rows are rendered by `renderModelRow(model)` from `app.js`, which:
1. Fetches model data via `adminGet('/admin/models')`
2. Calls `actions.click(id)` on row click → loads config panel
3. Shows status dots reflecting current runner state

This keeps the initial render lightweight — models load asynchronously after the layout tree is mounted.

### Log Polling

LogPane calls `startLogPoll(modelId)` which:
1. Fetches a snapshot (200 lines) immediately
2. Sets up interval polling using cursor (`since` param) for delta fetches
3. Appends new lines and auto-scrolls if near bottom
4. Pauses during chat streaming via `stopLogPoll()`

## Limitations

- No conditional rendering or loops in JSON
- No computed fields — data transformations handled by JS action handlers
- All action handlers must be registered in `actions` map before `wireEvents()` runs
- No built-in error states — handle HTTP failures in action handlers and render accordingly
- Widget names are case-sensitive (capitalized: SplitPane, LogPane)

## Streaming Audio

Real-time voice chat is handled by the `AudioStream` class in `widget-audio.js`. It manages a single WebSocket connection to `/ark/audio/stream` for bidirectional audio communication.

### AudioStream API

```js
const stream = window._audioStream;
```

| Method | Description |
|---|---|
| `connect()` | Opens WebSocket. Idempotent — safe to call multiple times. |
| `speak(text)` | Sends TTS request (`{type:"tts",text}`). Server replies with binary WAV bytes. |
| `startRecording()` | Opens mic via `getUserMedia`, samples at 16kHz, sends frames every 50ms as base64 JSON. |
| `stopRecording()` | Closes mic stream, waits for in-flight results (~200ms). |

**Callbacks:**

| Callback | Signature | Purpose |
|---|---|---|
| `onPartialTranscript(text)` | `(string) => void` | Called on each partial result — use for live display (grows while speaking). |
| `onFinalTranscript(text)` | `(string) => void` | Called when VAD detects end of speech sentence. |

**Example — Mic button handler:**
```js
micBtn.onclick = async () => {
  if (!isRecording) {
    await stream.connect();
    stream.onPartialTranscript = text => { displayArea.textContent = text; };
    stream.onFinalTranscript = text => { appendToChat(text); };
    await stream.startRecording();
    isRecording = true;
  } else {
    await stream.stopRecording();
    isRecording = false;
  }
};
```

### Audio Frame Format

Audio frames are sent as base64-encoded PCM float32 in JSON:
```json
{"type":"audio_frame","data":"<base64>"}
```

- **Rate**: 16kHz mono (client decimates from system capture rate ~48kHz)
- **Format**: IEEE float32, amplitude [-1.0, 1.0]
- **Chunk size**: ~50ms (~800 samples = 3200 bytes → ~4300B JSON frame)
- Server-side: librosa decodes and resamples as needed — STT input is always 16kHz mono float32

### TTS Playback

Server responds to `{"type":"tts","text":"..."}` with binary WAV bytes. The client decodes via `AudioContext.decodeAudioData()` and queues for playback automatically.

### Integration Points

- `sendTTS()` — delegates to `stream.speak()`, falls back to `/v1/audio/speech` POST if WebSocket fails
- `wireMicBtn()` — replaces batch MediaRecorder→POST pattern with streaming AudioStream API
- `wireAsrUpload()` — unchanged; still uses `/v1/audio/transcriptions` POST for file uploads (separate from mic)
