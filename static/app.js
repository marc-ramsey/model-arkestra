// ── 1.0 Configuration ─────────────────────────────────────
const ADMIN_KEY    = document.querySelector('meta[name="arkestra-admin-key"]')?.content || '';
const BASE_URL     = window.location.origin;
const POLL_INTERVAL = 2000;            // ms — settable constant for model status polling

// ── 1.2 Global State ──────────────────────────────────────
let modelsCache      = [];          // from GET /admin/models
let clusterMap       = {};          // { name: { base_url, healthy } }
let modelStates      = {};          // { modelName: { config, snapshot, dirty, expanded } }
let logModel         = '';          // currently selected model for logs pane
let chatModel        = '';          // currently selected model for chat pane
let backendOptions   = {};           // backends dict from /admin/models response
let runnerTypes      = [];          // runner types from /admin/models response

// Drag state for resize handles
let isVDragging  = false;           // vertical (left/right) divider
let isHDragging  = false;           // horizontal (logs/chat) divider

// Log polling state
let logPollTimer    = null;         // interval id for log polling
let logLastSeq    = 0;            // server-provided sequence cursor for log polling
let isChatStreaming  = false;       // true while a chat response is streaming

// Streaming markdown accumulator — tokens buffered and re-parsed incrementally
let _streamAcc    = '';               // accumulated raw text of current stream
let _streamBubble = null;             // DOM element receiving the stream

// ── 1.3 Fetch Helpers ─────────────────────────────────────
function adminFetch(path, options = {}) {
    const headers = { ...options.headers };
    if (ADMIN_KEY) {
        headers['X-Admin-Key'] = ADMIN_KEY;
    }
    console.log('[adminFetch]', BASE_URL + path, 'key=' + (ADMIN_KEY ? '✓' : '(none)'));
    const opts = { method: 'GET', headers, ...options };
    // Remove Content-Type for GET (browser handles it)
    if (opts.method === 'GET') delete opts.headers['Content-Type'];

    return fetch(`${BASE_URL}${path}`, opts).then(async r => {
        const body = await r.text();
        if (!r.ok) {
            let detail;
            try { detail = JSON.parse(body).detail || body; }
            catch { detail = body; }
            throw new Error(`[admin ${r.status}] ${detail}`);
        }
        return body ? JSON.parse(body) : null;
    });
}

function adminGet(path, params) {
    const qs = params ? '?' + new URLSearchParams(params).toString() : '';
    return adminFetch(path + qs);
}

function adminPost(path, body) {
    return adminFetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
}

function adminPut(path, body) {
    return adminFetch(path, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
}

// ── 4. Model List — fetch + render accordion items ───────────────
function updateModelCount() {
    const badge = document.getElementById('model-count');
    if (badge) badge.textContent = '(' + modelsCache.length + ' models)';
}

function buildAccordionItems() {
    const container = document.getElementById('model-accordion-items');
    if (!container) return;

    // Preserve expansion state before rebuilding
    const wasExpanded = new Set(
        Array.from(container.querySelectorAll('.cluster-item:not(.collapsed), .accordion-item:not(.collapsed)'))
            .map(el => el.id)
    );

    // Group models by cluster (strip prefix from model name)
    const modelsByCluster = {};
    for (const m of modelsCache) {
        let localId = m.id;
        if (m.id.includes('/')) {
            localId = m.id.split('/', 2)[1];
        }
        // Determine cluster from name prefix or default to 'local'
        let cluster = 'local';
        if (m.id.includes('/')) {
            cluster = m.id.split('/', 2)[0];
        } else if (Object.keys(clusterMap).length > 1) {
            cluster = 'local';
        }
        if (!modelsByCluster[cluster]) modelsByCluster[cluster] = [];
        modelsByCluster[cluster].push({ ...m, localId });
    }

    // Build HTML: clusters → model rows
    let html = '';
    for (const [clusterName, clusterModels] of Object.entries(modelsByCluster)) {
        const clustCfg = clusterMap[clusterName] || { healthy: true };
        const healthClass = clustCfg.healthy ? '' : ' unhealthy';
        const stateKey = 'sec-cluster-' + clusterName.replace(/[^a-zA-Z0-9_-]/g, '_');
        const isExpanded = wasExpanded.has(stateKey);

        html += '<div class="cluster-item collapsed" id="' + stateKey + '">';
        html += '  <div class="cluster-header">'
            + '    <span class="cluster-health-dot' + healthClass + '"></span>'
            + '    <span class="cluster-name">' + clusterName + '</span>'
            + '    <span class="cluster-count">' + clusterModels.length + ' model' + (clusterModels.length !== 1 ? 's' : '') + '</span>'
            + '  </div>';
        html += '  <div class="cluster-models">';

        for (const m of clusterModels) {
            const name = m.id;
            const statusClass = m.status ? m.status.replace('runnerstate.', '').toLowerCase() : 'uncached';
            const stateKeyM = 'sec-model-' + name.replace(/[^a-zA-Z0-9_-]/g, '_');
            html += '<div class="model-row" id="' + stateKeyM + '" data-model="' + name + '">'
                + '  <span class="status-dot ' + statusClass + '" id="dot-' + name.replace(/[^a-zA-Z0-9_-]/g, '_') + '"></span>'
                + '  <span class="model-name">' + (name.includes('/') ? m.localId : name) + '</span>'
                + '</div>';
            html += '  <div class="model-config-panel" id="body-' + name.replace(/[^a-zA-Z0-9_-]/g, '_') + '">'
                + '    <p class="placeholder-text">Click to load configuration…</p>'
                + '  </div>';
        }

        html += '  </div></div>';
    }
    container.innerHTML = html;

    // Restore previously-expanded items
    for (const id of wasExpanded) {
        const el = document.getElementById(id);
        if (el) el.classList.remove('collapsed');
    }
}

async function refreshModels() {
    try {
        // Fetch models, clusters, and config
        const [statusData, clusterData, configData] = await Promise.all([
            adminGet('/admin/models'),
            adminGet('/admin/clusters').catch(() => ({ clusters: [] })),
            adminGet('/admin/config'),
        ]);

        let listChanged = false;
        if (statusData && Array.isArray(statusData.models)) {
            const newNames = new Set(statusData.models.map(m => m.id));
            const oldNames = new Set(modelsCache.map(m => m.id));
            listChanged = newNames.size !== oldNames || ![...newNames].every(n => oldNames.has(n));

            modelsCache = statusData.models;
            backendOptions = statusData.backends || {};
            runnerTypes = statusData.runner_types || [];
        } else if (!modelsCache.length) {
            // Fallback: just names from config
            updateModelCount();
            buildAccordionItems();
            populateRightDropdowns();
            return;
        }

        // Parse cluster data
        if (clusterData && Array.isArray(clusterData.clusters)) {
            const oldKeys = new Set(Object.keys(clusterMap));
            const newKeys = new Set(clusterData.clusters.map(c => c.name));
            if (oldKeys.size !== newKeys || ![...newKeys].every(n => oldKeys.has(n))) {
                listChanged = true;
            }
            clusterMap = {};
            for (const c of clusterData.clusters) {
                clusterMap[c.name] = { base_url: c['base-url'], healthy: c.healthy };
            }
        } else if (!Object.keys(clusterMap).length) {
            // No clusters data — fall back to flat view with 'local' only
            clusterMap = { local: { base_url: null, healthy: true } };
        }

        updateModelCount();
        populateRightDropdowns();

        // Only rebuild DOM if models were added or removed
        if (listChanged || !document.getElementById('model-accordion-items')) {
            buildAccordionItems();
        } else {
            // Just update status dots in place
            for (const m of modelsCache) {
                const name = m.id;
                const dotId = 'dot-' + name.replace(/[^a-zA-Z0-9_-]/g, '_');
                const dot = document.getElementById(dotId);
                if (dot && m.status) {
                    const statusClass = m.status.replace('runnerstate.', '').toLowerCase();
                    dot.className = 'status-dot ' + statusClass;
                }
            }
        }

    } catch (e) {
        console.error('[admin] failed to refresh models:', e.message);
        const badge = document.getElementById('model-count');
        if (badge) badge.textContent = '(!error)';
    }
}

function populateRightDropdowns() {
    for (const id of ['log-model-select', 'chat-model-select']) {
        const sel = document.getElementById(id);
        if (!sel) continue;
        const current = sel.value;          // preserve selection
        let html = '<option value="">— none —</option>';
        for (const m of modelsCache) {
            html += '<option value="' + m.id + '"' + (m.id === current ? ' selected' : '') + '>' + m.id + '</option>';
        }
        if (id === 'log-model-select') {
            html += '<optgroup label="">';
            html += '<option value="*"' + ('*' === current ? ' selected' : '') + '>Server logs</option>';
            html += '</optgroup>';
        }
        sel.innerHTML = html;
    }
}

// ── 5. Per-model config panel on expand ─────────────────────
async function populateModelPanel(modelName) {
    const bodyId = 'body-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_');
    const panel = document.getElementById(bodyId);
    if (!panel) return;

    try {
        const data = await adminGet('/admin/config/' + encodeURIComponent(modelName));
        if (!data || !data.config) return;

        const cfg = data.config;
        // Available capabilities resolved from config (normal chain:
        // per-model → default-capabilities → hardcoded fallback)
        const allCaps = data.available_capabilities || ['chat'];
        // Deep clone for dirty detection snapshot
        modelStates[modelName] = { config: cfg, snapshot: JSON.parse(JSON.stringify(cfg)), dirty: false, allCaps };

        let html = '';

        // Checkpoint field
        html += '<div class="edit-field"><label for="mc-checkpoint-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_') + '">Checkpoint</label>';
        html += '  <input type="text" id="mc-checkpoint-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_') + '" value="' + escapeAttr(cfg.checkpoint || '') + '"></div>';

        // Backend select
        const backendSelId = 'mc-backend-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_');
        html += '<div class="edit-field"><label for="' + backendSelId + '">Backend</label>';
        html += '  <select id="' + backendSelId + '">';
        html += '    <option value="">(default)</option>';
        if (backendOptions) {
            for (const [key, val] of Object.entries(backendOptions)) {
                const label = (typeof val === 'object') ? (val.host || key) : key;
                html += '    <option value="' + escapeAttr(key) + '"' + ((cfg.backend === key) ? ' selected' : '') + '>' + escapeHtml(label) + '</option>';
            }
        }
        html += '  </select></div>';

        // Runner select
        const runnerSelId = 'mc-runner-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_');
        html += '<div class="edit-field"><label for="' + runnerSelId + '">Runner</label>';
        html += '  <select id="' + runnerSelId + '">';
        html += '    <option value="">(auto)</option>';
        if (runnerTypes) {
            for (const t of runnerTypes) {
                html += '    <option value="' + escapeAttr(t) + '"' + ((cfg.runner === t) ? ' selected' : '') + '>' + escapeHtml(t) + '</option>';
            }
        }
        html += '  </select></div>';

        // Args textarea
        html += '<div class="edit-field"><label for="mc-args-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_') + '">Args</label>';
        html += '  <textarea id="mc-args-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_') + '" rows="2">' + escapeHtml(cfg.args || '') + '</textarea></div>';

        // Capabilities chips
        const capsId = 'mc-caps-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_');
        html += '<div class="edit-field"><label>Capabilities</label><div class="capabilities-list" id="' + capsId + '">';
        const selectedCaps = new Set(cfg.capabilities || allCaps);
        for (const c of allCaps) {
            html += '  <span class="capability-chip' + (selectedCaps.has(c) ? ' active' : '') + '" data-cap="' + c + '">' + c + '</span>';
        }
        html += '</div></div>';

        // Log buffer size
        html += '<div class="edit-field"><label for="mc-log-buffer-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_') + '">Log Buffer Size</label>';
        html += '  <input type="number" id="mc-log-buffer-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_') + '" min="50" max="10000" step="50" value="' + (cfg.max_log_lines || 500) + '"></div>';

        // Action buttons
        html += '<div class="model-actions">';
        const btnId = (id) => 'mc-' + id + '-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_');
        html += '  <button type="button" title="Reset config to server value" data-model="' + escapeAttr(modelName) + '" data-action="reset" id="' + btnId('reset') + '">↺</button>';
        html += '  <button type="button" title="Stop model runner" data-model="' + escapeAttr(modelName) + '" data-action="stop" id="' + btnId('stop') + '" class="btn-danger">⏹</button>';
        html += '  <button type="button" title="Start/Restart with current config" data-model="' + escapeAttr(modelName) + '" data-action="start" id="' + btnId('start') + '" class="btn-success">▶</button>';
        html += '  <button type="button" title="Save config changes" data-model="' + escapeAttr(modelName) + '" data-action="save" id="' + btnId('save') + '">✓</button>';
        html += '  <button type="button" title="Eject model (stop + delete cache)" data-model="' + escapeAttr(modelName) + '" data-action="eject" id="' + btnId('eject') + '" class="btn-danger">⏏</button>';
        html += '</div>';

        panel.innerHTML = html;

        panel.dataset.fetched = '1';

        // Wire up capability chip toggles
        const chipsContainer = document.getElementById(capsId);
        if (chipsContainer) {
            chipsContainer.addEventListener('click', (e) => {
                const chip = e.target.closest('.capability-chip');
                if (!chip) return;
                const cap = chip.dataset.cap;
                if (selectedCaps.has(cap)) selectedCaps.delete(cap);
                else selectedCaps.add(cap);
                chip.classList.toggle('active');
                maybeUpdateDirty(modelName);
            });
        }

        // Wire up all inputs for dirty detection
        panel.querySelectorAll('input, select, textarea').forEach(el => {
            el.addEventListener('input', () => maybeUpdateDirty(modelName));
        });

        // Wire up action buttons via delegation on the panel
        panel.querySelector('[data-action="save"]')?.addEventListener('click', () => saveModelConfig(modelName));
        panel.querySelector('[data-action="reset"]')?.addEventListener('click', () => resetModelConfig(modelName));
        panel.querySelector('[data-action="start"]')?.addEventListener('click', () => startModel(modelName));
        panel.querySelector('[data-action="stop"]')?.addEventListener('click', () => stopModel(modelName));
        panel.querySelector('[data-action="eject"]')?.addEventListener('click', () => ejectModel(modelName));

    } catch (e) {
        console.error('[admin] failed to load config for ' + modelName + ':', e.message);
        panel.innerHTML = '<p class="placeholder-text" style="color:var(--red)">Failed to load config: ' + e.message + '</p>';
    }
}

// Helper: get values from a model config panel
function getModelFormValues(modelName) {
    const prefix = modelName.replace(/[^a-zA-Z0-9_-]/g, '_');
    return {
        checkpoint: document.getElementById('mc-checkpoint-' + prefix)?.value || '',
        backend: (document.getElementById('mc-backend-' + prefix)?.value) || undefined,
        runner: (document.getElementById('mc-runner-' + prefix)?.value) || undefined,
        args: (document.getElementById('mc-args-' + prefix)?.value) || undefined,
        max_log_lines: parseInt(document.getElementById('mc-log-buffer-' + prefix)?.value) || 500,
    };
}

function getModelCapabilities(modelName) {
    const prefix = modelName.replace(/[^a-zA-Z0-9_-]/g, '_');
    const chips = document.querySelectorAll('#mc-caps-' + prefix + ' .capability-chip.active');
    return Array.from(chips).map(c => c.dataset.cap);
}

// Dirty detection for a model panel
function maybeUpdateDirty(modelName) {
    if (!modelStates[modelName]) return;
    const current = getModelFormValues(modelName);
    const caps = getModelCapabilities(modelName);
    const snap = modelStates[modelName].snapshot;
    const isDirty = (
        current.checkpoint !== snap.checkpoint ||
        (current.backend ?? '') !== (snap.backend ?? '') ||
        (current.runner ?? '') !== (snap.runner ?? '') ||
        current.args !== (snap.args ?? '') ||
        current.max_log_lines !== (snap.max_log_lines ?? 500) ||
        JSON.stringify(caps.sort()) !== JSON.stringify((snap.capabilities || snap.allCaps || ['chat']).sort())
    );
    modelStates[modelName].dirty = isDirty;
    const btn = document.getElementById('mc-save-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_'));
    if (btn) btn.disabled = !isDirty;
}

// Action: Save config
async function saveModelConfig(modelName) {
    const body = getModelFormValues(modelName);
    body.capabilities = getModelCapabilities(modelName);
    try {
        await adminPut('/admin/config/' + encodeURIComponent(modelName), body);
        modelStates[modelName].snapshot = JSON.parse(JSON.stringify(body));
        modelStates[modelName].dirty = false;
        const btn = document.getElementById('mc-save-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_'));
        if (btn) btn.disabled = true;
        refreshModels();
    } catch (e) {
        console.error('[admin] save failed for ' + modelName + ':', e.message);
    }
}

// Action: Reset config
async function resetModelConfig(modelName) {
    try {
        const data = await adminGet('/admin/config/' + encodeURIComponent(modelName));
        if (data && data.config) {
            modelStates[modelName].snapshot = JSON.parse(JSON.stringify(data.config));
            modelStates[modelName].dirty = false;
            // Re-populate the panel with reset values
            populateModelPanel(modelName);
        }
    } catch (e) {
        console.error('[admin] reset failed for ' + modelName + ':', e.message);
    }
}

// Action: Start model
async function startModel(modelName) {
    const btn = document.getElementById('mc-start-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_'));
    if (!btn) return;
    const origDisabled = btn.disabled;
    btn.disabled = true;

    const body = getModelFormValues(modelName);
    body.capabilities = getModelCapabilities(modelName);

    try {
        await adminPost('/admin/start/' + encodeURIComponent(modelName), body);
        refreshModels();
    } catch (e) {
        console.error('[admin] start failed for ' + modelName + ':', e.message);
    } finally {
        btn.disabled = origDisabled;
    }
}

// Action: Stop model
async function stopModel(modelName) {
    const btn = document.getElementById('mc-stop-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_'));
    if (!btn) return;
    const origDisabled = btn.disabled;
    btn.disabled = true;

    try {
        await adminPost('/admin/stop/' + encodeURIComponent(modelName));
        refreshModels();
    } catch (e) {
        console.error('[admin] stop failed for ' + modelName + ':', e.message);
    } finally {
        btn.disabled = origDisabled;
    }
}

// Action: Eject model
async function ejectModel(modelName) {
    const btn = document.getElementById('mc-eject-' + modelName.replace(/[^a-zA-Z0-9_-]/g, '_'));
    if (!btn) return;
    const origDisabled = btn.disabled;
    btn.disabled = true;

    try {
        await adminPost('/admin/eject/' + encodeURIComponent(modelName));
        refreshModels();
    } catch (e) {
        console.error('[admin] eject failed for ' + modelName + ':', e.message);
    } finally {
        btn.disabled = origDisabled;
    }
}

// HTML escaping helpers
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
}
function escapeAttr(str) {
    return (str || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── 2.3 Model Selection ───────────────────────────────────
async function selectModel(name) {
    stopChatStream();
    clearChatHistory();
    if (name && modelsCache.length > 0) {
        loadLogSnapshot(name);
        startLogPoll(name);
    } else {
        const display = document.getElementById('log-display');
        if (display) {
            display.textContent = '';
            const ph = document.createElement('p');
            ph.className = 'placeholder-text';
            ph.textContent = 'Select a model to view logs.';
            display.appendChild(ph);
        }
    }
}

// ── 5. Logs pane functions ────────────────────────────────
function stopLogPoll() {
    if (logPollTimer) {
        clearInterval(logPollTimer);
        logPollTimer = null;
    }
    logLastSeq = 0;
}

async function loadLogSnapshot(modelName) {
    stopLogPoll();

    const display = document.getElementById('log-display');
    if (!display) return;
    display.textContent = '';

    if (!modelName) {
        const ph = document.createElement('p');
        ph.className = 'placeholder-text';
        ph.textContent = 'Select a model to view logs.';
        display.appendChild(ph);
        return;
    }

    try {
        const url = modelName === '*' ? '/admin/logs' : '/admin/log/' + encodeURIComponent(modelName);
        const data = await adminGet(url, { lines: 200 });
        const lines = data?.lines ?? [];
        logLastSeq = data.since ?? data.seq ?? 0;

        for (const entry of lines) {
            const div = document.createElement('div');
            div.className = 'log-line';
            div.textContent = entry.text;
            display.appendChild(div);
        }
    } catch (e) {
        console.error('[admin] log snapshot failed for ' + modelName + ':', e.message);
    }
}


function appendLogLines(lines) {
    const display = document.getElementById('log-display');
    if (!display || !lines.length) return;

    // Only auto-scroll if user is near bottom
    const nearBottom = display.scrollHeight - display.scrollTop <= display.clientHeight + 60;

    lines.forEach(line => {
        const div = document.createElement('div');
        div.className = 'log-line';
        div.textContent = line;
        display.appendChild(div);
    });

    logLastSeq += lines.length;

    // Auto-scroll only if user is already at bottom
    if (nearBottom) display.scrollTop = display.scrollHeight;
}

function startLogPoll(modelName) {
    stopLogPoll();
    const pollOne = async () => {
        try {
            // On first call (logLastSeq == 0), do a snapshot fetch
            if (logLastSeq === 0) {
                const url = modelName === '*' ? '/admin/logs' : '/admin/log/' + encodeURIComponent(modelName);
                const data = await adminGet(url, { lines: 200 });
                logLastSeq = data.since ?? data.seq ?? 0;
                const newEntries = data?.lines ?? [];
                if (newEntries.length > 0) appendLogLines(newEntries.map(e => e.text));
            } else {
                const url = modelName === '*' ? '/admin/logs' : '/admin/log/' + encodeURIComponent(modelName);
                const data = await adminGet(url, { lines: 200, since: logLastSeq });
                if (!data || !data.lines) return;
                logLastSeq = data.since ?? data.seq ?? logLastSeq;
                const newEntries = data.lines;
                if (newEntries.length > 0) appendLogLines(newEntries.map(e => e.text));
            }
        } catch (e) {
            // Silent fail — don't flood console on transient issues
        }
    };
    pollOne();          // fetch immediately
    logPollTimer = setInterval(pollOne, POLL_INTERVAL);
}

// ── 6. Chat pane functions ────────────────────────────────
let chatHistory      = [];   // full conversation history for selected model
let chatAbortCtrl    = null; // AbortController for active chat stream
const CHAT_PARAMS_KEY = 'arkestra-chat-params';

// 6.1 Parameters panel toggle
function toggleChatParams() {
    const panel = document.getElementById('chat-params-panel');
    const toggle = document.getElementById('right-chat-params-toggle');
    if (!panel || !toggle) return;
    panel.classList.toggle('open');
    const isOpen = panel.classList.contains('open');
    toggle.textContent = isOpen ? '⚙ Parameters ▴' : '⚙ Parameters ▾';
    requestAnimationFrame(() => layoutRightPanels());
}

// 6.2 Param persistence
function getChatParams() {
    const stored = localStorage.getItem(CHAT_PARAMS_KEY);
    if (stored) {
        try { return JSON.parse(stored); } catch { return {}; }
    }
    return {};
}

function saveChatParams() {
    if (!chatModel) return;
    const params = getChatParams();
    params[chatModel] = {
        temperature: parseFloat(document.getElementById('rcp-temp')?.value) || 0.7,
        top_p: parseFloat(document.getElementById('rcp-top-p')?.value) || 0.95,
        max_tokens: parseInt(document.getElementById('rcp-max-tokens')?.value) || 512,
    };
    localStorage.setItem(CHAT_PARAMS_KEY, JSON.stringify(params));
}

function restoreChatParams() {
    if (!chatModel) return;
    const params = getChatParams()[chatModel];
    if (!params) return;
    document.getElementById('rcp-temp').value     = String(params.temperature ?? 0.7);
    document.getElementById('rcp-top-p').value     = String(params.top_p ?? 0.95);
    document.getElementById('rcp-max-tokens').value = String(params.max_tokens ?? 512);
}

// 6.3 Message bubble helpers
function appendMessageBubble(role, content, isStreaming) {
    const container = document.getElementById('chat-display');
    if (!container) return;

    // Determine if we should auto-scroll: always for user messages,
    // only at bottom for assistant during streaming
    const nearBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 40;

    const div = document.createElement('div');
    div.className = `message ${role}${isStreaming ? ' streaming' : ''}`;

    if (role === 'user') {
        div.textContent = content;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;  // Always scroll for user
        return div;
    } else {
        div.innerHTML = isStreaming
            ? content + '<span class="cursor"></span>'
            : content;
        container.appendChild(div);
        if (nearBottom) container.scrollTop = container.scrollHeight;
        return div;
    }
}

// 6.3 Send message (form submit handler)
async function sendMessage(e) {
    e.preventDefault();
    const input = document.getElementById('right-chat-input');
    if (!input || !chatModel || isChatStreaming) return;

    const userText = input.value.trim();
    if (!userText) return;

    // Append user message to UI and history
    appendMessageBubble('user', userText, false);
    chatHistory.push({ role: 'user', content: userText });
    input.value = '';

    // Create assistant bubble (streaming)
    const assitDiv = appendMessageBubble('assistant', '', true);

    isChatStreaming = true;
    document.getElementById('right-chat-send-btn').disabled = true;

    // Initialize streaming accumulator and DOM reference
    _streamAcc = '';
    _streamBubble = assitDiv;

    try {
        await streamChatResponse(chatHistory, chatModel, (delta) => {
            // Accumulate tokens; renderMarkdownStream handles throttled re-parse
            _streamAcc += delta;
            renderMarkdownStream(_streamAcc);
        });

        // Stream complete — final markdown render without cursor
        isChatStreaming = false;
        finishStreamText(assitDiv);
    } catch (err) {
        if (err.name === 'AbortError') {
            // Model switched mid-stream — flag for clearChatHistory so
            // it shows an interruption marker instead of vanishing silently.
            _chatWasAborted = true;
            finishStreamText(assitDiv);  // remove cursor, show what we have
        } else {
            console.error('[chat] stream failed:', err.message);
            appendMessageBubble('assistant', `⚠ Error: ${err.message}`, false);
        }
        isChatStreaming = false;
    }

    document.getElementById('right-chat-send-btn').disabled = false;
}

// 6.4 Streaming response handler
async function streamChatResponse(messages, modelName, onToken) {
    stopLogPoll();    // pause log polling during chat streaming

    chatAbortCtrl = new AbortController();

    const modelInfo = modelsCache.find(m => m.id === modelName);
    if (!modelInfo || !modelInfo.port) {
        throw new Error('Model is not running or has no port');
    }

    const allParams = getChatParams();
    const mp = allParams[modelName] || { temperature: 0.7, top_p: 0.95, max_tokens: 512 };

    const payload = {
        model: modelName,
        messages: messages,
        stream: true,
        temperature: mp.temperature,
        top_p: mp.top_p,
        max_tokens: mp.max_tokens,
    };

    const port = modelInfo.port;
    const url = 'http://127.0.0.1:' + port + '/v1/chat/completions';

    const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: chatAbortCtrl.signal,
    });

    if (!resp.ok) throw new Error(`Chat API ${resp.status}: ${await resp.text()}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();  // partial line stays

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            // Handle SSE format: "data: {...}" or "data: [DONE]"
            let text = trimmed;
            if (text.startsWith('data: ')) {
                text = text.slice(6);
            }

            if (text === '[DONE]') return;

            try {
                const parsed = JSON.parse(text);
                const delta = parsed.choices?.[0]?.delta?.content || '';
                if (delta) onToken(delta);
            } catch { /* skip malformed SSE lines */ }
        }
    }
}

// 6.5 Markdown rendering with marked.js fallback
function renderMarkdown(element, text) {
    return new Promise((resolve) => {
        if (typeof marked !== 'undefined') {
            element.innerHTML = marked.parse(text);
            resolve();
        } else {
            // marked is loaded with defer — wait for it
            const maxWait = 50;
            let attempts = 0;
            const poll = () => {
                if (typeof marked !== 'undefined') {
                    element.innerHTML = marked.parse(text);
                    resolve();
                } else if (++attempts < maxWait) {
                    setTimeout(poll, 100);
                } else {
                    // Fallback: plain text with basic formatting
                    element.textContent = text;
                    resolve();
                }
            };
            poll();
        }
    });
}

function clearChatHistory() {
    const container = document.getElementById('chat-display');
    if (!container) { chatHistory = []; return; }

    // If a stream was interrupted by model switch, remove the ghost
    // streaming assistant bubble so the user doesn't see their message vanish.
    if (_chatWasAborted) {
        const bubbles = container.querySelectorAll('.message');
        // Remove all messages up to and including the last assistant bubble
        for (let i = bubbles.length - 1; i >= 0; i--) {
            if (bubbles[i].classList.contains('assistant')) {
                while (i >= 0) container.removeChild(bubbles[i--]);
                break;
            }
        }
        _chatWasAborted = false;
    } else {
        container.textContent = '';
    }
    chatHistory = [];

    // Also reset streaming markdown state
    _streamAcc = '';
    _streamBubble = null;
}

// ── Incremental streaming markdown rendering ────────────────

/** Render accumulated text with markdown + blinking cursor (throttled to rAF). */
let _streamRenderPending = false;

function renderMarkdownStream(text) {
    if (_streamRenderPending) return;  // debounce — one parse per frame max
    _streamRenderPending = true;
    requestAnimationFrame(() => {
        _streamRenderPending = false;
        const el = _streamBubble;
        if (!el) return;
        if (typeof marked !== 'undefined') {
            el.innerHTML = marked.parse(text);
        } else {
            // marked not loaded yet — fall back to plain text
            el.textContent = text;
        }
        // Append blinking cursor as last child
        let cursor = el.querySelector('.cursor');
        if (!cursor) {
            cursor = document.createElement('span');
            cursor.className = 'cursor';
            el.appendChild(cursor);
        }
    });
}

/** Finalize stream: remove cursor, set final markdown-rendered content. */
function finishStreamText(bubble) {
    if (!bubble) return;
    bubble.classList.remove('streaming');
    const fullText = _streamAcc || bubble.textContent.replace(/\u200B/g, '').trim();
    _streamAcc = '';
    _streamBubble = null;
    renderMarkdown(bubble, fullText);
}

// (stopChatStream - called implicitly on model change)
function stopChatStream() {
    if (chatAbortCtrl) {
        chatAbortCtrl.abort();
        chatAbortCtrl = null;
    }
}

// Tracks whether stream was aborted mid-response, so clearChatHistory
// can clean up the ghost assistant bubble instead of vanishing silently.
let _chatWasAborted = false;

// ── Lifecycle ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    refreshModels();
    // Poll for model status updates at configurable interval
    setInterval(refreshModels, POLL_INTERVAL);

    // Search filter (2.2)
    const search = document.getElementById('model-search');
    if (search) search.addEventListener('input', e => {
        renderModelList(e.target.value);
    });

    // Model click delegation on the list
    const list = document.getElementById('model-list');
    if (list) list.addEventListener('click', e => {
        const item = e.target.closest('.model-item');
        if (!item || !item.dataset.model) return;
        selectModel(item.dataset.model);
    });

    // ── 4.1 Accordion toggle (shared) ─────────────────────────
    window.toggleAccordion = function(id) {
        const el = document.getElementById(id);
        if (!el) return;
        const wasCollapsed = el.classList.contains('collapsed');
        el.classList.toggle('collapsed');

        // Model row expanded → show config panel + load config
        if (el.classList.contains('model-row')) {
            el.classList.toggle('expanded', !wasCollapsed);
            if (wasCollapsed) {
                const modelName = el.dataset.model;
                if (modelName) populateModelPanel(modelName);
            }
        }

        requestAnimationFrame(() => layoutRightPanels());
    };

    // Delegate clicks on accordion items and model rows to toggleAccordion
    const itemContainer = document.getElementById('model-accordion-items');
    if (itemContainer) {
        itemContainer.addEventListener('click', (e) => {
            e.stopImmediatePropagation();

            // Cluster header click → toggle cluster
            const clusterHeader = e.target.closest('.cluster-header');
            if (clusterHeader) {
                const clusterItem = clusterHeader.closest('.cluster-item');
                if (clusterItem) toggleAccordion(clusterItem.id);
                return;
            }

            // Model row click → toggle model panel
            const modelRow = e.target.closest('.model-row[data-model]');
            if (modelRow) {
                const id = modelRow.id;
                // Also expand parent cluster if collapsed
                const parentCluster = modelRow.closest('.cluster-item');
                if (parentCluster && parentCluster.classList.contains('collapsed')) {
                    toggleAccordion(parentCluster.id);
                }
                toggleAccordion(id);
                return;
            }

            // Legacy h3 click for Settings and old accordions
            const h3 = e.target.closest('h3[data-model]');
            if (!h3) return;
            const section = h3.closest('.accordion-item');
            if (!section) return;
            toggleAccordion(section.id);
        });
    }

    // ── 4.2 Right-accordion height splitting ─────────────────
    function layoutRightPanels() {
        const accordion = document.getElementById('right-accordion');
        if (!accordion) return;
        const items = Array.from(accordion.querySelectorAll('.accordion-item'));
        // Count open items (not collapsed)
        const openItems = items.filter(i => !i.classList.contains('collapsed'));
        const closedCount = items.length - openItems.length;

        if (openItems.length === 0) {
            accordion.style.height = '100%';
            return;
        }

        // Each open pane gets equal share of the height minus collapsed panes
        // Collapse a single item takes ~32px (h3 area), so subtract that for each closed item
        const availableHeight = 100 - (closedCount * 0);
        items.forEach(item => {
            if (item.classList.contains('collapsed')) {
                item.style.flex = '0 0 auto';
            } else {
                item.style.flex = `1 1 ${Math.round(availableHeight / openItems.length)}%`;
            }
        });
    }

                // ── Tab visibility: reload on show ─
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) {
            refreshModels();      // re-read model status on focus
            if (logModel && logPollTimer === null) {
                loadLogSnapshot(logModel);
                startLogPoll(logModel);
            }
        }
    });

    // ── 3. Resize handles (vertical + horizontal) ───────────────
    const rootStyle = document.documentElement.style;
    const VKEY = 'arkestra-col-width';
    const HKEY = 'arkestra-log-ratio';

    // Restore saved ratios
    const sv = localStorage.getItem(VKEY);
    if (sv && !isNaN(sv)) rootStyle.setProperty('--col-width', sv + '%');
    const sh = localStorage.getItem(HKEY);
    if (sh && !isNaN(sh)) {
        const logsP = document.getElementById('pane-logs');
        const chatP = document.getElementById('pane-chat');
        if (logsP && chatP) {
            logsP.style.flex = '0 0 ' + sh + '%';
            chatP.style.flex = '0 0 ' + (100 - sh) + '%';
        }
    }

    // Vertical divider: left ↔ right
    const vDivider = document.getElementById('col-divider');
    vDivider?.addEventListener('mousedown', e => {
        isVDragging = true;
        vDivider.classList.add('active');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    // Horizontal divider: logs ↔ chat
    const hDivider = document.getElementById('row-divider');
    hDivider?.addEventListener('mousedown', e => {
        isHDragging = true;
        hDivider.classList.add('active');
        document.body.style.cursor = 'row-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    window.addEventListener('mousemove', e => {
        const layout = document.querySelector('.layout');
        if (!layout) return;
        const rect = layout.getBoundingClientRect();

        if (isVDragging) {
            const pct = Math.round(((e.clientX - rect.left) / rect.width) * 100);
            if (pct >= 20 && pct <= 80) {
                rootStyle.setProperty('--col-width', pct + '%');
                localStorage.setItem(VKEY, String(pct));
            }
        } else if (isHDragging) {
            const rightCol = document.getElementById('right-panel');
            if (!rightCol) return;
            const rRect = rightCol.getBoundingClientRect();
            const ratio = Math.round(((e.clientY - rRect.top) / rRect.height) * 100);
            if (ratio >= 15 && ratio <= 85) {
                const logsP = document.getElementById('pane-logs');
                const chatP = document.getElementById('pane-chat');
                if (logsP && chatP) {
                    logsP.style.flex = '0 0 ' + ratio + '%';
                    chatP.style.flex = '0 0 ' + (100 - ratio) + '%';
                }
                localStorage.setItem(HKEY, String(ratio));
            }
        }
    });

    window.addEventListener('mouseup', () => {
        if (isVDragging) {
            isVDragging = false;
            vDivider?.classList.remove('active');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        } else if (isHDragging) {
            isHDragging = false;
            hDivider?.classList.remove('active');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });

// ── 5. Logs pane — polling on model select ────────────────
    const logSelect = document.getElementById('log-model-select');
    if (logSelect) {
        logSelect.addEventListener('change', () => {
            logModel = logSelect.value;
            if (logModel) {
                loadLogSnapshot(logModel);
                startLogPoll(logModel);
            } else {
                stopLogPoll();
                const display = document.getElementById('log-display');
                if (display) {
                    display.textContent = '';
                    const ph = document.createElement('p');
                    ph.className = 'placeholder-text';
                    ph.textContent = 'Select a model to view logs.';
                    display.appendChild(ph);
                }
            }
        });
    }

    // ── 6. Chat pane wireups ────────────────────────────────
    const chatSelect = document.getElementById('chat-model-select');
    if (chatSelect) {
        chatSelect.addEventListener('change', () => {
            chatModel = chatSelect.value;
            // Always clear history and display when switching models
            const container = document.getElementById('chat-display');
            if (container) {
                container.textContent = '';
                chatHistory = [];
            }
            if (chatModel) {
                restoreChatParams();
            }
        });
    }

    // Auto-select first model if available
    if (!chatModel && modelsCache.length > 0 && chatSelect) {
        chatModel = modelsCache[0].id;
        chatSelect.value = chatModel;
        restoreChatParams();
    }

    document.getElementById('right-chat-params-toggle')?.addEventListener('click', toggleChatParams);
    document.getElementById('right-chat-form')?.addEventListener('submit', sendMessage);

    // Auto-persist params on any change
    ['rcp-temp', 'rcp-top-p', 'rcp-max-tokens'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', saveChatParams);
    });

    // Collapse all accordion items
    document.getElementById('btn-collapse-all')?.addEventListener('click', () => {
        document.querySelectorAll('#model-accordion-items .cluster-item:not(.collapsed), '
            + '#model-accordion-items .model-row.expanded').forEach(item => {
                item.classList.add('collapsed');
                if (item.classList.contains('model-row')) {
                    item.classList.remove('expanded');
                }
        });
    });
});
