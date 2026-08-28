// ── 1.0 Configuration ─────────────────────────────────────
const ADMIN_KEY    = document.querySelector('meta[name="arkestra-admin-key"]')?.content || '';
const BASE_URL     = window.location.origin;
const POLL_INTERVAL = 2000;            // ms — settable constant for model status polling

// ── 1.1 Global State ──────────────────────────────────────
let modelsCache      = [];          // from GET /admin/models
let clusterMap       = {};          // { name: { base_url, healthy } }
let modelStates      = {};          // { modelName: { config, snapshot, dirty, expanded } }
let logModel         = '';          // currently selected model for logs pane
let chatModel        = '';          // currently selected model for chat pane
let backendOptions   = {};           // backends dict from /admin/models response
let runnerTypes      = [];          // runner types from /admin/models response

// Log polling state
let logPollTimer    = null;         // interval id for log polling
let logLastSeq     = 0;            // server-provided sequence cursor for log polling

// ── 1.3 Fetch Helpers ─────────────────────────────────────
async function adminFetch(path, options = {}) {
    const headers = { ...options.headers };
    if (ADMIN_KEY) {
        headers['X-Admin-Key'] = ADMIN_KEY;
    }
    console.log('[adminFetch]', BASE_URL + path, 'key=' + (ADMIN_KEY ? '✓' : '(none)'));
    const opts = { method: 'GET', headers, ...options };
    // Remove Content-Type for GET (browser handles it)
    if (opts.method === 'GET') delete opts.headers['Content-Type'];

    const r = await fetch(`${BASE_URL}${path}`, opts);
    const body = await r.text();
    if (!r.ok) {
        let detail;
        try { detail = JSON.parse(body).detail || body; }
        catch { detail = body; }
        throw new Error(`[admin ${r.status}] ${detail}`);
    }
    return body ? JSON.parse(body) : null;
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

// ── 1.4 ID Sanitization ─────────────────────────────────────
function sanitizeId(name) {
    return name.replace(/[^a-zA-Z0-9_-]/g, '_');
}

// Parse <cluster>/<model-id> into [cluster, localId]; returns ['local', id] if no slash.
function parseClusterPrefix(id) {
    const slash = id.indexOf('/');
    if (slash === -1) return ['local', id];
    return [id.slice(0, slash), id.slice(slash + 1)];
}

// Strip 'runnerstate.' prefix and lowercase for CSS class.
function normalizeStatus(status) {
    if (typeof status === 'object' && status !== null) status = status.value || '';
    return (status || '').replace('runnerstate.', '').toLowerCase();
}

// HTML-escape string — handles &, ", <, >.
function escapeHTML(str) {
    const s = str || '';
    return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── 2. Model List — fetch + render accordion items ───────────
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
    const wasRowExpanded = new Set(
        Array.from(container.querySelectorAll('.model-row.expanded'))
            .map(el => el.dataset.model)
    );

    // Group models by cluster (strip prefix from model name)
    const modelsByCluster = {};
    for (const m of modelsCache) {
        const [cluster, localId] = parseClusterPrefix(m.id);
        m.localId = localId;
        if (!modelsByCluster[cluster]) modelsByCluster[cluster] = [];
        modelsByCluster[cluster].push(m);
    }

    // Build HTML: clusters → model rows
    let html = '';
    for (const [clusterName, clusterModels] of Object.entries(modelsByCluster)) {
        const clustCfg = clusterMap[clusterName] || { healthy: true };
        const healthClass = clustCfg.healthy ? '' : ' unhealthy';
        const stateKey = 'sec-cluster-' + sanitizeId(clusterName);
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
            const statusClass = m.status ? normalizeStatus(m.status) : 'uncached';
            const sid = sanitizeId(name);
            html += '<div class="model-row" id="sec-model-' + sid + '" data-model="' + name + '">' 
                + '  <div class="model-name-bar">'
                + '    <span class="status-dot ' + statusClass + '" id="dot-' + sid + '"></span>'
                + '    <span class="model-name">' + m.localId + '</span>'
                + '  </div>'
                + '  <div class="model-config-panel" id="body-' + sid + '">' 
                + '    <p class="placeholder-text">Click to load configuration…</p>' 
                + '  </div>' 
                + '</div>';
        }

        html += '  </div></div>';
    }
    container.innerHTML = html;

    // Restore expanded clusters and rows
    for (const id of wasExpanded) {
        const el = document.getElementById(id);
        if (el) el.classList.remove('collapsed');
    }
    for (const modelName of wasRowExpanded) {
        const row = container.querySelector('[data-model="' + escapeHTML(modelName) + '"]');
        if (row) {
            row.classList.add('expanded');
            populateModelPanel(modelName);
        }
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

        // Rebuild DOM only when models are added/removed or clusters change.
        // Status-only changes just update dots — avoids destroying config panels,
        // button listeners, and expanded state on every 2s poll.
        if (listChanged || !document.getElementById('model-accordion-items')) {
            buildAccordionItems();
        } else {
            // Update status dots and cluster health in-place — no flicker.
            for (const m of modelsCache) {
                const name = m.id;
                const dotId = 'dot-' + sanitizeId(name);
                const dot = document.getElementById(dotId);
                if (dot && m.status) {
                    const statusClass = normalizeStatus(m.status);
                    dot.className = 'status-dot ' + statusClass;
                }
            }
            // Update cluster health dots and model counts
            for (const [clusterName, items] of Object.entries(
                modelsCache.reduce((acc, m) => {
                    const [c] = parseClusterPrefix(m.id);
                    (acc[c] ||= []).push(m);
                    return acc;
                }, {})
            )) {
                const el = document.getElementById('sec-cluster-' + sanitizeId(clusterName));
                if (!el) continue;
                const healthDot = el.querySelector('.cluster-health-dot');
                const clusterCfg = clusterMap[clusterName] || { healthy: true };
                if (healthDot) {
                    healthDot.className = 'cluster-health-dot' + (clusterCfg.healthy ? '' : ' unhealthy');
                }
                const countEl = el.querySelector('.cluster-count');
                if (countEl) {
                    countEl.textContent = items.length + ' model' + (items.length !== 1 ? 's' : '');
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

// ── 3. Per-model config panel on expand ────────────────────
async function populateModelPanel(modelName) {
    // Find the model row by data-model attribute, then get its nested config panel
    const row = document.querySelector('.model-row[data-model="' + escapeHTML(modelName) + '"]');
    if (!row) return;
    const panel = row.querySelector('.model-config-panel');
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
        html += '<div class="edit-field"><label for="mc-checkpoint-' + sanitizeId(modelName) + '">Checkpoint</label>';
        html += '  <input type="text" id="mc-checkpoint-' + sanitizeId(modelName) + '" value="' + escapeHTML(cfg.checkpoint || '') + '"></div>';

        // Backend select
        const backendSelId = 'mc-backend-' + sanitizeId(modelName);
        html += '<div class="edit-field"><label for="' + backendSelId + '">Backend</label>';
        html += '  <select id="' + backendSelId + '">';
        html += '    <option value="">(default)</option>';
        if (backendOptions) {
            for (const [key, val] of Object.entries(backendOptions)) {
                const label = (typeof val === 'object') ? (val.host || key) : key;
                html += '    <option value="' + escapeHTML(key) + '"' + ((cfg.backend === key) ? ' selected' : '') + '>' + escapeHTML(label) + '</option>';
            }
        }
        html += '  </select></div>';

        // Runner select
        const runnerSelId = 'mc-runner-' + sanitizeId(modelName);
        html += '<div class="edit-field"><label for="' + runnerSelId + '">Runner</label>';
        html += '  <select id="' + runnerSelId + '">';
        html += '    <option value="">(auto)</option>';
        if (runnerTypes) {
            for (const t of runnerTypes) {
                html += '    <option value="' + escapeHTML(t) + '"' + ((cfg.runner === t) ? ' selected' : '') + '>' + escapeHTML(t) + '</option>';
            }
        }
        html += '  </select></div>';

        // Args textarea
        html += '<div class="edit-field"><label for="mc-args-' + sanitizeId(modelName) + '">Args</label>';
        html += '  <textarea id="mc-args-' + sanitizeId(modelName) + '" rows="2">' + escapeHTML(typeof cfg.args === 'object' ? JSON.stringify(cfg.args, null, 2) : (cfg.args || '')) + '</textarea></div>';

        // Capabilities chips
        const capsId = 'mc-caps-' + sanitizeId(modelName);
        html += '<div class="edit-field"><label>Capabilities</label><div class="capabilities-list" id="' + capsId + '">';
        const selectedCaps = new Set(cfg.capabilities || allCaps);
        for (const c of allCaps) {
            html += '  <span class="capability-chip' + (selectedCaps.has(c) ? ' active' : '') + '" data-cap="' + c + '">' + c + '</span>';
        }
        html += '</div></div>';

        // Log buffer size
        html += '<div class="edit-field"><label for="mc-log-buffer-' + sanitizeId(modelName) + '">Log Buffer Size</label>';
        html += '  <input type="number" id="mc-log-buffer-' + sanitizeId(modelName) + '" min="50" max="10000" step="50" value="' + (cfg.max_log_lines || 500) + '"></div>';

        // Action buttons
        html += '<div class="model-actions">';
        const btnId = (id) => 'mc-' + id + '-' + sanitizeId(modelName);
        html += '  <button type="button" title="Reset config to server value" data-model="' + escapeHTML(modelName) + '" data-action="reset" id="' + btnId('reset') + '">↺</button>';
        html += '  <button type="button" title="Stop model runner" data-model="' + escapeHTML(modelName) + '" data-action="stop" id="' + btnId('stop') + '" class="btn-danger">⏹</button>';
        html += '  <button type="button" title="Start/Restart with current config" data-model="' + escapeHTML(modelName) + '" data-action="start" id="' + btnId('start') + '" class="btn-success">▶</button>';
        html += '  <button type="button" title="Save config changes" data-model="' + escapeHTML(modelName) + '" data-action="save" id="' + btnId('save') + '">✓</button>';
        html += '  <button type="button" title="Eject model (stop + delete cache)" data-model="' + escapeHTML(modelName) + '" data-action="eject" id="' + btnId('eject') + '" class="btn-danger">⏏</button>';
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

        // Wire up action buttons — stopPropagation to prevent bubbling
        // to .model-row click handler which would toggle accordion state
        panel.querySelector('[data-action="save"]')?.addEventListener('click', e => { e.stopPropagation(); saveModelConfig(modelName); });
        panel.querySelector('[data-action="reset"]')?.addEventListener('click', e => { e.stopPropagation(); resetModelConfig(modelName); });
        panel.querySelector('[data-action="start"]')?.addEventListener('click', e => { e.stopPropagation(); startModel(modelName); });
        panel.querySelector('[data-action="stop"]')?.addEventListener('click', e => { e.stopPropagation(); stopModel(modelName); });
        panel.querySelector('[data-action="eject"]')?.addEventListener('click', e => { e.stopPropagation(); ejectModel(modelName); });

    } catch (e) {
        console.error('[admin] failed to load config for ' + modelName + ':', e.message);
        panel.innerHTML = '<p class="placeholder-text" style="color:var(--red)">Failed to load config: ' + e.message + '</p>';
    }
}

// Helper: get values from a model config panel
function getModelFormValues(modelName) {
    const prefix = sanitizeId(modelName);
    return {
        checkpoint: document.getElementById('mc-checkpoint-' + prefix)?.value || '',
        backend: (document.getElementById('mc-backend-' + prefix)?.value) || undefined,
        runner: (document.getElementById('mc-runner-' + prefix)?.value) || undefined,
        args: (document.getElementById('mc-args-' + prefix)?.value) || undefined,
        max_log_lines: parseInt(document.getElementById('mc-log-buffer-' + prefix)?.value) || 500,
    };
}

function getModelCapabilities(modelName) {
    const prefix = sanitizeId(modelName);
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
    const btn = document.getElementById('mc-save-' + sanitizeId(modelName));
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
        const btn = document.getElementById('mc-save-' + sanitizeId(modelName));
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

// ── 4.5 Model Actions ────────────────────────────────────────
async function modelAction(modelName, action) {
    const btn = document.getElementById('mc-' + action + '-' + sanitizeId(modelName));
    if (!btn) return;

    const origDisabled = btn.disabled;
    btn.disabled = true;

    try {
        let body = {};
        if (action === 'start') {
            body = { ...getModelFormValues(modelName), capabilities: getModelCapabilities(modelName) };
        }
        await adminPost('/admin/' + action + '/' + encodeURIComponent(modelName), body);
        refreshModels();
    } catch (e) {
        console.error('[admin] ' + action + ' failed for ' + modelName + ':', e.message);
    } finally {
        btn.disabled = origDisabled;
    }
}

async function startModel(modelName) { await modelAction(modelName, 'start'); }
async function stopModel(modelName)  { await modelAction(modelName, 'stop'); }
async function ejectModel(modelName) { await modelAction(modelName, 'eject'); }

// ── 4. Model Selection ────────────────────────────────────
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

// ── 5. Logs Pane ───────────────────────────────────────────
function logUrl(modelName) {
    return modelName === '*' ? '/admin/logs' : '/admin/log/' + encodeURIComponent(modelName);
}

function createLogLine(text) {
    const div = document.createElement('div');
    div.className = 'log-line';
    div.textContent = text;
    return div;
}

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
        const data = await adminGet(logUrl(modelName), { lines: 200 });
        const lines = data?.lines ?? [];
        logLastSeq = data.since ?? data.seq ?? 0;

        for (const entry of lines) {
            display.appendChild(createLogLine(entry.text));
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
        display.appendChild(createLogLine(line));
    });

    logLastSeq += lines.length;

    // Auto-scroll only if user is already at bottom
    if (nearBottom) display.scrollTop = display.scrollHeight;
}

function startLogPoll(modelName) {
    stopLogPoll();
    const pollOne = async () => {
        try {
            const data = await adminGet(logUrl(modelName), {
                lines: 200,
                ...(logLastSeq > 0 ? { since: logLastSeq } : {}),
            });
            if (!data || !data.lines) return;
            const newEntries = (logLastSeq === 0) ? data.lines.map(e => e.text) : data.lines;
            if (newEntries.length > 0) appendLogLines(newEntries);
            logLastSeq = data.since ?? data.seq ?? logLastSeq;
        } catch (e) {
            // Silent fail — don't flood console on transient issues
        }
    };
    pollOne();          // fetch immediately
    logPollTimer = setInterval(pollOne, POLL_INTERVAL);
}

// ── 6. Chat Pane Functions ─────────────────────────────────
let chatHistory      = [];   // full conversation history for selected model
let chatAbortCtrl    = null; // AbortController for active chat stream
const CHAT_PARAMS_KEY = 'arkestra-chat-params';
let isChatStreaming  = false;       // true while a chat response is streaming
let _streamAcc      = '';           // accumulated raw text of current stream
let _streamBubble   = null;         // DOM element receiving the stream

// 6.1 Parameters Panel Toggle
function toggleChatParams() {
    const panel = document.getElementById('chat-params-panel');
    const toggle = document.getElementById('right-chat-params-toggle');
    if (!panel || !toggle) return;
    panel.classList.toggle('open');
    const isOpen = panel.classList.contains('open');
    toggle.textContent = isOpen ? '⚙ Parameters ▴' : '⚙ Parameters ▾';
    requestAnimationFrame(() => layoutRightPanels());
}

// 6.2 Param Persistence
const CHAT_PARAM_FIELDS = [
    { id: 'rcp-temp',     store: 'temperature', default: 0.7,  parse: v => parseFloat(v) || 0.7 },
    { id: 'rcp-top-p',    store: 'top_p',       default: 0.95, parse: v => parseFloat(v) || 0.95 },
    { id: 'rcp-max-tokens', store: 'max_tokens', default: 512,  parse: v => parseInt(v) || 512 },
];

function getChatParams() {
    const stored = localStorage.getItem(CHAT_PARAMS_KEY);
    if (stored) { try { return JSON.parse(stored); } catch { /* ignore */ } }
    return {};
}

function saveChatParams() {
    if (!chatModel) return;
    const params = getChatParams();
    params[chatModel] = Object.fromEntries(
        CHAT_PARAM_FIELDS.map(f => [f.store, f.parse(document.getElementById(f.id)?.value)])
    );
    localStorage.setItem(CHAT_PARAMS_KEY, JSON.stringify(params));
}

function restoreChatParams() {
    if (!chatModel) return;
    const params = getChatParams()[chatModel];
    if (!params) return;
    for (const f of CHAT_PARAM_FIELDS) {
        document.getElementById(f.id).value = String(params[f.store] ?? f.default);
    }
}

// 6.3 Message Bubble Helpers
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

// 6.4 Send Message (form submit handler)
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

// 6.5 Streaming Response Handler
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

// 6.6 Markdown Rendering with marked.js Fallback
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

// ── Lifecycle — DOMContentLoaded ───────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    refreshModels();
    // Poll for model status updates at configurable interval
    setInterval(refreshModels, POLL_INTERVAL);

    // ── Accordion toggle — shared handler for cluster and model rows ───────────
    window.toggleAccordion = function(id) {
        const el = document.getElementById(id);
        if (!el) return;
        const wasCollapsed = el.classList.contains('collapsed');
        el.classList.toggle('collapsed');

        // Model row expanded → show config panel + load config
        if (el.classList.contains('model-row')) {
            const wasExpanded = el.classList.contains('expanded');
            el.classList.toggle('expanded', !wasCollapsed);
            if (!wasExpanded) {
                const modelName = el.dataset.model;
                if (modelName) populateModelPanel(modelName);
            }
        }

        requestAnimationFrame(() => layoutRightPanels());
    };

    // Delegated click handler for accordion items
    function handleAccordionClick(e) {
        const clusterHeader = e.target.closest('.cluster-header');
        if (clusterHeader) {
            toggleAccordion(clusterHeader.closest('.cluster-item').id);
            return;
        }
        const modelRow = e.target.closest('.model-row[data-model]');
        if (modelRow) {
            const parentCluster = modelRow.closest('.cluster-item');
            if (parentCluster?.classList.contains('collapsed')) {
                toggleAccordion(parentCluster.id);
            }
            toggleAccordion(modelRow.id);
            return;
        }
    }

    document.getElementById('model-accordion-items')?.addEventListener('click', handleAccordionClick);

    // Right-panel height splitting — each open pane gets equal share.
    function layoutRightPanels() {
        const accordion = document.getElementById('right-accordion');
        if (!accordion) return;
        const items = Array.from(accordion.querySelectorAll('.accordion-item'));
        const openItems = items.filter(i => !i.classList.contains('collapsed'));

        if (openItems.length === 0) {
            accordion.style.height = '100%';
            return;
        }

        const share = Math.round(100 / openItems.length);
        items.forEach(item => {
            if (item.classList.contains('collapsed')) {
                item.style.flex = '0 0 auto';
            } else {
                item.style.flex = `1 1 ${share}%`;
            }
        });
    }

                // ── Tab visibility — reload on focus ─
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) {
            refreshModels();      // re-read model status on focus
            if (logModel && logPollTimer === null) {
                loadLogSnapshot(logModel);
                startLogPoll(logModel);
            }
        }
    });

    // ── Resize handles (vertical + horizontal) — single drag manager ──
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

    // Drag state: null | { axis: 'v'|'h', divider, pct: number }
    let drag = null;

    function handleDragStart(e) {
        const vDiv = document.getElementById('col-divider');
        const hDiv = document.getElementById('row-divider');
        if (e.target === vDiv || vDiv?.contains(e.target)) {
            drag = { axis: 'v', divider: vDiv };
        } else if (e.target === hDiv || hDiv?.contains(e.target)) {
            drag = { axis: 'h', divider: hDiv };
        }
        if (!drag) return;
        e.preventDefault();
        document.body.style.userSelect = 'none';
    }

    function handleDragMove(e) {
        if (!drag) return;
        if (drag.axis === 'v') {
            const layout = document.querySelector('.layout');
            const rect = layout.getBoundingClientRect();
            const pct = Math.round(((e.clientX - rect.left) / rect.width) * 100);
            if (pct >= 20 && pct <= 80) rootStyle.setProperty('--col-width', pct + '%');
        } else {
            const rightCol = document.getElementById('right-panel');
            const rRect = rightCol.getBoundingClientRect();
            const ratio = Math.round(((e.clientY - rRect.top) / rRect.height) * 100);
            if (ratio >= 15 && ratio <= 85) {
                const logsP = document.getElementById('pane-logs');
                const chatP = document.getElementById('pane-chat');
                if (logsP && chatP) {
                    logsP.style.flex = '0 0 ' + ratio + '%';
                    chatP.style.flex = '0 0 ' + (100 - ratio) + '%';
                }
            }
        }
    }

    function handleDragEnd() {
        if (!drag) return;
        drag.divider.classList.remove('active');
        document.body.style.userSelect = '';
        drag = null;
    }

    document.getElementById('col-divider')?.addEventListener('mousedown', handleDragStart);
    document.getElementById('row-divider')?.addEventListener('mousedown', handleDragStart);
    window.addEventListener('mousemove', handleDragMove);
    window.addEventListener('mouseup', handleDragEnd);

// ── 4.6 Logs pane — polling on model select ────────────────
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
    CHAT_PARAM_FIELDS.forEach(f => {
        const el = document.getElementById(f.id);
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
