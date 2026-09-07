/*
 * widget.js — JSON-driven layout engine for ModelArkestra admin.
 * Pattern: data (JSON tree) -> render() -> DOM. Conventions wire events.
 */

const BASE_URL = (window.BASE_URL || "").replace(/^\/+|\/+$/g, '');

// ── Auth mode detection ───────────────────────────────────────
const HAS_ADMIN_KEY = !!document.querySelector('meta[name="arkestra-admin-key"]')?.content;

// ── Pub/sub bus ───────────────────────────────────────────────
const EventBus = {
    _h: new Map(),
    on(e, fn)  { const l = this._h.get(e)||[]; l.push(fn); this._h.set(e,l); return () => this.off(e,fn); },
    off(e, fn) { const l = this._h.get(e); if(l){const i=l.indexOf(fn);if(i>=0)l.splice(i,1);} },
    emit(e, d)  { for(const fn of this._h.get(e)||[]) fn(d); },
};

// ── Config constants ───────────────────────────────────────────
const CFG = {
    STORAGE_CHAT_PARAMS: 'arkestra-chat-params',
    DEFAULT_TTS:        'default-tts',
    DEFAULT_WHISPER:    'default-whisper',
    POLL_INTERVAL:      2000,
};

// ── Shared utilities ───────────────────────────────────────────
function esc(s)  { return (s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function sanitizeId(n) { return (n||'').replace(/[^a-zA-Z0-9_-]/g, '_'); }
function normalizeStatus(s) { return (s?.value || s || '').replace('runnerstate.','').toLowerCase(); }
function keyToLabel(key) { return key.replace(/[-_](.)/g, (_, c) => c.toUpperCase()); }

// ── Session Registry — pinned session management ───────────────
const SessionRegistry = {
    _sessions: [],   // [{ id, type:'chat'|'log', modelId, cluster, history[], abortCtrl }]
    _nextId: 0,

    add(type, modelId) {
        const id = 's-' + (++this._nextId);
        this._sessions.push({ id, type, modelId, history:[], abortCtrl:null });
        EventBus.emit('session.add', { id, type, modelId });
        return id;
    },

    remove(id) {
        const i = this._sessions.findIndex(s => s.id === id);
        if (i < 0) return null;
        const session = this._sessions.splice(i, 1)[0];
        session.abortCtrl?.abort();
        EventBus.emit('session.remove', session);
        return session;
    },

    get(id) { return this._sessions.find(s => s.id === id); },
    list()  { return [...this._sessions]; },
};

// ── Personal defaults sync store ───────────────────────────────
const SyncStore = {
    _cache: {},  // { modelId → { temperature, max_tokens, top_p, top_k } }

    async load(modelId) {
        const stored = await window.arkestraDB.getSettings(modelId);
        if (stored && stored.chat_params) this._cache[modelId] = stored.chat_params;
        else this._cache[modelId] = {};
        return this._cache[modelId];
    },

    async save(modelId) {
        await window.arkestraDB.saveSettings(modelId, { chat_params: this._cache[modelId] || {} });
    },

    get(modelId, key) { return (this._cache[modelId] || {})[key]; },
    set(modelId, key, value) { if (!this._cache[modelId]) this._cache[modelId] = {}; this._cache[modelId][key] = value; },
    async persist(modelId) { await this.save(modelId); }
};

// ── Button map ─────────────────────────────────────────────────
const BTN_MAP = { start:'▶', stop:'■', eject:'⏏', cancel:'✕', save:'✓', reset:'↺' };

function escapeSelector(id) { return id.replace(/["'#%&*,/:<=>?@[\\]^`{|}~]/g, '\\$&'); }

// ── render() — walks JSON tree -> DOM ─────────────────────────
function render(node) {
    if (!node?.widget) return null;
    const key = node.widget.charAt(0).toUpperCase() + node.widget.slice(1);
    return renderers[key]?.(node) ?? null;
}

const renderers = {};

// ═══════════════════════════════════════════════════════════
// Layout containers (preserved infrastructure — DO NOT REMOVE)
// ═══════════════════════════════════════════════════════════

renderers.SplitPane = function({ axis, ratio, children }) {
    const el = document.createElement('div');
    el.style.display = 'flex';
    el.style.overflow = 'hidden';
    if (axis === 'v') el.style.flexDirection = 'column';

    for (let i = 0; i < children.length; i++) {
        const c = render(children[i]);
        if (!c) continue;
        const last = i === children.length - 1;
        c.style.minHeight = axis === 'v' ? '0' : '';
        c.style.minWidth = axis === 'h' ? '0' : '';
        if (ratio !== undefined && !last) {
            c.style.flex = `0 0 ${ratio}%`;
            el.appendChild(c);
            const div = document.createElement('div');
            div.className = axis === 'h' ? 'divider-v' : 'divider-h';
            el.appendChild(div);
        } else {
            c.style.flex = '1';
            el.appendChild(c);
        }
    }

    setTimeout(() => {
        for (const d of el.querySelectorAll(':scope > .divider-v, :scope > .divider-h')) {
            if (d.dataset.dragWired) continue;
            d.dataset.dragWired = '1';
            const isH = d.classList.contains('divider-v');
            let dragging = false;
            const onMove = (ev) => {
                if (!dragging) return;
                const rect = d.parentElement.getBoundingClientRect();
                const val = isH ? ((ev.clientX-rect.left)/rect.width)*100 : ((ev.clientY-rect.top)/rect.height)*100;
                if (val < 20 || val > 80) return;
                const p = d.previousElementSibling, n = d.nextElementSibling;
                if (p && n) { p.style.flex=`0 0 ${val}%`; n.style.flex=`0 0 ${100-val}%`; }
                localStorage.setItem('arkestra-layout-' + (isH?'h':'v'), Math.round(val));
            };
            const onUp = () => { dragging=false; d.classList.remove('active'); document.body.style.userSelect=''; window.removeEventListener('mousemove',onMove); window.removeEventListener('mouseup',onUp); document.removeEventListener('mouseup',onUpGlobal); };
            const onUpGlobal = () => { if (dragging) onUp(); };
            d.addEventListener('mousedown', (e) => { e.preventDefault(); document.body.style.userSelect='none'; d.classList.add('active'); dragging=true; window.addEventListener('mousemove',onMove); window.addEventListener('mouseup',onUp); document.addEventListener('mouseup',onUpGlobal); });
        }
    }, 0);

    el.style.flex = '1'; el.style.minHeight = '0';
    return el;
};

renderers.AccordionContainer = function() {
    const el = document.createElement('div');
    el.className = 'accordion'; el.id = 'left-accordion';
    const header = document.createElement('h3');
    header.textContent = 'Model Cluster';
    el.appendChild(header);
    const body = document.createElement('div');
    body.className = 'acc-body'; body.id = 'model-accordion-items';
    el.appendChild(body);
    let collapsed = false;
    header.addEventListener('click', () => { collapsed=!collapsed; body.style.display=collapsed?'none':''; });
    return el;
};

renderers.AccGroup = function({ title, group }) {
    const el = document.createElement('div');
    el.className = 'acc-group'; el.dataset.group = group;
    const header = document.createElement('h3');
    header.className = 'acc-group-header';
    header.innerHTML = `<span>${esc(title)}</span><span class="acc-group-count"></span><span></span>`;
    const body = document.createElement('div');
    body.className = 'acc-body'; body.id = 'models-'+group+'-items';
    el.appendChild(header); el.appendChild(body);
    let collapsed = false;
    header.addEventListener('click', () => { collapsed=!collapsed; body.style.display=collapsed?'none':''; });
    return { element:el, body:body, header:header };
};

// ═══════════════════════════════════════════════════════════
// New layout: SessionLayout (single top-level widget)
// ═══════════════════════════════════════════════════════════

renderers.SessionLayout = function({ axis='v', ratio=30 }) {
    const el = document.createElement('div');
    el.style.display = 'flex'; el.style.overflow = 'hidden';
    el.style.flex = '1'; el.style.minHeight = '0';

    const stored = localStorage.getItem('arkestra-layout-v') ?? localStorage.getItem('arkestra-layout-h');
    const finalAxis = stored ? (parseInt(stored) > 25 ? 'v' : 'h') : axis;

    const split = renderers.SplitPane({ axis: finalAxis, ratio, children: [
        { widget:'ClusterTree', cluster: true },
        { widget:'SessionDock' }
    ]});

    el.appendChild(split);
    return el;
};

// ═══════════════════════════════════════════════════════════
// Cluster tree — navigational accordion (top/left pane)
// Structure: cluster → model
// ═══════════════════════════════════════════════════════════

renderers.ClusterTree = function({ cluster }) {
    const el = document.createElement('div');
    el.className = 'cluster-tree';

    // Local cluster entry — always first
    const localGroup = _accNode('Local');
    localGroup.header.innerHTML += '<span class="cluster-health-dot"></span>';
    el.appendChild(localGroup.el);

    // Remote clusters — populated from data
    const remoteContainer = document.createElement('div');
    remoteContainer.className = 'remote-clusters';
    el.appendChild(remoteContainer);

    // Expose refs for app.js
    el._localModelBody = localGroup.body;
    el._remoteContainer = remoteContainer;

    return el;
};

// Internal: create an accordion node (cluster or model level)
function _accNode(label) {
    const el = document.createElement('div');
    el.className = 'acc-node';
    const header = document.createElement('h3');
    header.className = 'acc-node-header';
    header.innerHTML = `<span>${esc(label)}</span><span class="acc-arrow">▸</span>`;
    const body = document.createElement('div');
    body.className = 'acc-body';
    el.appendChild(header); el.appendChild(body);

    let collapsed = false;
    header.addEventListener('click', (e) => {
        if (e.target.closest('[data-action]')) return;
        collapsed = !collapsed;
        body.style.display = collapsed ? 'none' : '';
        header.querySelector('.acc-arrow').textContent = collapsed ? '▸' : '▾';
    });

    return { el, header, body };
}
renderers._accNode = _accNode;

// ═══════════════════════════════════════════════════════════
// Docked sessions — pinned workspace (bottom/right pane)
// ═══════════════════════════════════════════════════════════

renderers.SessionDock = function() {
    const el = document.createElement('div');
    el.className = 'session-dock';
    el.style.display = 'flex'; el.style.flexDirection = 'column';
    el.style.overflow = 'hidden';

    const bar = document.createElement('div');
    bar.className = 'dock-header';
    bar.innerHTML = '<span class="dock-title">Sessions</span><span class="dock-count"></span>';
    if (HAS_ADMIN_KEY) {
        const toggle = document.createElement('button');
        toggle.className = 'layout-toggle';
        toggle.title = 'Toggle layout';
        toggle.textContent = '⇔';
        bar.appendChild(toggle);
    }
    el.appendChild(bar);

    const container = document.createElement('div');
    container.className = 'docked-sessions';
    container.style.flex = '1'; container.style.overflowY = 'auto';
    el.appendChild(container);

    const empty = document.createElement('div');
    empty.className = 'dock-empty';
    empty.textContent = 'Select a model above and click + to open a chat or log session.';
    container.appendChild(empty);

    // Expose for app.js to populate
    el._sessionContainer = container;
    el._emptyHint = empty;

    bar.querySelector('.layout-toggle')?.addEventListener('click', () => {
        const currentH = localStorage.getItem('arkestra-layout-h');
        if (currentH) localStorage.removeItem('arkestra-layout-h');
        else localStorage.setItem('arkestra-layout-h', '40');
        location.reload();
    });

    EventBus.on('session.add', ({ id }) => { dockedSessionId = id; updateDock(); });
    EventBus.on('session.remove', () => updateDock());

    let dockedSessionId = null;

    function updateDock() {
        const sessions = SessionRegistry.list();
        empty.classList.toggle('hidden', sessions.length > 0);
        document.querySelector('.dock-count').textContent = ' (' + sessions.length + ')';
    }

    return el;
};

// ═══════════════════════════════════════════════════════════
// Pane components (preserved — used as session children)
// ═══════════════════════════════════════════════════════════

renderers.LogPane = function() {
    const el = document.createElement('div');
    el.className = 'pane pane-logs';
    const header = document.createElement('div');
    header.className = 'pane-header';
    header.innerHTML = '<span class="pane-title">Log</span><label>Select Model:</label><select id="log-model-select"></select>';
    el.appendChild(header);
    const display = document.createElement('pre');
    display.className = 'log-container'; display.id = 'log-display';
    el.appendChild(display);
    return el;
};

renderers.ChatPane = function({ sessionId, modelId }) {
    const sid = sessionId || '';
    const el = document.createElement('div');
    el.className = 'pane pane-chat';
    if (sessionId) el.dataset.sessionId = sessionId;

    // Header
    const header = document.createElement('div');
    header.className = 'pane-header';
    header.innerHTML = `<span class="pane-title">Chat</span>` +
        `<select id="chat-model-select-${sid}"></select>` +
        `<span class="chat-params-toggle" data-session="${sid}">Params ▸</span>` +
        `<span class="chat-tts-toggle" title="Text-to-Speech on/off">TTS: <span class="tts-status-${sid}">Off</span></span>`;
    el.appendChild(header);

    // Messages area
    const messages = document.createElement('div');
    messages.className = 'chat-messages'; messages.id = 'chat-display-' + sid;
    el.appendChild(messages);

    // Audio playback bar
    const audioBar = document.createElement('div');
    audioBar.className = 'audio-playback-bar hidden';
    audioBar.innerHTML = '<span class="audio-label">🔊</span>' +
        '<input type="range" min="0" max="100" value="0" step="0.1">' +
        '<span class="audio-time">0:00</span> / <span class="audio-time">0:00</span>' +
        '<button title="Pause/Resume">⏸</button><button title="Stop">⏹</button>';
    el.appendChild(audioBar);

    // Params panel (inline, hideable)
    const paramsPanel = document.createElement('div');
    paramsPanel.className = 'chat-params-panel';
    paramsPanel.id = 'chat-params-' + sid;
    paramsPanel.innerHTML = `<div class="chat-params-grid">` +
        `<div class="chat-param"><label>Temp</label><input type="number" id="f-chat-temp-${sid}" min="0" max="2" step="0.05" value="0.7"></div>` +
        `<div class="chat-param"><label>Max Tokens</label><input type="number" id="f-chat-max-tokens-${sid}" min="1" max="8192" step="1" value="4096"></div>` +
        `<div class="chat-param"><label>Top-P</label><input type="number" id="f-chat-top-p-${sid}" min="0" max="1" step="0.05" value="0.95"></div>` +
        `<div class="chat-param"><label>Top-K</label><input type="number" id="f-chat-top-k-${sid}" min="1" max="256" step="1" value="40"></div>` +
        `</div>`;
    el.appendChild(paramsPanel);

    // Input bar
    const inputBar = document.createElement('div');
    inputBar.className = 'chat-input-bar';
    inputBar.innerHTML = `<textarea id="f-chat-input-${sid}" rows="3" placeholder="Type a message..."></textarea>` +
        '<button title="Speak (TTS)">🔊</button>' +
        '<button title="Send">Send</button>' +
        `<span class="chat-status" id="chat-status-${sid}"></span>`;
    el.appendChild(inputBar);

    // ── Wire ChatPane internals (per-session) ───────────────
    let ttsActive = false;
    header.querySelector('.chat-tts-toggle')?.addEventListener('click', () => {
        ttsActive = !ttsActive;
        const s = header.querySelector('.tts-status-' + sid);
        if (s) s.textContent = ttsActive ? 'On' : 'Off';
        header.querySelector('.chat-tts-toggle').style.color = ttsActive ? 'var(--green)' : '';
    });

    // Params toggle
    header.querySelector('.chat-params-toggle')?.addEventListener('click', () => {
        paramsPanel.classList.toggle('open');
        const arrow = header.querySelector('.chat-params-toggle');
        arrow.textContent = paramsPanel.classList.contains('open') ? 'Params ▾' : 'Params ▸';
    });

    // TTS button
    inputBar.querySelector('button[title="Speak"]')?.addEventListener('click', () => {
        const textEl = document.getElementById('f-chat-input-' + sid);
        window._audio?.sendTTS(textEl?.value?.trim() || '');
    });

    // Send chat — wired dynamically via app.js with session context
    inputBar.querySelector('button[title="Send"]')?.addEventListener('click', () => {
        const sel = document.getElementById('chat-model-select-' + sid);
        const textEl = document.getElementById('f-chat-input-' + sid);
        if (sel?.value && textEl?.value.trim()) window._doSendChat(sel.value, sessionId, textEl);
    });

    // Params live change
    paramsPanel.addEventListener('input', (e) => {
        if (!e.target.id?.startsWith('f-chat-')) return;
        const name = e.target.id.replace('f-chat-' + sid + '-', '');
        const apiName = { temp:'temperature', 'max-tokens':'max_tokens', 'top-p':'top_p', 'top-k':'top_k' }[name] || name;
        const val = e.target.type === 'number' ? Number(e.target.value) : e.target.value;
        SyncStore.set(modelId, apiName, val);
    });

    // Enter key in textarea sends chat
    inputBar.querySelector('textarea')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            inputBar.querySelector('button[title="Send"]').click();
        }
    });

    return el;
};

// ═══════════════════════════════════════════════════════════
// Model row — simple clickable item with action buttons
// ═══════════════════════════════════════════════════════════

function formatSizeGB(sizeGb) {
    if (!sizeGb || sizeGb <= 0) return '';
    return ' ' + (sizeGb >= 1 ? sizeGb.toFixed(1) + 'GB' : Math.round(sizeGb * 1024) + 'MB');
}

function renderModelRow(model, isConfigView) {
    const name = model.id.includes('/') ? model.id.split('/').pop() : model.id;
    const statusClass = normalizeStatus(model.status);
    const sizeStr = formatSizeGB(model.size_gb);

    const row = document.createElement('div');
    row.className = 'model-row';
    row.dataset.model = model.id;

    // Session spawn buttons (always visible)
    let spawnBtns = `<button type="button" data-action="spawn-chat" data-model="${esc(model.id)}" title="New Chat">+</button>`;
    if (HAS_ADMIN_KEY && !isConfigView) {
        spawnBtns += `<button type="button" data-action="spawn-log" data-model="${esc(model.id)}" title="New Log">+</button>`;
    }

    // Inline action buttons — gated by auth
    let btns = '';
    if (HAS_ADMIN_KEY) {
        const adminBtns = ['start','stop','eject'];
        if (isConfigView) adminBtns.push('save','reset','cancel');
        for (const a of adminBtns) {
            btns += `<button type="button" data-action="${a}" data-model="${esc(model.id)}" title="${a}">${BTN_MAP[a]}</button>`;
        }
    }

    row.innerHTML = `<div class="model-name-bar"><span class="status-dot ${statusClass}"></span>` +
        `<span class="model-name">${esc(name)}</span>` +
        (sizeStr ? `<span class="model-size">${esc(sizeStr)}</span>` : '') +
        (isConfigView && model.runner_type ? '<span style="font-size:10px;color:var(--text-dim);margin-left:auto;flex-shrink:0;">('+esc(model.runner_type)+')</span>' : '') +
        `<div class="model-actions-inline">${spawnBtns}${btns}</div></div>`;

    // Pull progress bar (admin only)
    if (isConfigView && HAS_ADMIN_KEY) {
        const prog = document.createElement('div');
        prog.className = 'pull-progress hidden';
        prog.innerHTML = '<span class="pull-status"></span><span class="pull-pct"></span>';
        row.appendChild(prog);
    }

    return row;
}

// ── Helpers: pull progress, button sync, dirty detection ────────

function syncRowButtons(row, info) {
    const btns = row.querySelectorAll('.model-actions-inline button[data-action]:not([data-action^="spawn"])');
    if (!btns.length) return;
    const state = info?.state || 'stopped';
    for (const btn of btns) {
        const action = btn.dataset.action;
        switch(state) {
            case 'downloading':
                if (action === 'eject') { btn.disabled=true; }
                else if (!['start','stop'].includes(action)) { btn.style.display='none'; }
                break;
            case 'running':
                if (action==='start'||action==='eject') btn.disabled=true;
                break;
            case 'stopped': case 'error':
                if (action==='stop') btn.disabled=true;
                break;
            case 'uncached':
                if (action==='start') btn.disabled=true;
                else if (!['start','stop','eject'].includes(action)) { btn.style.display='none'; }
                break;
        }
        if ((action==='stop'||action==='eject') && !btn.disabled) btn.className='btn-danger';
        else if (action==='start' && !btn.disabled) btn.className='btn-success';
        else btn.className='';
    }
}

// ── Pull helpers (admin only) ──────────────────────────────────
let _pullProgressTimers = {};

async function startPull(modelId) {
    const row = document.querySelector('.model-row[data-model="'+escapeSelector(modelId)+'"]');
    if (!row) return;
    try {
        await window.apiRequest('/admin/pull/'+encodeURIComponent(modelId), {}, 'POST', {});
        syncRowButtons(row, { state:'downloading', statusClass:'loading' });
        const progEl = row.querySelector('.pull-progress');
        if (progEl) progEl.classList.remove('hidden');
        pollPullProgress(modelId);
    } catch(e) { console.error('[widget] pull failed:',e.message); }
}

async function cancelPull(modelId) {
    const row = document.querySelector('.model-row[data-model="'+escapeSelector(modelId)+'"]');
    if (!row) return;
    try {
        await window.apiRequest('/admin/cancel-pull/'+encodeURIComponent(modelId), {}, 'POST', {});
        syncRowButtons(row, { state:'uncached', statusClass:'uncached' });
        const progEl = row.querySelector('.pull-progress');
        if (progEl) progEl.classList.add('hidden');
        stopPullProgress(modelId);
    } catch(e) { console.error('[widget] cancel failed:',e.message); }
}

function pollPullProgress(modelId) {
    const row = document.querySelector('.model-row[data-model="'+escapeSelector(modelId)+'"]');
    if (!row) return;

    const poll = async () => {
        try {
            const data = await window.apiRequest('/admin/log/'+encodeURIComponent(modelId), { lines:10 });
            if (!data?.lines?.length) return;
            const progEl = row.querySelector('.pull-progress');
            const statusEl = progEl?.querySelector('.pull-status');
            const pctEl = progEl?.querySelector('.pull-pct');
            for (let i=data.lines.length-1; i>=0; i--) {
                const line = data.lines[i].text;
                if (!line.includes('[pull]')) continue;
                const m = line.match(/:\s*(\d+)%?/);
                if (m) { pctEl.textContent=m[1]+'%'; statusEl.textContent=''; }
                else { statusEl.textContent=line.trim(); }
            }
        } catch {}
    };

    poll();
    _pullProgressTimers[modelId] = setInterval(poll, 2000);
}

function stopPullProgress(modelId) { clearInterval(_pullProgressTimers[modelId]); delete _pullProgressTimers[modelId]; }

// ── Event wiring ───────────────────────────────────────────────
function wireEvents(actions) {
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-action]');
        if (!btn) return;
        actions[btn.dataset.action]?.(btn.dataset.model);
    });

    // Session spawn: + Chat / + Log buttons on model rows
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-action^="spawn-"]');
        if (!btn) return;
        const modelId = btn.dataset.model;
        const type = btn.dataset.action.replace('spawn-','');
        if (type === 'chat') SessionRegistry.add('chat', modelId);
        else if (type === 'log' && HAS_ADMIN_KEY) SessionRegistry.add('log', modelId);
    });

    document.addEventListener('click', (e) => {
        const closeBtn = e.target.closest('[data-action="close-session"]');
        if (!closeBtn) return;
        SessionRegistry.remove(closeBtn.dataset.sessionId);
    });
}

function populateSelect(selectId, models, current) {
    const sel = document.getElementById(selectId);
    if (!sel) return;
    let html = '<option value="">- none -</option>';
    for (const m of (models||[])) {
        const name = m.id.includes('/') ? m.id.split('/').pop() : m.id;
        html += `<option value="${esc(m.id)}"${m.id===current?' selected':''}>${esc(name)}</option>`;
    }
    sel.innerHTML = html;
}

// ── Chat session logic — per-session history and streaming ─────
async function _doSendChat(modelName, sessionId, inputEl) {
    const text = inputEl?.value?.trim();
    if (!text || !modelName) return;

    const session = SessionRegistry.get(sessionId);
    if (!session) return;

    const chatDisplay = document.getElementById('chat-display-' + (sessionId||''));
    const statusEl = document.getElementById('chat-status-' + (sessionId||''));

    // ── Auto-start if model is not running ──────────────────────
    const info = window._modelsCache?.find(m => m.id === modelName);
    const state = info?.status?.value ?? '';
    const canChatState = ['running', 'stopped', 'error'].includes(state) || !state;

    if (!canChatState || !info?.port) {
        try {
            const startBody = {};
            ['temperature','max_tokens','top_p','top_k'].forEach(k => {
                const v = SyncStore.get(modelName, k);
                if (v !== undefined && v !== '') startBody[k] = v;
            });
            if (statusEl) statusEl.textContent = 'Starting…';
            await window.apiRequest('/admin/start/'+encodeURIComponent(modelName), {}, 'POST', startBody);
        } catch(e) {
            if (statusEl) { statusEl.textContent='Start failed: '+e.message; setTimeout(()=>{statusEl.textContent=''},3000); }
            return;
        }
    }

    // Append user message
    chatDisplay?.appendChild(makeBubble('user', text));
    session.history.push({ role:'user', content:text });
    if (inputEl) inputEl.value = '';
    const status = document.getElementById('chat-status-' + (sessionId||''));
    if (status) status.textContent = '';

    // Create assistant bubble
    const bubble = makeBubble('assistant', '', true, sessionId);
    let acc = '';

    const abortCtrl = new AbortController();
    session.abortCtrl = abortCtrl;

    try {
        await doStream(text, modelName, (delta) => {
            if (abortCtrl.signal.aborted) return;
            acc += delta;
            requestAnimationFrame(() => {
                if (!abortCtrl.signal.aborted) {
                    bubble.innerHTML = typeof marked !== 'undefined' ? marked.parse(acc) : acc + '<span class="cursor"></span>';
                    chatDisplay?.scrollTo(0, chatDisplay.scrollHeight);
                }
            });
        }, { signal: abortCtrl.signal });

        bubble.classList.remove('streaming');
        const final = acc || (bubble.textContent||'').replace(/\u200B/g,'').trim();
        if (typeof marked !== 'undefined') bubble.innerHTML = marked.parse(final);
        session.history.push({ role:'assistant', content:final });
    } catch(e) {
        if (e.name !== 'AbortError') chatDisplay?.appendChild(makeBubble('assistant', '⚠ '+e.message));
    }
}

function makeBubble(role, content, streaming, sessionId) {
    const el = document.getElementById('chat-display-' + (sessionId||''));
    if (!el) return;
    const div = document.createElement('div');
    div.className = 'message ' + role + (streaming ? ' streaming' : '');
    if (role === 'user') div.textContent = content;
    else div.innerHTML = streaming ? content+'<span class="cursor"></span>' : content;
    el.appendChild(div);
    if (el.scrollHeight - el.scrollTop <= el.clientHeight+40) el.scrollTop = el.scrollHeight;
    return div;
}

async function doStream(text, modelName, onData, options={}) {
    const { signal } = options;
    const info = window._modelsCache?.find(m => m.id === modelName);
    if (!info?.port) throw new Error('Model not running');

    const params = {};
    ['temperature','max_tokens','top_p','top_k'].forEach(k => {
        const v = SyncStore.get(modelName, k);
        if (v !== undefined && v !== '') params[k] = v;
    });

    const resp = await fetch(BASE_URL+'/v1/chat/completions', {
        method:'POST', headers:{'Content-Type':'application/json'}, signal,
        body: JSON.stringify({
            model: info.id || modelName, messages: [], stream:true,
            temperature: params.temperature ?? 0.7, top_p: params.top_p ?? 0.95,
            ...(params.max_tokens ? { max_tokens: params.max_tokens } : {}),
        }),
    });
    if (!resp.ok) throw new Error('Chat API '+resp.status);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream:true });
        let idx;
        while ((idx=buf.indexOf('\n'))>=0) {
            const line = buf.slice(0,idx).trim();
            buf = buf.slice(idx+1);
            if (!line || line==='[DONE]') continue;
            let t = line.startsWith('data: ') ? line.slice(6) : line;
            try {
                const p = JSON.parse(t);
                if (p.error) { const err=new Error(p.error); err.streamError=true; throw err; }
                const d = p.choices?.[0]?.delta?.content;
                if (d) onData(d);
            } catch(e) { if (e.streamError) throw e; }
        }
    }
}

// ── Log polling (admin only, per-session) ──────────────────────
let logTimers = {}; // sessionId → timerId
let logSinces = {}; // sessionId → cursor

function startLogPoll(sessionId, modelId) {
    if (!HAS_ADMIN_KEY) return;
    stopLogPoll(sessionId);
    const poll = async () => {
        try {
            const params = { lines:200 };
            if (logSinces[sessionId]) params.since = logSinces[sessionId];
            const data = await window.apiRequest('/admin/log/'+encodeURIComponent(modelId), params);
            if (!data?.lines?.length) return;
            const display = document.getElementById('log-display-' + sessionId);
            if (!display) return;
            for (const entry of data.lines) {
                const div = document.createElement('div');
                div.className = 'log-line'; div.textContent = entry.text;
                display.appendChild(div);
            }
            display.scrollTop = display.scrollHeight;
            logSinces[sessionId] = data.since ?? logSinces[sessionId];
        } catch {}
    };
    poll();
    logTimers[sessionId] = setInterval(poll, CFG.POLL_INTERVAL);
}

function stopLogPoll(sessionId) {
    if (logTimers[sessionId]) clearInterval(logTimers[sessionId]);
    delete logTimers[sessionId];
    delete logSinces[sessionId];
}

// ── API helpers ────────────────────────────────────────────────
function _adminKey() { return document.querySelector('meta[name="arkestra-admin-key"]')?.content||''; }

async function apiFetch(path, opts={}) {
    const base = window.location.origin + BASE_URL;
    let r;
    try { r = await fetch(base+path, opts); } catch(e) { throw e; }

    // 401/403 fallback from /admin/* → /api/* for GET requests
    if ((!r.ok || r.status===401||r.status===403) && opts.method!=='POST' && opts.method!=='DELETE') {
        try { r = await fetch(base+path.replace(/^\/admin/,'/api'), opts); } catch(e2) {}
    }

    if (!r.ok) {
        let detail = 'HTTP '+r.status;
        try { detail += ': '+(await r.text()); } catch {}
        throw new Error(detail);
    }
    return r.json();
}

async function apiRequest(path, params, method, body) {
    let actualMethod='GET', actualBody=null, actualParams={};
    if (method && body!==undefined) { actualMethod=method; actualBody=body; actualParams=params||{}; }
    else { actualMethod='GET'; actualParams=params||{}; }

    const qs = new URLSearchParams(actualParams).toString();
    const opts = { method:actualMethod };
    if (actualBody!==null) {
        opts.headers={'Content-Type':'application/json'};
        if (HAS_ADMIN_KEY) opts.headers['X-Admin-Key']=_adminKey();
        opts.body=JSON.stringify(actualBody);
    } else if (HAS_ADMIN_KEY && actualMethod==='GET') {
        opts.headers={'X-Admin-Key':_adminKey()};
    }
    return apiFetch(path+(qs?'?'+qs:''), opts);
}

async function adminGet(path, params) { return apiRequest(path, params, 'GET', null); }
async function adminPost(path, body) { return apiRequest(path, {}, 'POST', body); }

// ── Expose public API ──────────────────────────────────────────
window.EventBus = EventBus;
window.CFG = CFG;
window.esc = esc;
window.render = render;
window.wireEvents = wireEvents;
window.adminPost = adminPost;
window.adminGet = adminGet;
window.apiRequest = apiRequest;
window.sanitizeId = sanitizeId;
window.normalizeStatus = normalizeStatus;
window.startLogPoll = startLogPoll;
window.stopLogPoll = stopLogPoll;
window._doSendChat = _doSendChat;  // per-session send
window.makeBubble = makeBubble;
window.renderModelRow = renderModelRow;
window.populateSelect = populateSelect;
window.SyncStore = SyncStore;
window.SessionRegistry = SessionRegistry;
window.startPull = startPull;
window.cancelPull = cancelPull;
window.syncRowButtons = syncRowButtons;
window.pollPullProgress = pollPullProgress;
window.stopPullProgress = stopPullProgress;
window.formatSizeGB = formatSizeGB;
