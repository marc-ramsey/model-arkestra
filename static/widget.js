/*
 * widget.js — JSON-driven layout engine for ModelArkestra admin.
 * Pattern: data (JSON tree) -> render() -> DOM. Conventions wire events.
 */

// ── Pub/sub bus ────────────────────────────────────────────────
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
function keyToLabel(key) {
    return key.replace(/[-_](.)/g, (_, c) => c.toUpperCase());
}

function resolveArgValue(cfg, key) {
    if (cfg[key] !== undefined && cfg[key] !== '') return String(cfg[key]);
    if (cfg.args?.[key] !== undefined) return String(cfg.args[key]);
    return '';
}

// ── render() — walks JSON tree -> DOM ─────────────────────────
function render(node) {
    if (!node?.widget) return null;
    const key = node.widget.charAt(0).toUpperCase() + node.widget.slice(1);
    return renderers[key]?.(node) ?? null;
}

const renderers = {};

// ═══════════════════════════════════════════════════════════
// Layout containers
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

    // Wire drag-to-resize on dividers in this container
    setTimeout(() => {
        for (const d of el.querySelectorAll(':scope > .divider-v, :scope > .divider-h')) {
            if (d.dataset.dragWired) continue;
            d.dataset.dragWired = '1';
            const isH = d.classList.contains('divider-v');
            let dragging = false, val = 0;
            const onMove = (ev) => {
                if (!dragging) return;
                const rect = d.parentElement.getBoundingClientRect();
                val = isH ? ((ev.clientX-rect.left)/rect.width)*100 : ((ev.clientY-rect.top)/rect.height)*100;
                if (val < 20 || val > 80) return;
                const p = d.previousElementSibling, n = d.nextElementSibling;
                if (p && n) { p.style.flex=`0 0 ${val}%`; n.style.flex=`0 0 ${100-val}%`; }
                localStorage.setItem('arkestra-layout-' + (isH?'h':'v'), Math.round(val));
            };
            const onUp = () => {
                dragging=false; d.classList.remove('active');
                document.body.style.userSelect='';
                window.removeEventListener('mousemove',onMove);
                window.removeEventListener('mouseup',onUp);
                document.removeEventListener('mouseup',onUpGlobal);
            };
            const onUpGlobal = () => { if (dragging) onUp(); };
            d.addEventListener('mousedown', (e) => {
                e.preventDefault();
                document.body.style.userSelect = 'none';
                d.classList.add('active');
                dragging = true;
                window.addEventListener('mousemove',onMove);
                window.addEventListener('mouseup',onUp);
                document.addEventListener('mouseup',onUpGlobal);
            });
        }
    }, 0);

    el.style.flex = '1';
    el.style.minHeight = '0';
    return el;
};

renderers.AccordionContainer = function() {
    const el = document.createElement('div');
    el.className = 'accordion';
    el.id = 'left-accordion';
    const header = document.createElement('h3');
    header.textContent = 'Model Cluster';
    el.appendChild(header);
    const body = document.createElement('div');
    body.className = 'acc-body';
    body.id = 'model-accordion-items';
    el.appendChild(body);

    let collapsed = false;
    header.addEventListener('click', () => {
        collapsed = !collapsed;
        body.style.display = collapsed ? 'none' : '';
    });
    return el;
};

renderers.AccGroup = function({ title, group }) {
    const el = document.createElement('div');
    el.className = 'acc-group';
    el.dataset.group = group;
    const header = document.createElement('h3');
    header.className = 'acc-group-header';
    header.innerHTML = '<span>' + esc(title) + '</span><span class="acc-group-count">' + (group === 'ready' ? 0 : 0) + '</span><span></span>';
    const body = document.createElement('div');
    body.className = 'acc-body';
    body.id = 'models-' + group + '-items';
    el.appendChild(header);
    el.appendChild(body);

    let collapsed = false;
    header.addEventListener('click', () => {
        collapsed = !collapsed;
        body.style.display = collapsed ? 'none' : '';
    });
    return { element: el, body: body, header: header };
};

// ═══════════════════════════════════════════════════════════
// Panes - domain-specific UI containers
// ═══════════════════════════════════════════════════════════

renderers.LogPane = function() {
    const el = document.createElement('div');
    el.className = 'pane pane-logs';
    const header = document.createElement('div');
    header.className = 'pane-header';
    header.innerHTML = '<span class="pane-title">Log</span><label>Select Model:</label><select id="log-model-select"></select>';
    el.appendChild(header);
    const display = document.createElement('pre');
    display.className = 'log-container';
    display.id = 'log-display';
    el.appendChild(display);
    return el;
};

renderers.ChatPane = function() {
    const el = document.createElement('div');
    el.className = 'pane pane-chat';
    const header = document.createElement('div');
    header.className = 'pane-header';
    header.innerHTML = '<span class="pane-title">Chat</span><label>Select Model:</label><select id="chat-model-select"></select>' +
        '<span class="chat-params-toggle" id="btn-toggle-chat-params">Params</span>' +
        '<span class="chat-tts-toggle" title="Text-to-Speech on/off">TTS: <span id="tts-status">Off</span></span>';
    el.appendChild(header);

    const messages = document.createElement('div');
    messages.className = 'chat-messages';
    messages.id = 'chat-display';
    el.appendChild(messages);

    // Audio playback bar (hidden by default)
    const audioBar = document.createElement('div');
    audioBar.className = 'audio-playback-bar hidden';
    audioBar.id = 'audio-playback-bar';
    audioBar.innerHTML = '<span class="audio-label">🔊</span>' +
        '<input type="range" id="audio-progress" min="0" max="100" value="0" step="0.1">' +
        '<span class="audio-time" id="audio-current">0:00</span> / ' +
        '<span class="audio-time" id="audio-duration">0:00</span>' +
        '<button id="btn-pause-audio" title="Pause/Resume">⏸</button>' +
        '<button id="btn-stop-audio" title="Stop">⏹</button>';
    el.appendChild(audioBar);

    const paramsPanel = document.createElement('div');
    paramsPanel.className = 'chat-params-panel';
    paramsPanel.id = 'chat-params-panel';
    paramsPanel.innerHTML = '<div class="chat-params-grid">' +
        '<div class="chat-param"><label>Temp</label><input type="number" id="f-chat-temp" min="0" max="2" step="0.05" value="0.7"></div>' +
        '<div class="chat-param"><label>Max Tokens</label><input type="number" id="f-chat-max-tokens" min="1" max="8192" step="1" value="4096"></div>' +
        '<div class="chat-param"><label>Top-P</label><input type="number" id="f-chat-top-p" min="0" max="1" step="0.05" value="0.95"></div>' +
        '<div class="chat-param"><label>Top-K</label><input type="number" id="f-chat-top-k" min="1" max="256" step="1" value="40"></div>' +
        '</div>';
    el.appendChild(paramsPanel);

    const inputBar = document.createElement('div');
    inputBar.className = 'chat-input-bar';
    inputBar.innerHTML = '<textarea id="f-chat-input" rows="4" placeholder="Type a message..."></textarea>' +
        '<button id="btn-send-tts" title="Speak (TTS)">🔊</button>' +
        '<button id="btn-send-chat" title="Send">Send</button>' +
        '<span class="chat-status" id="chat-status"></span>';
    el.appendChild(inputBar);

    // ── Wire ChatPane internals ───────────────────────────────
    let ttsActive = false;
    header.querySelector('.chat-tts-toggle')?.addEventListener('click', () => {
        ttsActive = !ttsActive;
        const statusEl = document.getElementById('tts-status');
        if (statusEl) statusEl.textContent = ttsActive ? 'On' : 'Off';
        header.querySelector('.chat-tts-toggle').style.color = ttsActive ? 'var(--green)' : '';
    });

    // Params panel toggle
    header.querySelector('#btn-toggle-chat-params')?.addEventListener('click', () => {
        document.getElementById('chat-params-panel')?.classList.toggle('open');
    });

    // TTS speak button
    inputBar.querySelector('#btn-send-tts')?.addEventListener('click', () => {
        const textEl = document.getElementById('f-chat-input');
        window._audio?.sendTTS(textEl?.value?.trim() || '');
    });

    // Send chat button and Enter key
    inputBar.querySelector('#btn-send-chat')?.addEventListener('click', () => {
        const sel = document.getElementById('chat-model-select');
        const textEl = document.getElementById('f-chat-input');
        if (sel?.value && textEl?.value.trim()) window.sendChat(sel.value, textEl);
    });

    // Save chat params to localStorage on change
    inputBar.addEventListener('input', (e) => {
        if (!e.target.id?.startsWith('f-chat-')) return;
        const name = e.target.id.replace('f-chat-', '');
        const modelName = document.getElementById('chat-model-select')?.value;
        if (!modelName) return;
        try {
            const params = JSON.parse(localStorage.getItem(CFG.STORAGE_CHAT_PARAMS)||'{}');
            if (!params[modelName]) params[modelName] = {};
            const apiName = name === 'temp' ? 'temperature' :
                            name === 'max-tokens' ? 'max_tokens' :
                            name === 'top-p' ? 'top_p' :
                            name === 'top-k' ? 'top_k' : name;
            params[modelName][apiName] = e.target.type === 'number' ? Number(e.target.value) : e.target.value;
            localStorage.setItem(CFG.STORAGE_CHAT_PARAMS, JSON.stringify(params));
        } catch {}
    });

    return el;
};

// ═══════════════════════════════════════════════════════════
// ConfigPanel - per-model edit form
// ═══════════════════════════════════════════════════════════

renderers.ConfigPanel = function({ id, fields }) {
    const el = document.createElement('div');
    el.className = 'config-panel';
    el.id = 'config-' + sanitizeId(id);
    el.dataset.model = id;

    (fields||[]).forEach(f => {
        const isFullWidth = f.widget === 'TextArea';
        const label = document.createElement('label');
        label.className = 'field-label' + (isFullWidth ? ' full-width' : '');
        label.textContent = f.label || keyToLabel(f.name);
        el.appendChild(label);

        let input;
        if (f.options) {
            input = document.createElement('select');
            for (const opt of f.options) {
                const o = document.createElement('option');
                o.value = String(opt.value); o.textContent = opt.label || opt.value;
                input.appendChild(o);
            }
        } else if (f.widget === 'TagsInput' && f.options) {
            input = document.createElement('select');
            input.multiple = true; input.size = Math.min(f.options.length, 6);
            const current = f.currentTags || [];
            for (const opt of f.options) {
                const o = document.createElement('option');
                o.value = String(opt.value);
                o.textContent = opt.value;
                if (current.includes(opt.value)) o.selected = true;
                input.appendChild(o);
            }
            input.dataset.tagsValue = JSON.stringify(current);
            input.addEventListener('change', () => {
                const selected = Array.from(input.selectedOptions).map(o => o.value);
                input.dataset.tagsValue = JSON.stringify(selected);
            });
        } else if (f.schema?.type === 'integer') {
            input = document.createElement('input'); input.type = 'number'; input.step = '1';
            f.minimum != null && (input.min = f.minimum);
            f.maximum != null && (input.max = f.maximum);
        } else if (f.schema?.type === 'float') {
            input = document.createElement('input'); input.type = 'number'; input.step = f.step ?? 1;
            f.minimum != null && (input.min = f.minimum);
            f.maximum != null && (input.max = f.maximum);
        } else if (f.schema?.type === 'bool') {
            input = document.createElement('select');
            input.innerHTML = '<option value="false">false</option><option value="true">true</option>';
        } else if (isFullWidth) {
            const ta = document.createElement('textarea');
            ta.rows = f.schema?.rows ?? 2;
            input = ta;
        } else {
            input = document.createElement('input'); input.type = 'text';
        }

        const valueCell = document.createElement('div');
        valueCell.className = 'field-value' + (isFullWidth ? ' full-width' : '');
        if (input) {
            input.id = f.name;
            if (!input.multiple) input.value = f.value ?? '';
            valueCell.appendChild(input);
        }
        el.appendChild(valueCell);
    });

    // Action buttons row
    const bar = document.createElement('div');
    bar.className = 'model-actions';
    (window._configActions||[]).forEach(a => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.dataset.action = a;
        btn.dataset.model = id;
        btn.title = a.charAt(0).toUpperCase() + a.slice(1);
        const iconMap = { start:'▶', stop:'■', save:'✓', reset:'↺', eject:'⏏' };
        btn.textContent = iconMap[a] || a;
        if (['stop','eject'].includes(a)) btn.classList.add('btn-danger');
        if (a === 'start') btn.classList.add('btn-success');
        bar.appendChild(btn);
    });
    el.appendChild(bar);

    return el;
};

// ═══════════════════════════════════════════════════════════
// Model row - clickable accordion item with deferred config panel
// ═══════════════════════════════════════════════════════════

function renderModelRow(model) {
    const name = model.id.includes('/') ? model.id.split('/').pop() : model.id;
    const statusClass = normalizeStatus(model.status);

    const row = document.createElement('div');
    row.className = 'model-row';
    row.dataset.model = model.id;

    // Build inline action buttons HTML
    let btns = '';
    (window._configActions||[]).forEach(a => {
        btns += '<button type="button" data-action="'+a+'" data-model="'+esc(model.id)+'" title="'+a.charAt(0).toUpperCase()+a.slice(1)+'"></button>';
    });

    row.innerHTML = '<div class="model-name-bar"><span class="status-dot '+statusClass+'"></span>' +
        '<span class="model-name">'+esc(name)+'</span>' +
        (model.runner_type ? '<span class="runner-badge" style="font-size:10px;color:var(--text-dim);margin-left:auto;flex-shrink:0;">('+esc(model.runner_type)+')</span>' : '') +
        '<div class="model-actions-inline">'+btns+'</div></div>';
    
    // Pull progress bar (hidden until downloading)
    const prog = document.createElement('div');
    prog.className = 'pull-progress hidden';
    prog.innerHTML = '<span class="pull-status"></span><span class="pull-pct"></span>';
    row.appendChild(prog);

    row.addEventListener('click', async (e) => {
        if (row.querySelector('.config-panel')?.contains(e.target)) return;
        if (e.target.tagName === 'BUTTON') return;
        EventBus.emit('model.select', { modelId: model.id });
        row.classList.toggle('expanded');
        if (row.querySelector('.config-panel')) return;

        try {
            const data = await window.adminGet('/admin/config/' + encodeURIComponent(model.id));
            if (!data?.config) return;
            window._argSchema = data.args_schema || {};

            const cp = data.config.checkpoint || '';
            const fields = [];
            const defaults = data.default || {};

            for (const k of ['name', 'repo', 'model']) {
                const schema = data.args_schema[k];
                if (!schema) continue;
                let value = resolveArgValue(data.config, k);
                if (value === '' && schema.default !== undefined) value = String(schema.default);
                fields.push({ name:k, value: k==='name' ? (value || data.model) : value, label:k, schema, options:schema.options?.map(v=>({value:v})) || undefined });
            }

            const mmprojSchema = data.args_schema['mmproj'];
            if (mmprojSchema) {
                let value = resolveArgValue(data.config, 'mmproj');
                fields.push({ name:'mmproj', value: value || '', label:'mmproj', schema:mmprojSchema });
            }

            const resolvedBackend = data.config.backend || defaults.backend || '';
            const bkOpts = Object.entries(data.backends||{}).map(([k,v]) => ({
                value: k, label: (typeof v==='object')?(v.host||k):k
            }));
            if (bkOpts.length) {
                fields.push({ name:'backend', value:resolvedBackend, options:bkOpts, widget:'SelectInput' });
            }

            const resolvedRunner = data.config.runner || defaults.runner || '';
            const rnOpts = data.runner_types?.map(t => ({value:t})) || [];
            if (rnOpts.length) {
                const all = [{value:''}]; for (const r of rnOpts) all.push(r);
                fields.push({ name:'runner', value:resolvedRunner, options:all, widget:'SelectInput' });
            }

            const currentTags = data.config.tags || [];
            const tagOpts = (data.tags || []).map(t => ({value: t}));
            if (tagOpts.length) {
                fields.push({ name: 'tags', value: Array.isArray(currentTags) ? currentTags.join(',') : '',
                    label: 'tags', schema: { type: 'string' }, widget: 'TagsInput', options: tagOpts, currentTags });
            }

            for (const k of Object.keys(data.args_schema || {})) {
                if (['name','repo','model','mmproj'].includes(k)) continue;
                const schema = data.args_schema[k];
                let value = resolveArgValue(data.config, k);
                if (value === '' && defaults[k] !== undefined) value = String(defaults[k]);
                else if (value === '' && schema?.default !== undefined) value = String(schema.default);
                fields.push({ name:k, value, label:k, schema, options:schema.options?.map(v=>({value:v})) || undefined });
            }

            const panel = renderers.ConfigPanel({ id: model.id, fields });
            _configSnapshots[model.id] = JSON.parse(JSON.stringify(data.config));
            panel.querySelectorAll('input, select, textarea').forEach(el => {
                el.addEventListener('input', () => checkDirty(model.id, panel));
            });
            checkDirty(model.id, panel);

            row.appendChild(panel);
        } catch(e) { console.error('[widget] config fetch failed:', e.message); }
    });

    return row;
}

// ── Pull helpers ───────────────────────────────────────────────
async function startPull(modelId) {
    const row = document.querySelector('.model-row[data-model="'+escapeSelector(modelId)+'"]');
    if (!row) return;
    try {
        await window.adminPost('/admin/pull/'+encodeURIComponent(modelId), {});
        updateRowForState(row, { state: 'downloading', statusClass: 'loading' });
        const progEl = row.querySelector('.pull-progress');
        if (progEl) progEl.classList.remove('hidden');
        pollPullProgress(modelId);
    } catch(e) {
        console.error('[widget] pull failed:', e.message);
    }
}

async function cancelPull(modelId) {
    const row = document.querySelector('.model-row[data-model="'+escapeSelector(modelId)+'"]');
    if (!row) return;
    try {
        await window.adminPost('/admin/pull/stop/'+encodeURIComponent(modelId), {});
        updateRowForState(row, { state: 'uncached', statusClass: 'uncached' });
        const progEl = row.querySelector('.pull-progress');
        if (progEl) progEl.classList.add('hidden');
        stopPullProgress(modelId);
    } catch(e) {
        console.error('[widget] cancel failed:', e.message);
    }
}

function escapeSelector(id) {
    return id.replace(/["'#%&*,/:<=>?@[\\]^`{|}~]/g, '\\$&');
}

function updateRowForState(row, info) {
    const modelId = row.dataset.model;
    if (info.statusClass !== undefined) {
        const dot = row.querySelector('.status-dot');
        if (dot) { dot.className = 'status-dot ' + info.statusClass; }
    }
    syncRowButtons(row, info);
    if (info.moveToGroup) {
        const group = document.getElementById('models-'+info.moveToGroup+'-items');
        if (group) group.appendChild(row);
    }
}

// ── Progress polling for downloads ─────────────────────────────
let _pullProgressTimers = {};

function pollPullProgress(modelId) {
    const row = document.querySelector('.model-row[data-model="'+escapeSelector(modelId)+'"]');
    if (!row) return;
    let lastPct = '', lastMsg = '';

    const poll = async () => {
        try {
            const data = await window.adminGet('/admin/log/'+encodeURIComponent(modelId), { lines: 10 });
            if (!data?.lines?.length) return;
            const progEl = row.querySelector('.pull-progress');
            const statusEl = progEl?.querySelector('.pull-status');
            const pctEl = progEl?.querySelector('.pull-pct');
            
            for (let i = data.lines.length - 1; i >= 0; i--) {
                const line = data.lines[i].text;
                if (!line.includes('[pull]')) continue;
                const m = line.match(/:\s*(\d+)%?/);
                if (m) {
                    lastPct = m[1];
                    pctEl.textContent = m[1] + '%';
                    statusEl.textContent = '';
                } else {
                    statusEl.textContent = line.trim();
                }
            }
        } catch {}
    };

    poll();
    _pullProgressTimers[modelId] = setInterval(poll, 2000);
}

function stopPullProgress(modelId) {
    clearInterval(_pullProgressTimers[modelId]);
    delete _pullProgressTimers[modelId];
}

// ── Row button state sync ──────────────────────────────────────
const BTN_MAP = { start:'▶', stop:'■', eject:'⏏', cancel:'✕', save:'✓', reset:'↺' };

function syncRowButtons(row, info) {
    const btns = row.querySelectorAll('.model-actions-inline button');
    if (!btns.length) return;
    
    const state = info?.state || 'stopped'; // stopped, running, error, uncached, downloading
    for (const btn of btns) {
        const action = btn.dataset.action;
        btn.disabled = false;
        let text = BTN_MAP[action] || action;

        switch(state) {
            case 'downloading':
                if (action === 'eject') { btn.textContent = '⏏'; btn.title = 'Pulling...'; btn.disabled = true; }
                else if (!['start','stop','save','reset'].includes(action)) { text = ''; }
                break;
            case 'running':
                if (action === 'start') btn.disabled = true;
                if (action === 'eject') btn.disabled = true;
                if (action === 'save') btn.disabled = false; // can save config while running
                if (action === 'reset') btn.disabled = false;
                break;
            case 'stopped':
                if (action === 'stop') btn.disabled = true;
                if (action === 'eject') btn.disabled = false;
                if (action === 'save') btn.disabled = true; // nothing changed
                break;
            case 'error':
                if (action === 'stop') btn.disabled = true;
                if (action === 'eject') btn.disabled = false;
                if (action === 'save') btn.disabled = true;
                break;
            case 'uncached':
                // Replace eject icon with pull for uncached models
                if (action === 'eject') { text = '⏏'; btn.title = 'Pull model checkpoint'; }
                else if (!['start','stop','save','reset'].includes(action)) { text = ''; }
                break;
        }

        // For downloading state, hide non-pull buttons from visible space
        if (state === 'downloading' && !['start','stop','save','reset'].includes(action)) {
            btn.style.display = action === 'eject' ? '' : 'none';
        } else {
            btn.style.display = '';
        }

        // Update text content and add danger/success classes
        btn.textContent = text;
        btn.className = '';
        if (action === 'stop' || action === 'eject') btn.classList.add('btn-danger');
        if (action === 'start') btn.classList.add('btn-success');
    }
}

// ── Dirty detection ────────────────────────────────────────────
function checkDirty(modelId, panel) {
    const snap = _configSnapshots?.[modelId];
    if (!snap) return;

    const val = (name) => {
        const el = panel.querySelector('#' + name);
        return el ? (el.type === 'number' ? Number(el.value) : el.value) : '';
    };

    let isDirty = false;
    const argKeys = Object.keys(window._argSchema || {});
    for (const key of argKeys) {
        if (val(key) !== resolveArgValue(snap, key)) { isDirty = true; break; }
    }
    if (!isDirty && val('repo') !== resolveArgValue(snap, 'repo')) isDirty = true;
    else if (!isDirty && val('model') !== resolveArgValue(snap, 'model')) isDirty = true;
    else if (!isDirty && val('backend') !== snap.backend) isDirty = true;
    else if (!isDirty && val('runner') !== snap.runner) isDirty = true;

    const btn = panel.querySelector('[data-action="save"]');
    if (btn) btn.disabled = !isDirty;
}

// ── Event wiring conventions ───────────────────────────────────
function wireEvents(actions) {
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-action]');
        if (!btn) return;
        actions[btn.dataset.action]?.(btn.dataset.model);
    });

    document.addEventListener('input', (e) => {
        const input = e.target;
        if (!input.id) return;
        const panel = input.closest('.config-panel');
        e.detail = { model: panel?.dataset.model ?? '', name: input.id, value: input.value };
        input.dispatchEvent(new CustomEvent('field.change', { bubbles:true, detail:e.detail }));
    });
}

function populateSelect(selectId, models, current) {
    const sel = document.getElementById(selectId);
    if (!sel) return;
    let html = '<option value="">- none -</option>';
    for (const m of (models||[])) {
        const name = m.id.includes('/') ? m.id.split('/').pop() : m.id;
        html += '<option value="'+esc(m.id)+'"' + (m.id===current?' selected':'') + '>' + esc(name) + '</option>';
    }
    sel.innerHTML = html;
}

// ── Log polling - delta fetch with cursor ──────────────────────
let logTimer = null, logSince = 0;

function startLogPoll(modelId) {
    stopLogPoll();
    const poll = async () => {
        try {
            if (!modelId) return;
            const params = { lines: 200 };
            if (logSince) params.since = logSince;
            const data = await window.adminGet('/admin/log/' + encodeURIComponent(modelId), params);
            if (!data?.lines?.length) return;

            const display = document.getElementById('log-display');
            if (!display) return;

            for (const entry of data.lines) {
                const div = document.createElement('div');
                div.className = 'log-line';
                div.textContent = entry.text;
                display.appendChild(div);
            }

            if (display.scrollHeight - display.scrollTop <= display.clientHeight + 60) {
                display.scrollTop = display.scrollHeight;
            }
            logSince = data.since ?? logSince;
        } catch {}
    };
    poll();
    logTimer = setInterval(poll, CFG.POLL_INTERVAL);
}

function stopLogPoll() { if (logTimer) clearInterval(logTimer); logTimer=null; logSince=0; }

// ── Chat streaming - SSE to model's port /v1/chat/completions ─
let _streamAcc = '', _streamBubble = null;

function appendBubble(role, content, streaming) {
    const el = document.getElementById('chat-display');
    if (!el) return;
    const div = document.createElement('div');
    div.className = 'message ' + role + (streaming?' streaming':'');
    div.textContent = role === 'user' ? content : '';
    if (role === 'assistant') div.innerHTML = streaming ? content+'<span class="cursor"></span>' : content;
    el.appendChild(div);
    if (el.scrollHeight - el.scrollTop <= el.clientHeight+40) el.scrollTop = el.scrollHeight;
    return div;
}

async function doStream(text, modelName, onData, options = {}) {
    const { signal } = options;
    const info = window._modelsCache?.find(m => m.id === modelName);
    if (!info?.port) throw new Error('Model not running');

    const params = JSON.parse(localStorage.getItem(CFG.STORAGE_CHAT_PARAMS)||'{}')[modelName] || {};
    const resp = await fetch('/v1/chat/completions', {
        method: 'POST', headers:{'Content-Type':'application/json'}, signal,
        body: JSON.stringify({
            model: info.id || modelName, messages:chatHistory, stream:true,
            temperature: params.temperature??0.7,
            top_p: params.top_p??0.95,
            ...(params.max_tokens ? { max_tokens: params.max_tokens } : {}),
        }),
    });
    if (!resp.ok) throw new Error('Chat API ' + resp.status);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream:true });
        let idx;
        while ((idx = buf.indexOf('\n')) >= 0) {
            const line = buf.slice(0, idx).trim();
            buf = buf.slice(idx + 1);
            if (!line || line === '[DONE]') continue;
            let t = line.startsWith('data: ') ? line.slice(6) : line;
            try {
                const p = JSON.parse(t);
                if (p.error) { const err = new Error(p.error); err.streamError = true; throw err; }
                const d = p.choices?.[0]?.delta?.content;
                if (d) onData(d);
            } catch (e) {
                if (e.streamError) throw e;
            }
        }
    }
}

async function sendChat(modelName, inputEl = null) {
    const el = inputEl ?? document.getElementById('f-chat-input');
    if (!el) return;
    const text = el.value.trim();
    if (!text || !modelName) return;

    appendBubble('user', text);
    chatHistory.push({ role:'user', content:text });
    el.value = '';

    if (_streamBubble && _streamBubble.classList.contains('streaming')) {
        _streamBubble.classList.remove('streaming');
    }
    const bubble = appendBubble('assistant', '', true);
    _streamAcc = '';
    _streamBubble = bubble;
    stopLogPoll();

    const abortCtrl = new AbortController();
    try {
        await doStream(text, modelName, (delta) => {
            if (abortCtrl.signal.aborted) return;
            _streamAcc += delta;
            requestAnimationFrame(() => {
                if (_streamBubble && !abortCtrl.signal.aborted) {
                    _streamBubble.innerHTML = typeof marked !== 'undefined' ? marked.parse(_streamAcc) : _streamAcc + '<span class="cursor"></span>';
                }
                const chatDisplay = document.getElementById('chat-display');
                if (chatDisplay && chatDisplay.scrollHeight - chatDisplay.scrollTop <= chatDisplay.clientHeight + 40) {
                    chatDisplay.scrollTop = chatDisplay.scrollHeight;
                }
            });
        }, { signal: abortCtrl.signal });

        _streamBubble?.classList.remove('streaming');
        const final = _streamAcc || (_streamBubble?.textContent||'').replace(/\u200B/g,'').trim();
        if (typeof marked !== 'undefined') _streamBubble.innerHTML = marked.parse(final);
        chatHistory.push({ role:'assistant', content:final });
    } catch(e) {
        if (e.name !== 'AbortError') appendBubble('assistant', '\u26a0 ' + e.message);
    }
}

// ── API helpers ────────────────────────────────────────────────
function _adminKey() { return document.querySelector('meta[name="arkestra-admin-key"]')?.content||''; }

async function adminGet(path, params) {
    const qs = new URLSearchParams(params).toString();
    const r = await fetch(window.location.origin+path+(qs?'?'+qs:''), { headers:_adminKey()?{'X-Admin-Key':_adminKey()}:{} });
    if (!r.ok) throw new Error('HTTP '+r.status);
    return r.json();
}

async function adminPost(path, body) {
    const k = _adminKey();
    const r = await fetch(window.location.origin+path, {
        method:'POST', headers:{'Content-Type':'application/json',...(k?{'X-Admin-Key':k}:{})},
        body: JSON.stringify(body),
    });
    if (!r.ok) { const t=await r.text(); throw new Error('HTTP '+r.status+': '+t); }
    return r.json();
}

// ── Module-scoped state (closed over by deferred handlers) ────
let _configSnapshots = {};
let chatHistory = [];

// ── Expose public API ──────────────────────────────────────────
window.EventBus = EventBus;
window.CFG = CFG;
window.esc = esc;
window.render = render;
window.wireEvents = wireEvents;
window.adminPost = adminPost;
window.adminGet = adminGet;
window.sanitizeId = sanitizeId;
window.normalizeStatus = normalizeStatus;
window.startLogPoll = startLogPoll;
window.stopLogPoll = stopLogPoll;
window.sendChat = sendChat;
window.appendBubble = appendBubble;
window.renderModelRow = renderModelRow;
window.populateSelect = populateSelect;
