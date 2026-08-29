/*
 * widget.js — JSON-driven layout engine for ModelArkestra admin.
 * Pattern: data (JSON tree) -> render() -> DOM. Conventions wire events.
 */

// EventBus - simple pub/sub
const EventBus = {
    _h: new Map(),
    on(e, fn)  { const l = this._h.get(e)||[]; l.push(fn); this._h.set(e,l); return () => this.off(e,fn); },
    off(e, fn) { const l = this._h.get(e); if(l){const i=l.indexOf(fn);if(i>=0)l.splice(i,1);} },
    emit(e, d)  { for(const fn of this._h.get(e)||[]) fn(d); },
};

function esc(s)  { return (s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function sanitizeId(n) { return (n||'').replace(/[^a-zA-Z0-9_-]/g, '_'); }
function normalizeStatus(s) { return (s?.value || s || '').replace('runnerstate.','').toLowerCase(); }

// Convert snake_case/hyphenated key to Title-Case label: "top-p" → "Top-P"
function keyToLabel(key) {
    return key.replace(/[-_](.)/g, (_, c) => c.toUpperCase());
}

// render() - walks JSON tree -> DOM. Returns element.
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
    if (axis === 'v') { el.style.flexDirection = 'column'; }

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
            const isH = d.classList.contains('divider-v');
            let dragging = false;
            let val = 0;
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

    // Toggle collapse
    let collapsed = false;
    header.addEventListener('click', () => {
        collapsed = !collapsed;
        body.style.display = collapsed ? 'none' : '';
    });

    return el;
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
        '<span class="chat-params-toggle" id="btn-toggle-chat-params">Params</span>';
    el.appendChild(header);

    const messages = document.createElement('div');
    messages.className = 'chat-messages';
    messages.id = 'chat-display';
    el.appendChild(messages);

    const paramsPanel = document.createElement('div');
    paramsPanel.className = 'chat-params-panel';
    paramsPanel.id = 'chat-params-panel';
    paramsPanel.innerHTML = '<div class="chat-params-grid">' +
        '<div class="chat-param"><label>Temp</label><input type="number" id="f-chat-temp" min="0" max="2" step="0.05" value="0.7"></div>' +
        '<div class="chat-param"><label>Max Tokens</label><input type="number" id="f-chat-max-tokens" min="1" max="8192" step="1" value="512"></div>' +
        '<div class="chat-param"><label>Top-P</label><input type="number" id="f-chat-top-p" min="0" max="1" step="0.05" value="0.95"></div>' +
        '<div class="chat-param"><label>Top-K</label><input type="number" id="f-chat-top-k" min="1" max="256" step="1" value="40"></div>' +
        '</div>';
    el.appendChild(paramsPanel);

    const inputBar = document.createElement('div');
    inputBar.className = 'chat-input-bar';
    inputBar.innerHTML = '<input type="text" id="f-chat-input" placeholder="Type a message...">' +
        '<button id="btn-send-chat" title="Send">Send</button>' +
        '<span class="chat-status" id="chat-status"></span>';
    el.appendChild(inputBar);

    return el;
};

// ═══════════════════════════════════════════════════════════
// ConfigPanel - per-model edit form
// ═══════════════════════════════════════════════════════════

renderers.ConfigPanel = function({ id, fields }) {
    const el = document.createElement('div');
    el.className = 'config-panel';
    el.id = 'config-' + sanitizeId(id);

    (fields||[]).forEach(f => {
        const wrapper = document.createElement('div');
        wrapper.className = 'field-wrapper field-' + (f.schema?.type || 'string');

        const label = document.createElement('label');
        label.textContent = f.label || keyToLabel(f.name);
        wrapper.appendChild(label);

        let input;
        if (f.options) {
            // SelectInput — schema.type is ignored, explicit options provided
            input = document.createElement('select');
            for (const opt of f.options) {
                const o = document.createElement('option');
                o.value = String(opt.value); o.textContent = opt.label || opt.value;
                input.appendChild(o);
            }
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
        } else if (f.widget === 'TextArea') {
            // Explicit TextArea — multi-line
            const ta = document.createElement('textarea');
            ta.rows = 2;
            input = ta;
        } else {
            input = document.createElement('input'); input.type = 'text';
        }

        if (input) {
            input.id = 'f-' + sanitizeId(id) + '-' + f.name;
            input.value = f.value ?? '';
            wrapper.appendChild(input);
        }

        el.appendChild(wrapper);
    });

    // Action buttons row
    const bar = document.createElement('div');
    bar.className = 'model-actions';
    (window._configActions||[]).forEach(a => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.id = 'btn-' + a + '-' + sanitizeId(id);
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
    row.innerHTML = '<div class="model-name-bar"><span class="status-dot '+statusClass+'"></span>' +
        '<span class="model-name">'+esc(name)+'</span></div>';

    // Click -> expand + fetch config (deferred)
    row.addEventListener('click', async (e) => {
        if (row.querySelector('.config-panel')?.contains(e.target)) return;
        if (e.target.tagName === 'BUTTON') return;
        EventBus.emit('model.select', { modelId: model.id });
        row.classList.toggle('expanded');
        if (row.querySelector('.config-panel')) return;

        try {
            const data = await adminGet('/admin/config/' + encodeURIComponent(model.id));
            if (!data?.config) return;
            window._argSchema = data.args_schema || {};

            const cp = data.config.checkpoint || '';
            // Parse into repo + model (backwards compat: no prefix -> all in model)
            let repoVal = '', modelVal = cp;
            if (cp.includes('/')) {
                const [first, ...rest] = cp.split('/');
                if (['hf'].includes(first)) { repoVal = first; modelVal = rest.join('/'); }
            }

            const fields = [
                { name:'repo', value:repoVal, label:'Repo' },
                { name:'model', value:modelVal, label:'Model' },
            ];

            // Backend selector
            const bkOpts = Object.entries(data.backends||{}).map(([k,v]) => ({
                value: k, label: (typeof v==='object')?(v.host||k):k
            }));
            if (bkOpts.length) fields.push({ name:'backend', value:data.config.backend,
                options:bkOpts, widget:'SelectInput' });

            // Runner selector
            const rnOpts = data.runner_types?.map(t => ({value:t})) || [];
            if (rnOpts.length) {
                const all = [{value:''}]; for (const r of rnOpts) all.push(r);
                fields.push({ name:'runner', value:data.config.runner, options:all,
                    widget:'SelectInput' });
            }

            // Individual args from backend-provided schema
            for (const [k, schema] of Object.entries(data.args_schema || {})) {
                fields.push({ name:k, value:String(data.config.args?.[k] ?? ''),
                    label:k, schema:schema });
            }

            const panel = renderers.ConfigPanel({ id: model.id, fields });

            // Snapshot for dirty detection
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

// ═══════════════════════════════════════════════════════════
// Dirty detection
// ═══════════════════════════════════════════════════════════

function checkDirty(modelId, panel) {
    const snap = _configSnapshots?.[modelId];
    if (!snap) return;

    const val = (name) => {
        const el = panel.querySelector('#f-' + sanitizeId(modelId) + '-' + name);
        return el ? (el.type === 'number' ? Number(el.value) : el.value) : '';
    };

    let isDirty = false;
    const argKeys = Object.keys(window._argSchema || {});
    for (const key of argKeys) {
        if (val(key) !== (snap.args?.[key] ?? '')) { isDirty = true; break; }
    }
    if (!isDirty && val('repo') !== (snap.repo||'')) isDirty = true;
    else if (!isDirty && val('model') !== (snap.model||snap.checkpoint||'')) isDirty = true;
    else if (!isDirty && val('backend') !== snap.backend) isDirty = true;
    else if (!isDirty && val('runner') !== snap.runner) isDirty = true;

    const btn = panel.querySelector('#btn-save-' + sanitizeId(modelId));
    if (btn) btn.disabled = !isDirty;
}

// ═══════════════════════════════════════════════════════════
// Event wiring conventions
// ═══════════════════════════════════════════════════════════

function wireEvents(actions) {
    // Click delegation: id="btn-{action}-{id}" -> actions[action](id)
    document.addEventListener('click', (e) => {
        const m = e.target.id?.match(/^btn-(.+)-(.+)$/);
        if (!m) return;
        actions[m[1]]?.(m[2].replace(/_/g, '-'));
    });

    // Field change: id="f-{context}-{name}" -> fires CustomEvent('field.change')
    document.addEventListener('input', (e) => {
        const m = e.target.id?.match(/^f-(.+)-(.+)$/);
        if (!m) return;
        e.detail = { context: m[1].replace(/_/g,'-'), name: m[2], value: e.target.value };
        e.target.dispatchEvent(new CustomEvent('field.change', { bubbles:true, detail:e.detail }));
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

// ═══════════════════════════════════════════════════════════
// Log polling - delta fetch with cursor
// ═══════════════════════════════════════════════════════════

let logTimer = null, logSince = 0;

function startLogPoll(modelId) {
    stopLogPoll();

    const poll = async () => {
        try {
            if (!modelId) return;
            const params = { lines: 200 };
            if (logSince) params.since = logSince;
            const data = await adminGet('/admin/log/' + encodeURIComponent(modelId), params);
            if (!data?.lines?.length) return;

            const display = document.getElementById('log-display');
            if (!display) return;

            for (const entry of data.lines) {
                const div = document.createElement('div');
                div.className = 'log-line';
                div.textContent = entry.text;
                display.appendChild(div);
            }

            // Auto-scroll if near bottom
            if (display.scrollHeight - display.scrollTop <= display.clientHeight + 60) {
                display.scrollTop = display.scrollHeight;
            }

            logSince = data.since ?? logSince;
        } catch {}
    };

    poll();
    logTimer = setInterval(poll, POLL_INTERVAL);
}

function stopLogPoll() { if (logTimer) clearInterval(logTimer); logTimer=null; logSince=0; }

// ═══════════════════════════════════════════════════════════
// Chat streaming - SSE to model's port /v1/chat/completions
// Handles internal 'chat.send' event for self-contained operation
// ═══════════════════════════════════════════════════════════

let _streamAcc = '', _streamBubble = null;

// Start streaming chat for a given model - called by EventBus or externally
async function sendChat(modelName) {
    const textEl = document.getElementById('f-chat-input');
    if (!textEl) return;
    const text = textEl.value.trim();
    if (!text || !modelName) return;

    appendBubble('user', text);
    chatHistory.push({ role:'user', content:text });
    textEl.value = '';

    // Clean up any in-flight stream from a previous message
    if (_streamBubble && _streamBubble.classList.contains('streaming')) {
        _streamBubble.classList.remove('streaming');
    }
    const bubble = appendBubble('assistant', '', true);
    _streamAcc = '';
    _streamBubble = bubble;

    stopLogPoll();  // pause logs during stream

    const abortCtrl = new AbortController();
    try {
        await doStream(text, modelName, (delta) => {
            if (abortCtrl.signal.aborted) return; // cancelled by new message
            _streamAcc += delta;
            requestAnimationFrame(() => {
                if (_streamBubble && !abortCtrl.signal.aborted) {
                    _streamBubble.innerHTML = typeof marked !== 'undefined' ? marked.parse(_streamAcc) : _streamAcc + '<span class="cursor"></span>';
                }
            });
        }, { signal: abortCtrl.signal });

        // Done streaming
        _streamBubble?.classList.remove('streaming');
        const final = _streamAcc || (_streamBubble?.textContent||'').replace(/\u200B/g,'').trim();
        if (typeof marked !== 'undefined') _streamBubble.innerHTML = marked.parse(final);
        chatHistory.push({ role:'assistant', content:final });
    } catch(e) {
        if (e.name !== 'AbortError') appendBubble('assistant', '\u26a0 ' + e.message);
    }
}

// Internal SSE streaming logic
async function doStream(text, modelName, onData, options = {}) {
    const { signal } = options;
    const info = window._modelsCache?.find(m => m.id === modelName);
    if (!info?.port) throw new Error('Model not running');

    const params = JSON.parse(localStorage.getItem('arkestra-chat-params')||'{}')[modelName] || {};
    const resp = await fetch('http://127.0.0.1:' + info.port + '/v1/chat/completions', {
        method: 'POST', headers:{'Content-Type':'application/json'},
        signal,
        body: JSON.stringify({
            model:modelName, messages:chatHistory, stream:true,
            temperature: params.temperature??0.7,
            top_p: params.top_p??0.95,
            max_tokens: params.max_tokens??512,
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
        const lines = buf.split('\n');
        buf = lines.pop() || '';

        for (const line of lines) {
            let t = line.trim();
            if (!t || t === '[DONE]') continue;
            if (t.startsWith('data: ')) t = t.slice(6);
            try {
                const p = JSON.parse(t);
                const d = p.choices?.[0]?.delta?.content;
                if (d) onData(d);
            } catch {}
        }
    }
}

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

// Collapse all button (document-wide)
const collapseBtn = document.getElementById('btn-collapse-all');
if (collapseBtn) collapseBtn.addEventListener('click', () => {
    document.querySelectorAll('.accordion').forEach(a => a.classList.add('collapsed'));
});

// ═══════════════════════════════════════════════════════════
// API helpers - fetch with X-Admin-Key header
// ═══════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════
// Constants and globals for app.js
// ═══════════════════════════════════════════════════════════

const POLL_INTERVAL = 2000;
let chatHistory = [];
let _configSnapshots = {};

window.EventBus = EventBus;
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
