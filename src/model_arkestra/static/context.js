/* context.js — contexts: tabs, streaming, persistence, model select.
 *
 * A context = one conversation: { id, model, name, params, history[] }.
 * State lives here (store + IndexedDB), never in the DOM. Tabs are views.
 * A context's stream keeps running while another tab is active.
 */

const DEFAULT_SYSTEM = 'You are a helpful assistant.';
const DEFAULT_PARAMS = { temperature: 0.7, top_p: 0.95, max_tokens: 4096 };
const PARAM_LABELS = [
    ['temperature', 'Temp'],
    ['top_p', 'Top-P'],
    ['max_tokens', 'Max Tokens'],
];
const UNNAMED = 'untitled';
const NAME_MAX = 40;

// ── Create / close ─────────────────────────────────────────────────

function newContext(modelName) {
    const id = store.nextId();
    const rec = {
        id,
        model: modelName || (store.active()?.model) || store.models[0]?.name || '',
        name: UNNAMED,
        named: false,          // auto-name once the first user message lands
        params: { ...DEFAULT_PARAMS },
        history: [{ role: 'system', content: DEFAULT_SYSTEM }],
        streaming: false,
        error: null,
        modelDead: false,
        abort: null,
    };
    store.contexts.set(id, rec);
    store.setActive(id);
    renderTabs();
    renderPane();
    return rec;
}

function closeContext(id) {
    const rec = store.contexts.get(id);
    if (!rec) return;
    rec.abort?.abort();
    store.contexts.delete(id);
    if (store.activeId === id) {
        const remaining = [...store.contexts.keys()];
        store.setActive(remaining.length ? remaining[remaining.length - 1] : null);
    }
    renderTabs();
    renderPane();
}

function openSavedList() {
    const saved = window.db.list().reverse();   // newest first
    if (!saved.length) { showToast('No saved conversations'); return; }

    const list = document.createElement('div');
    list.className = 'saved-list hidden';
    for (const s of saved) {
        const row = document.createElement('div');
        row.className = 'saved-row';
        row.innerHTML = `<span class="saved-name">${esc(s.name || UNNAMED)}</span>` +
            `<span class="saved-meta">${esc(s.model)} · ${s.history.length - 1} msgs</span>`;
        row.addEventListener('click', async () => {
            list.remove();
            const rec = await loadSaved(s);
            store.contexts.set(rec.id, rec);
            store.setActive(rec.id);
            renderTabs();
            renderPane();
        });
        list.appendChild(row);
    }
    const wrap = document.getElementById('saved-wrap');
    wrap.appendChild(list);
    list.classList.remove('hidden');
    list.addEventListener('click', (e) => {
        if (e.target === list) list.remove();
    });
}

async function loadSaved(saved) {
    const rec = {
        id: saved.id,
        model: saved.model,
        name: saved.name || UNNAMED,
        named: saved.name !== undefined && saved.name !== UNNAMED,
        params: { ...DEFAULT_PARAMS, ...(saved.params || {}) },
        history: saved.history?.length
            ? saved.history
            : [{ role: 'system', content: DEFAULT_SYSTEM }],
        streaming: false,
        error: null,
        modelDead: false,
        abort: null,
    };
    return rec;
}

// ── Send + stream ──────────────────────────────────────────────────

async function sendActive() {
    const rec = store.active();
    const input = document.getElementById('chat-input');
    if (!rec || rec.streaming) return;
    const text = input?.value.trim();
    if (!text) return;

    if (!rec.model) { showToast('Pick a model'); return; }
    if (rec.modelDead) { showToast(rec.model + ' is not available'); return; }

    input.value = '';
    rec.error = null;
    rec.modelDead = false;
    rec.history.push({ role: 'user', content: text });
    if (!rec.named) nameFrom(text);

    // Save point 1: user message persisted before generation starts.
    await persist(rec);

    const bubbleEl = appendBubble('assistant', '', true);
    rec.streaming = true;
    rec.abort = new AbortController();
    renderTabs();

    let acc = '';
    const onDelta = (delta) => {
        acc += delta;
        if (store.activeId === rec.id) {
            bubbleEl.innerHTML = md(acc);
            scrollChat();
        }
    };

    try {
        await api.streamChat(rec.model, rec.history, rec.params, onDelta, rec.abort.signal);
        rec.history.push({ role: 'assistant', content: acc });
        if (store.activeId === rec.id) {
            bubbleEl.classList.remove('streaming');
            bubbleEl.innerHTML = md(acc || '(empty response)');
            scrollChat();
        }
    } catch (e) {
        rec.error = e.message;
        if (e.name === 'AbortError') {
            // User aborted: keep what arrived; drop it if empty.
            if (acc) rec.history.push({ role: 'assistant', content: acc });
            if (store.activeId === rec.id) {
                bubbleEl.classList.remove('streaming');
                bubbleEl.innerHTML = md(acc || '⊘ aborted');
            }
        } else {
            if (e.status === 503) rec.modelDead = true;
            rec.history.pop();      // failed send — drop the user message
            if (store.activeId === rec.id) {
                bubbleEl.classList.remove('streaming');
                bubbleEl.innerHTML = '⚠ ' + esc(e.message);
            }
        }
    } finally {
        rec.streaming = false;
        rec.abort = null;
        renderTabs();
        renderBanner(rec);
        renderModelSelect(rec);
    }

    // Save point 2: stream finished (or failed) — persist final state.
    await persist(rec);
}

function stopActive() {
    store.active()?.abort?.abort();
}

// ── Naming ─────────────────────────────────────────────────────────

function nameFrom(text) {
    const rec = store.active();
    if (!rec) return;
    const clean = text.replace(/\s+/g, ' ').trim();
    rec.name = clean.length > NAME_MAX ? clean.slice(0, NAME_MAX - 1) + '…' : clean;
    rec.named = true;
    renderTabs();
}

function renameActive() {
    const rec = store.active();
    if (!rec) return;
    const name = prompt('Conversation name:', rec.name);
    if (!name?.trim()) return;
    rec.name = name.trim().slice(0, NAME_MAX);
    rec.named = true;
    renderTabs();
}

// ── Persistence ────────────────────────────────────────────────────

async function persist(rec) {
    try {
        await window.db.put({
            id: rec.id, model: rec.model, name: rec.name,
            params: rec.params, history: rec.history, updated: Date.now(),
        });
    } catch (e) { console.warn('[context] persist failed:', e); }
}

function newConversation() {
    const rec = store.active();
    if (!rec) return;
    if (rec.streaming) return;
    rec.history = [{ role: 'system', content: DEFAULT_SYSTEM }];
    rec.params = { ...DEFAULT_PARAMS };
    rec.name = UNNAMED;
    rec.named = false;
    rec.modelDead = false;
    rec.error = null;
    window.db.del(rec.id);
    renderPane();
    renderTabs();
    showToast('New conversation (model kept)');
}

// ── Rendering ──────────────────────────────────────────────────────

function renderTabs() {
    const strip = document.getElementById('tab-strip');
    strip.innerHTML = '';
    for (const rec of store.contexts.values()) {
        const tab = document.createElement('button');
        tab.type = 'button';
        tab.className = 'tab' + (rec.id === store.activeId ? ' active' : '');
        const dot = rec.streaming ? 'streaming' : (rec.modelDead ? 'dead' : '');
        tab.innerHTML = `<span class="dot ${dot}"></span><span class="tab-name">${esc(rec.name)}</span>` +
            `<span class="tab-close" title="Close (conversation is saved)">×</span>`;
        tab.addEventListener('click', (e) => {
            if (e.target.closest('.tab-close')) { closeContext(rec.id); return; }
            store.setActive(rec.id);
            renderTabs();
            renderPane();
        });
        tab.addEventListener('dblclick', renameActive);
        strip.appendChild(tab);
    }
}

function renderPane() {
    const rec = store.active();
    const body = document.getElementById('chat-body');
    body.innerHTML = '';
    if (!rec) {
        // No active context — clear all context chrome.
        document.getElementById('pane-empty').classList.remove('hidden');
        document.getElementById('model-select').innerHTML = '';
        document.getElementById('params').innerHTML = '';
        document.getElementById('input-bar').innerHTML = '';
        document.getElementById('banner').classList.add('hidden');
        return;
    }
    document.getElementById('pane-empty').classList.add('hidden');
    renderBanner(rec);
    for (const m of rec.history) {
        if (m.role === 'system') continue;
        appendBubble(m.role, m.content, false, true);
    }
    renderModelSelect(rec);
    renderParams(rec);
    renderInput(rec);
    scrollChat();
}

function appendBubble(role, content, streaming, raw) {
    const body = document.getElementById('chat-body');
    const div = document.createElement('div');
    div.className = 'msg ' + role + (streaming ? ' streaming' : '');
    if (role === 'user') div.textContent = content;
    else div.innerHTML = raw ? esc(content) : md(content);
    body.appendChild(div);
    return div;
}

function renderModelSelect(rec) {
    const sel = document.getElementById('model-select');
    const opts = store.models.map(m => `<option value="${esc(m.name)}">${esc(m.name)}</option>`)
        .join('');
    sel.innerHTML = opts;
    sel.value = rec.model;
    sel.disabled = rec.streaming;
}

function renderParams(rec) {
    const wrap = document.getElementById('params');
    wrap.innerHTML = '';
    for (const [key, label] of PARAM_LABELS) {
        const div = document.createElement('label');
        div.className = 'param';
        div.innerHTML = `${label} <input type="number" step="any" data-param="${key}"
            value="${rec.params[key] ?? DEFAULT_PARAMS[key]}">`;
        div.querySelector('input').addEventListener('change', (e) => {
            rec.params[key] = Number(e.target.value);
            persist(rec);
        });
        wrap.appendChild(div);
    }
}

function renderInput(rec) {
    const bar = document.getElementById('input-bar');
    bar.innerHTML = '';
    const ta = document.createElement('textarea');
    ta.id = 'chat-input';
    ta.rows = 2;
    ta.placeholder = 'Type a message… (Enter to send, Shift+Enter for newline)';
    ta.disabled = rec.streaming;

    const btns = document.createElement('div');
    btns.className = 'input-btns';
    const send = document.createElement('button');
    send.textContent = rec.streaming ? 'Stop' : 'Send';
    send.addEventListener('click', () => rec.streaming ? stopActive() : sendActive());
    const fresh = document.createElement('button');
    fresh.textContent = 'New';
    fresh.title = 'Clear this conversation (same model)';
    fresh.disabled = rec.streaming;
    fresh.addEventListener('click', newConversation);
    btns.append(send, fresh);

    ta.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!rec.streaming) sendActive();
        }
    });
    bar.append(ta, btns);
    ta.focus();
}

function renderBanner(rec) {
    const b = document.getElementById('banner');
    if (rec.modelDead) {
        b.textContent = `⚠ ${rec.model} is not available — pick another model`;
        b.classList.remove('hidden');
    } else if (rec.error) {
        b.textContent = '⚠ ' + rec.error;
        b.classList.remove('hidden');
    } else {
        b.classList.add('hidden');
    }
}

function scrollChat() {
    const body = document.getElementById('chat-body');
    body.scrollTop = body.scrollHeight;
}

// ── Markdown ───────────────────────────────────────────────────────

function md(text) {
    if (typeof marked !== 'undefined') return marked.parse(text);
    return esc(text).replace(/\n/g, '<br>');
}

// ── Toast ──────────────────────────────────────────────────────────

let _toastTimer = null;
function showToast(msg) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.classList.remove('hidden');
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => el.classList.add('hidden'), 4000);
}

// ── Shared utils ───────────────────────────────────────────────────

function esc(s) {
    return (s ?? '').toString()
        .replace(/&/g, '&amp;').replace(/"/g, '&quot;')
        .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

window.context = {
    newContext, closeContext, loadSaved, openSavedList, sendActive, stopActive,
    nameFrom, renameActive, persist, newConversation,
    renderTabs, renderPane, appendBubble, showToast,
};
