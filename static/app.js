/* app.js — Glue: actions + data loading + init */

let toastTimer = null;
function showToast(msg) {
    const el = document.getElementById('toast-msg');
    if (!el) return;
    clearTimeout(toastTimer);
    el.textContent = msg;
    el.classList.remove('hidden');
    toastTimer = setTimeout(() => el.classList.add('hidden'), 4000);
}

(async () => {
    // ── Load and render JSON tree ────────────────────────────────
    const tree = await fetch('/static/app.json').then(r => r.json());
    const root = document.getElementById('app-root');
    if (root) root.replaceWith(window.render(tree));

    // ── Actions registry ─────────────────────────────────────────
    window._configActions = ['reset','stop','start','save','eject'];

    const actions = {
        async start(id)   { await window.adminPost('/admin/start/'+encodeURIComponent(id), {}); },
        async stop(id)    { await window.adminPost('/admin/stop/'+encodeURIComponent(id), {}); },
        async eject(id)   { try { await window.adminPost('/admin/eject/'+encodeURIComponent(id), {}); } catch(e) { console.error('[app] eject:',e.message); } },

        sendChat() {
            const modelId = document.getElementById('chat-model-select')?.value;
            const textEl = document.getElementById('f-chat-input');
            if (!modelId || !textEl?.value.trim()) return;
            window.sendChat(modelId, textEl);
        },

        toggleChatParams() {
            document.getElementById('chat-params-panel')?.classList.toggle('open');
        },

        async save(id) {
            const panel = document.getElementById('config-' + sanitizeId(id));
            if (!panel) return;
            const body = { args: {} };
            for (const w of panel.querySelectorAll('.field-wrapper')) {
                const el = w.querySelector('input, select');
                if (!el) continue;
                const m = el.id.match(/^f-(.+?)-(?:repo|model|backend|runner|checkpoint|max-tokens|top-p|top-k)$/);
                const name = m ? m[1].replace(/_/g, '-') : null;
                if (!name) continue;
                if (name === 'repo') { body.repo = el.value; continue; }
                if (name === 'model') {
                    const repo = body.repo || '';
                    body.checkpoint = repo ? repo + '/' + el.value : el.value;
                    continue;
                }
                const isArg = window._argSchema?.hasOwnProperty(name);
                if (isArg) {
                    body.args[name] = el.type === 'number' ? Number(el.value) : el.value;
                } else {
                    body[name] = el.value;
                }
            }
            try {
                await window.adminPost('/admin/config/'+encodeURIComponent(id), body);
                window._configSnapshots[id] = { ...body };
            } catch(e) {
                showToast('Save failed: ' + e.message);
            }
        },

        async reset() {
            if (!confirm('Reset the whole page?')) return;
            location.reload();
        },
    };

    // Wire Enter key on chat input to sendChat action
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.target.id === 'f-chat-input' && !e.shiftKey) {
            e.preventDefault();
            actions.sendChat();
        }
    });

    // Clear chat history when model changes
    const sel = document.getElementById('chat-model-select');
    if (sel) {
        sel.addEventListener('change', (e) => {
            chatHistory = [];
            EventBus.emit('model.select', { modelId: e.target.value });
        });
    }

    window.wireEvents(actions);

    // Toggle chat params panel (one-time registration, not per-render)
    document.addEventListener('click', (e) => {
        if (e.target.id === 'btn-toggle-chat-params') {
            document.getElementById('chat-params-panel')?.classList.toggle('open');
        }
    });

    // Save chat params to localStorage on change
    document.addEventListener('input', (e) => {
        if (!e.target.id?.startsWith('f-chat-')) return;
        const name = e.target.id.replace('f-chat-', '');
        const key = 'arkestra-chat-params';
        try {
            const params = JSON.parse(localStorage.getItem(key)||'{}');
            const chatModelSel = document.getElementById('chat-model-select');
            const modelName = chatModelSel?.value;
            if (!modelName) return;
            if (!params[modelName]) params[modelName] = {};
            // Map widget names to API field names
            const apiName = name === 'temp' ? 'temperature' :
                            name === 'max-tokens' ? 'max_tokens' :
                            name === 'top-p' ? 'top_p' :
                            name === 'top-k' ? 'top_k' : name;
            params[modelName][apiName] = e.target.type === 'number' ? Number(e.target.value) : e.target.value;
            localStorage.setItem(key, JSON.stringify(params));
        } catch {}
    });

    // ── Load model data and populate tree ────────────────────────
    try {
        const data = await window.adminGet('/admin/models');
        window._modelsCache = data.models || [];
        window.backendOptions = data.backends || {};
        window.runnerTypes = data.runner_types || [];

        // Populate dropdowns
        window.populateSelect('log-model-select', window._modelsCache, '');
        window.populateSelect('chat-model-select', window._modelsCache, '');

        // Insert model rows into accordion body
        const accBody = document.getElementById('model-accordion-items');
        if (accBody) {
            for (const m of window._modelsCache) {
                accBody.appendChild(window.renderModelRow(m));
            }
        }
    } catch(e) { console.error('[app] models fetch failed:', e.message); }

    // ── Cross-widget wiring ──────────────────────────────────────
    EventBus.on('model.select', ({ modelId }) => {
        if (modelId) window.startLogPoll(modelId);
        else window.stopLogPoll();
    });

    // ── Periodic status refresh ──────────────────────────────────
    setInterval(async () => {
        try {
            const data = await window.adminGet('/admin/models');
            for (const m of (data?.models||[])) {
                const row = document.querySelector('.model-row[data-model="'+m.id+'"]');
                if (!row) continue;
                const dot = row.querySelector('.status-dot');
                if (dot && m.status) dot.className = 'status-dot ' + window.normalizeStatus(m.status);
            }
        } catch {}
    }, POLL_INTERVAL);

})();
