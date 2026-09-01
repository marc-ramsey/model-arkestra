/* app.js — Glue: actions + data loading + init */

let _init = false;
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
    if (_init) return;
    _init = true;

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

        async save(id) {
            const panel = document.getElementById('config-' + sanitizeId(id));
            if (!panel) return;
            const body = { args: {} };
            for (const fv of panel.querySelectorAll('.field-value')) {
                const el = fv.querySelector('input, select');
                if (!el) continue;
                const name = el.id;
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

        async sendTTS(text) { await window._audio?.sendTTS(text); },
        async sendASR(file) { await window._audio?.sendASR(file); },
    };

    // ── Load model data and populate tree ────────────────────────
    try {
        const data = await window.adminGet('/admin/models');
        window._modelsCache = data.models || [];
        window.backendOptions = data.backends || {};
        window.runnerTypes = data.runner_types || [];

        // Populate dropdowns
        window.populateSelect('log-model-select', window._modelsCache, '');
        window.populateSelect('chat-model-select', window._modelsCache, '');
        window.populateSelect('asr-model-select', window._modelsCache, '');

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
    }, CFG.POLL_INTERVAL);

})(); // end IIFE

// ── Audio playback helpers (called from within IIFE via window.*) ──
window._audioState = { playing: false, currentSec: 0, durationSec: 0 };

