/* app.js — Glue: actions + data loading + init */

const BASE_URL = "{{BASE_URL}}" || "";

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

// ── Group containers (AccGroup return values) ───────────────────
let _readyGroup = null, _downloadGroup = null;

(async () => {
    if (_init) return;
    _init = true;

    // Load and render JSON layout tree
    const tree = await fetch('/static/app.json').then(r => r.json());
    const root = document.getElementById('app-root');
    if (root) root.replaceWith(window.render(tree));

    // Actions registry — wired at bottom via wireEvents()
    window._configActions = ['reset','stop','start','save','eject'];

    const actions = {
        async start(id)   { try { await window.adminPost('/admin/start/'+encodeURIComponent(id), {}); } catch(e) { showToast('Start failed: '+e.message); } },
        async stop(id)    { try { await window.adminPost('/admin/stop/'+encodeURIComponent(id), {}); } catch(e) { showToast('Stop failed: '+e.message); } },
        async eject(id)   { try { await window.adminPost('/admin/eject/'+encodeURIComponent(id), {}); } catch(e) { console.error('[app] eject:',e.message); } },

        async save(id) {
            const panel = document.getElementById('config-' + sanitizeId(id));
            if (!panel) return;
            const body = {};
            for (const fv of panel.querySelectorAll('.field-value')) {
                const el = fv.querySelector('input, select');
                if (!el) continue;
                const name = el.id;
                if (el.multiple) {
                    body[name] = Array.from(el.selectedOptions).map(o => o.value);
                } else if (el.type === 'number') {
                    body[name] = Number(el.value);
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

        async pull(id)  { await window.startPull(id); },
        async cancel(id){ await window.cancelPull(id); },

        async sendTTS(text) { await window._audio?.sendTTS(text); },
    };

    wireEvents(actions);

    // Load model data and populate tree
    try {
        const data = await window.adminGet('/admin/models');
        window._modelsCache = data.models || [];
        window.backendOptions = data.backends || {};
        window.runnerTypes = data.runner_types || [];

        window.populateSelect('log-model-select', window._modelsCache, '');
        window.populateSelect('chat-model-select', window._modelsCache, '');

        const accBody = document.getElementById('model-accordion-items');
        if (!accBody) return;

        // Create two AccGroups: Ready + Needs Download
        _readyGroup = renderers.AccGroup({ title: 'Ready', group: 'ready' });
        _downloadGroup = renderers.AccGroup({ title: 'Needs Download', group: 'download' });
        accBody.appendChild(_readyGroup.element);
        accBody.appendChild(_downloadGroup.element);

        // Place model rows in correct groups
        for (const m of window._modelsCache) {
            const row = window.renderModelRow(m);
            const target = (m.status?.value === 'cached') ? _readyGroup.body : _downloadGroup.body;
            if (target) target.appendChild(row);
        }

        // Update count badges
        updateGroupCounts();
    } catch(e) { console.error('[app] models fetch failed:', e.message); }

    // Cross-widget wiring: model select triggers log polling
    EventBus.on('model.select', ({ modelId }) => {
        if (modelId) window.startLogPoll(modelId);
        else window.stopLogPoll();
    });

    // Periodic status refresh — also handles group migration
    setInterval(async () => {
        try {
            const data = await window.adminGet('/admin/models');
            window._modelsCache = data.models || [];
            for (const m of data.models||[]) {
                const row = document.querySelector('.model-row[data-model="'+m.id+'"]');
                if (!row) continue;
                const dot = row.querySelector('.status-dot');
                if (dot && m.status) dot.className = 'status-dot ' + window.normalizeStatus(m.status);

                // Handle group migration: download → ready after pull completes
                const currentGroup = m.status?.value === 'cached' ? _readyGroup : _downloadGroup;
                if (currentGroup && currentGroup.body !== row.parentNode) {
                    currentGroup.body.appendChild(row);
                    window.syncRowButtons(row, { state: window.normalizeStatus(m.status) });
                    updateGroupCounts();
                }

                // Handle downloading → start progress poll
                const isDownloading = m.status?.value === 'loading';
                const progEl = row.querySelector('.pull-progress');
                if (isDownloading && progEl?.classList.contains('hidden')) {
                    progEl.classList.remove('hidden');
                    window.syncRowButtons(row, { state: 'downloading', statusClass: 'loading' });
                    pollPullProgress(m.id);
                } else if (!isDownloading) {
                    stopPullProgress(m.id);
                    if (progEl && !progEl.querySelector('.pull-status').textContent) {
                        progEl.classList.add('hidden');
                    }
                }
            }
        } catch {}
    }, CFG.POLL_INTERVAL);

})(); // end IIFE

function updateGroupCounts() {
    const ready = _readyGroup?.body?.querySelectorAll('.model-row')?.length || 0;
    const download = _downloadGroup?.body?.querySelectorAll('.model-row')?.length || 0;
    const rBadge = _readyGroup?.header?.querySelector('.acc-group-count');
    const dBadge = _downloadGroup?.header?.querySelector('.acc-group-count');
    if (rBadge) rBadge.textContent = '(' + ready + ')';
    if (dBadge) dBadge.textContent = '(' + download + ')';
}

window._audioState = { playing: false, currentSec: 0, durationSec: 0 };
