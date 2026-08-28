/* app.js — Glue: actions + data loading + init */

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
        reset(id)         { location.reload(); },
        async eject(id)       { try { await window.adminPost('/admin/eject/'+encodeURIComponent(id), {}); } catch(e) { console.error('[app] eject:',e.message); } },

        async save(id) {
            const panel = document.getElementById('config-' + sanitizeId(id));
            if (!panel) return;
            const snap = window._configSnapshots?.[id];
            const body = {};
            // Read fields from DOM
            for (const w of panel.querySelectorAll('.field-wrapper')) {
                const el = w.querySelector('input, select');
                if (!el) continue;
                const name = el.id.match(/f-[^-]+-(.+)$/)?.[1]?.replace(/_/g,'-');
                if (name && name !== 'args') body[name] = el.value;
            }
            const ta = panel.querySelector('textarea');
            if (ta) { try { body.args = JSON.parse(ta.value); } catch(e) { body.args = ta.value; } }

            await window.adminPost('/admin/config/'+encodeURIComponent(id), body);
            window._configSnapshots[id] = {...body};
        },
    };

    window.wireEvents(actions);

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
