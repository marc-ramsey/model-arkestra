/* app.js — Glue: actions + data loading + init */

// BASE_URL defined in widget.js (global scope)

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

const _configSnapshots = {};
let logTimer = null, logSince = 0;

(async () => {
    console.log('app.js IIFE STARTING');
    if (_init) { console.log('app.js already initialized'); return; }
    _init = true;
    console.log('app.js initializing...');

    // Load and render JSON layout tree
    const tree = await fetch('/static/app.json').then(r => r.json());
    const root = document.getElementById('app-root');
    if (root) root.replaceWith(window.render(tree));

    // Toggle admin-key class for CSS gating
    if (HAS_ADMIN_KEY) document.documentElement.classList.add('has-admin-key');
    else document.documentElement.classList.remove('has-admin-key');

    // Actions — wired at bottom via wireEvents()
    const actions = {
        async 'spawn-chat'(modelId)   { SessionRegistry.add('chat', modelId); },
        async 'spawn-log'(modelId)   { if(HAS_ADMIN_KEY) SessionRegistry.add('log', modelId); },

        start(id)  { apiRequest('/admin/start/'+encodeURIComponent(id), {}, 'POST', {}); },
        stop(id)   { apiRequest('/admin/stop/'+encodeURIComponent(id), {}, 'POST', {}); },
        eject(id)  { apiRequest('/admin/eject/'+encodeURIComponent(id), {}, 'POST', {}); },
        pull(id)   { window.startPull(id); },
        cancel(id) { window.cancelPull(id); },
        sendTTS(t) { window._audio?.sendTTS(t); },

        async 'add-model'() {
            const name = prompt('Model name:');
            if (!name) return;
            try {
                await apiRequest('/admin/config', {}, 'POST', { model: name, name });
                showToast('Added: '+name);
                loadModels();
            } catch(e) { showToast('Add failed: '+e.message); }
        },

        async 'save-model'(id) {
            const row = document.querySelector('.model-row[data-model="'+id+'"]');
            if (!row) return;
            // Read admin section fields from the expanded config panel
            const body = {};
            row.querySelectorAll('.admin-section input, .admin-section select').forEach(el => {
                const name = el.id;
                body[name] = el.multiple ? Array.from(el.selectedOptions).map(o=>o.value) :
                    (el.type==='number' ? Number(el.value) : el.value);
            });
            try { await apiRequest('/admin/config/'+encodeURIComponent(id), {}, 'POST', body); } catch(e) {}
        },

        reset() { if(!confirm('Reset?')) return; location.reload(); },
    };

    wireEvents(actions);

    // ── Load models + populate cluster tree ───────────────────
    await loadModels();
    populateClusterTree(window._modelsCache || []);
    if (HAS_ADMIN_KEY) { try { await loadClusters(); } catch{} }

    // Periodic refresh with 401 fallback
    setInterval(async () => {
        try {
            const data = await apiRequest('/admin/models');
            window._modelsCache = data.models || [];
            window.backendOptions = data.backends || {};
            window.runnerTypes = data.runner_types || [];

            // Update model rows in cluster tree
            for (const m of data.models||[]) {
                const row = document.querySelector('.model-row[data-model="'+m.id+'"]');
                if (!row) continue;
                const dot = row.querySelector('.status-dot');
                if (dot && m.status) dot.className = 'status-dot '+window.normalizeStatus(m.status);

                // Update download progress
                const isDL = m.status?.value === 'loading';
                const progEl = row.querySelector('.pull-progress');
                if (isDL && progEl?.classList.contains('hidden')) {
                    progEl.classList.remove('hidden');
                    window.syncRowButtons(row, {state:'downloading', statusClass:'loading'});
                    window.pollPullProgress(m.id);
                } else if (!isDL) {
                    window.stopPullProgress(m.id);
                    if (progEl && !progEl.querySelector('.pull-status').textContent) progEl.classList.add('hidden');
                }

                // Size display
                const sizeEl = row.querySelector('.model-size');
                if (sizeEl && m.size_gb) sizeEl.textContent = formatSizeGB(m.size_gb);
                else if (sizeEl) sizeEl.remove();
            }
        } catch {}
    }, CFG.POLL_INTERVAL);

})();

// ── Cluster tree population ────────────────────────────────────
function populateClusterTree(models) {
    const trees = document.querySelectorAll('.cluster-tree');
    if (trees.length === 0) return;

    // Already populated?
    const firstTree = trees[0];
    if (firstTree.querySelectorAll('.acc-node').length > 1) return;

    // Group models by cluster
    const localModels = [];
    const remoteModels = {};

    for (const m of models) {
        if (m.id.includes('/')) {
            const [cluster] = m.id.split('/');
            if (!remoteModels[cluster]) remoteModels[cluster] = [];
            remoteModels[cluster].push(m);
        } else {
            localModels.push(m);
        }
    }

    // Populate each tree
    for (const tree of trees) {
        const localBody = tree._localModelBody || tree.querySelector(':scope > .acc-node:first-child .acc-body');
        const remoteContainer = tree._remoteContainer || tree.querySelector(':scope > .remote-clusters');

        // Populate Local cluster models
        if (localBody) {
            for (const m of localModels) {
                localBody.appendChild(renderModelRow(m, true));
            }
        }

        // Populate remote clusters
        if (remoteContainer) {
            for (const [clusterName, clusterModels] of Object.entries(remoteModels)) {
                const cNode = _accNode(clusterName);
                remoteContainer.appendChild(cNode.el);

                // Health status from cached data
                try {
                    const clusters = JSON.parse(localStorage.getItem('arkestra-clusters') || '[]');
                    const c = clusters.find(x => x.name === clusterName);
                    if (c) {
                        const dot = cNode.header.querySelector('.cluster-health-dot');
                        if (dot) dot.className = 'cluster-health-dot' + (c.healthy ? '' : ' unhealthy');
                    }
                } catch {}

                for (const m of clusterModels) {
                    cNode.body.appendChild(renderModelRow(m, true));
                }
            }
        }
    }
}

async function loadClusters() {
    try {
        const data = await apiRequest('/admin/clusters');
        localStorage.setItem('arkestra-clusters', JSON.stringify(data.clusters || []));
    } catch {}
}

// ── Model loader ────────────────────────────────────────────────
async function loadModels() {
    try {
        const data = await apiRequest('/admin/models');
        window._modelsCache = data.models || [];
        window.backendOptions = data.backends || {};
        window.runnerTypes = data.runner_types || [];

        // Populate all select dropdowns (per-session ChatPane and LogPane)
        document.querySelectorAll('#chat-model-select').forEach(s => populateSelect(s.id, window._modelsCache, ''));
        document.querySelectorAll('#log-model-select').forEach(s => populateSelect(s.id, window._modelsCache, ''));

        // Load personal defaults for models with open sessions
        for (const session of SessionRegistry.list()) {
            if (session.type === 'chat') await SyncStore.load(session.modelId);
        }
    } catch(e) { console.error('[app] models fetch failed:', e.message); }
}

// ── Session dock management — called on session lifecycle events ─
function updateSessionDock() {
    const sessions = SessionRegistry.list();
    const dock = document.querySelector('.session-dock');
    if (!dock) return;

    const container = dock._sessionContainer || dock.querySelector('.docked-sessions');
    if (!container) return;

    // Update count
    const countEl = dock.querySelector('.dock-count');
    if (countEl) countEl.textContent = ' (' + sessions.length + ')';

    // Clear existing session entries and re-render
    container.innerHTML = '';
    const empty = document.createElement('div');
    empty.className = 'dock-empty hidden';
    empty.textContent = 'Select a model above and click + to open a chat or log session.';
    container.appendChild(empty);

    for (const s of sessions) {
        // Build session entry — collapsible subhead with pane inside
        const entry = document.createElement('div');
        entry.className = 'session-entry';
        entry.dataset.sessionId = s.id;

        // Subhead: model name + type + close button
        const header = document.createElement('div');
        header.className = 'session-header';

        const modelName = s.modelId.includes('/') ? s.modelId.split('/').pop() : s.modelId;
        const typeIcon = s.type === 'chat' ? '💬' : '📋';
        const statusClass = normalizeStatus(getModelStatus(s.modelId));

        header.innerHTML = '<span class="status-dot '+statusClass+'"></span>' +
            '<span class="session-label">'+typeIcon+' '+esc(modelName)+'</span>' +
            '<button type="button" data-action="close-session" data-session-id="'+s.id+'" title="Close">×</button>';

        // Expand/collapse
        header.addEventListener('click', (e) => {
            if (e.target.closest('[data-action]')) return;
            const body = entry.querySelector('.session-body');
            if (!body) return;
            body.style.display = body.style.display === 'none' ? '' : 'none';
            header.classList.toggle('collapsed', body.style.display === 'none');
        });

        entry.appendChild(header);

        // Body: instantiate the actual pane (ChatPane or LogPane)
        const body = document.createElement('div');
        body.className = 'session-body';

        if (s.type === 'chat') {
            const pane = renderers.ChatPane({ sessionId: s.id, modelId: s.modelId });
            // Populate the select with available models
            populateSelect('chat-model-select-'+s.id, window._modelsCache || [], s.modelId);
            body.appendChild(pane);

            // Load personal params for this session's model
            SyncStore.load(s.modelId).then(() => {
                const paramMap = { temp:'temperature', 'max-tokens':'max_tokens', 'top-p':'top_p', 'top-k':'top_k' };
                for (const [uiKey, apiKey] of Object.entries(paramMap)) {
                    const el = document.getElementById('f-chat-'+apiKey+'-'+s.id);
                    if (!el) continue;
                    let value = SyncStore.get(s.modelId, apiKey);
                    // Server default fallback
                    try {
                        const modelInfo = window._modelsCache?.find(m => m.id === s.modelId);
                        if (modelInfo?.config?.args?.[apiKey] !== undefined) {
                            value = value ?? Number(modelInfo.config.args[apiKey]);
                        } else if (value === undefined) {
                            const fallbacks = { temperature:0.7, max_tokens:4096, top_p:0.95, top_k:40 };
                            value = fallbacks[apiKey] ?? 0.7;
                        }
                    } catch {}
                    el.value = String(value);
                }
            });

        } else if (s.type === 'log' && HAS_ADMIN_KEY) {
            const pane = renderers.LogPane();
            // Clone log-display with session-scoped id
            const origDisplay = pane.querySelector('#log-display');
            if (origDisplay) { origDisplay.id = 'log-display-'+s.id; }
            populateSelect('log-model-select', window._modelsCache || [], s.modelId);
            body.appendChild(pane);

            // Start log polling for this session
            startLogPoll(s.id, s.modelId);
        }

        entry.appendChild(body);
        container.appendChild(entry);
    }
}

function getModelStatus(modelId) {
    return (window._modelsCache || []).find(m => m.id === modelId)?.status;
}

// Wire session lifecycle to dock updates
EventBus.on('session.add', () => updateSessionDock());
EventBus.on('session.remove', () => updateSessionDock());
