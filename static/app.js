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

        // ── TTS: send text to TTS model and play audio ───
        async sendTTS() {
            const textEl = document.getElementById('f-chat-input');
            if (!textEl?.value.trim()) return;
            const statusEl = document.getElementById('chat-status');
            statusEl.textContent = 'Synthesizing...';

            try {
                // Determine TTS model — use the chat model or fall back to a known TTS type
                const chatModelSel = document.getElementById('chat-model-select');
                const modelName = chatModelSel?.value || window.CFG?.DEFAULT_TTS;
                
                // Call speech endpoint with text input
                const resp = await fetch(window.location.origin + '/v1/audio/speech', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: modelName, input: textEl.value.trim() }),
                });

                if (!resp.ok) throw new Error('TTS API ' + resp.status);

                const blob = await resp.blob();
                const audioUrl = URL.createObjectURL(blob);
                playAudioFromUrl(audioUrl);
            } catch(e) {
                statusEl.textContent = 'TTS error: ' + e.message;
                setTimeout(() => { statusEl.textContent = ''; }, 3000);
            }
        },

        // ── ASR: upload audio file and get transcription ───
        async sendASR(file) {
            const modelSel = document.getElementById('asr-model-select');
            const modelName = modelSel?.value || window.CFG?.DEFAULT_WHISPER;
            const resultDiv = document.getElementById('asr-result-display');

            if (!file) { showToast('No audio file selected'); return; }
            if (!resultDiv) return;

            resultDiv.innerHTML = '<div class="asr-status">Transcribing...</div>';

            try {
                const formData = new FormData();
                formData.append('model', modelName);
                formData.append('file', file);

                const resp = await fetch(window.location.origin + '/v1/audio/transcriptions', {
                    method: 'POST',
                    body: formData,
                });

                if (!resp.ok) throw new Error('ASR API ' + resp.status);

                const data = await resp.json();
                resultDiv.innerHTML = '<div class="asr-text">' + escapeHtml(data.text || '') + '</div>' +
                    (data.language ? '<div class="asr-lang">Language: ' + data.language + '</div>' : '');
            } catch(e) {
                resultDiv.innerHTML = '<div class="asr-status" style="color:var(--red)">Error: ' + e.message + '</div>';
            }
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
    window._actions = actions;  // expose for audio wiring

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
        const key = window.CFG?.STORAGE_CHAT_PARAMS;
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
    }, POLL_INTERVAL);

})(); // end IIFE

// ── Audio playback helpers (called from within IIFE via window.*) ──
let audioEl = null;
let _onLoadedMetadata = null, _onTimeUpdate = null, _onEnded = null;

// Expose for widget.js renderers
window.playAudioFromUrl = playAudioFromUrl;
Object.defineProperty(window, 'audioEl', { get: () => audioEl, configurable: true });

function escapeHtml(s) {
    return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function formatTime(sec) {
    if (!sec || isNaN(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
}

function playAudioFromUrl(url) {
    // Stop any existing playback — detach its handlers to prevent stale fire
    if (audioEl) { audioEl.pause(); URL.revokeObjectURL(audioEl.src); }

    // Remove handlers from previous audioEl so closures don't fire on it
    if (audioEl && typeof audioEl.removeEventListener === 'function') {
        if (_onLoadedMetadata) audioEl.removeEventListener('loadedmetadata', _onLoadedMetadata);
        if (_onTimeUpdate)     audioEl.removeEventListener('timeupdate',     _onTimeUpdate);
        if (_onEnded)          audioEl.removeEventListener('ended',         _onEnded);
    }

    audioEl = new Audio(url);
    const bar = document.getElementById('audio-playback-bar');
    const progress = document.getElementById('audio-progress');
    const durEl = document.getElementById('audio-duration');
    const currEl = document.getElementById('audio-current');

    if (!bar || !progress) return;
    bar.classList.remove('hidden');

    // Named closures — stored for cleanup on next play()
    _onLoadedMetadata = () => {
        durEl.textContent = formatTime(audioEl.duration);
        progress.max = 100;
    };
    _onTimeUpdate = () => {
        if (!audioEl?.duration) return;
        const pct = (audioEl.currentTime / audioEl.duration) * 100;
        progress.value = pct;
        currEl.textContent = formatTime(audioEl.currentTime);
    };
    _onEnded = () => {
        audioEl = null;
        bar.classList.add('hidden');
    };

    audioEl.addEventListener('loadedmetadata', _onLoadedMetadata);
    audioEl.addEventListener('timeupdate',     _onTimeUpdate);
    audioEl.addEventListener('ended',         _onEnded);

    // Progress seek — bound once in widget.js ChatPane renderer

    audioEl.play();
}

