/* api.js — transport layer: Bearer auth + the two public endpoints. */

const API_KEY_LS = 'arkestra-api-key';

const api = {
    base: (window.BASE_URL || '').replace(/^\/+|\/+$/g, ''),

    /* Key sources: server-rendered (Jinja), else localStorage, else prompt on 401. */
    key: window.API_KEY || '',

    ensureKey() {
        if (!this.key) {
            const stored = localStorage.getItem(API_KEY_LS);
            if (stored) this.key = stored;
        }
        return this.key;
    },

    _promptKey() {
        const k = prompt('API key required (401). Enter the server API key:');
        if (k) {
            this.key = k;
            localStorage.setItem(API_KEY_LS, k);
        }
        return this.key;
    },

    async request(method, path, body) {
        let resp = await this._fetch(method, path, body);
        if (resp.status === 401) {
            // First 401 with no key: prompt. Repeated 401 (wrong key stored):
            // clear it and prompt again — don't lock the user out.
            this.key = '';
            localStorage.removeItem(API_KEY_LS);
            this._promptKey();
            resp = await this._fetch(method, path, body);
        }
        if (!resp.ok) {
            const text = await resp.text().catch(() => '');
            const err = new Error(`HTTP ${resp.status}${text ? ': ' + text : ''}`);
            err.status = resp.status;
            throw err;
        }
        return resp.json();
    },

    _headers() {
        const h = { 'Accept': 'application/json' };
        const key = this.ensureKey();
        if (key) h['Authorization'] = `Bearer ${key}`;
        return h;
    },

    _fetch(method, path, body) {
        const opts = { method, headers: this._headers() };
        if (body !== undefined) {
            opts.headers['Content-Type'] = 'application/json';
            opts.body = JSON.stringify(body);
        }
        return fetch(this.base + path, opts);
    },

    /* GET /api/models — cached models: {name, model, size}[] */
    async listModels() {
        const data = await this.request('GET', '/api/models');
        return data.models || [];
    },

    /* Streaming chat. Calls onDelta(text) per token; resolves with nothing.
     * History is sent in full on every request (server is stateless). */
    async streamChat(model, messages, params, onDelta, signal) {
        const body = { model, messages, stream: true, ...params };
        const resp = await fetch(this.base + '/v1/chat/completions', {
            method: 'POST',
            headers: this._headers(),
            body: JSON.stringify(body),
            signal,
        });
        if (!resp.ok) {
            const text = await resp.text().catch(() => '');
            const err = new Error(`Chat API ${resp.status}${text ? ': ' + text : ''}`);
            err.status = resp.status;
            throw err;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            let idx;
            while ((idx = buf.indexOf('\n')) >= 0) {
                const line = buf.slice(0, idx).trim();
                buf = buf.slice(idx + 1);
                if (!line || line === 'data: [DONE]') continue;
                const payload = line.startsWith('data: ') ? line.slice(6) : line;
                let chunk;
                try { chunk = JSON.parse(payload); } catch { continue; }
                if (chunk.error) throw new Error(chunk.error.message || chunk.error);
                const delta = chunk.choices?.[0]?.delta?.content;
                if (delta) onDelta(delta);
            }
        }
    },
};

window.api = api;
