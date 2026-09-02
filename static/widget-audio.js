/* widget-audio.js — TTS, ASR, audio playback */

// ── Streaming AudioStream class (WebSocket for live mic + TTS) ──
class AudioStream {
    constructor() {
        this._ws = null;
        this.onPartialTranscript = null;  // partial text from streaming ASR
        this.onFinalTranscript = null;    // final sentence transcription
        this._pcmBuffer = [];
        this._audioCtx = null;
    }

    async connect() {
        if (this._ws?.readyState === WebSocket.OPEN) return;
        const ws = new WebSocket(`${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ark/audio/stream`);
        ws.binaryType = 'arraybuffer';

        ws.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                this._pcmBuffer.push(new Uint8Array(event.data));
                this._drainPCM();
            } else {
                const msg = JSON.parse(event.data);
                if (msg.type === 'partial')   this.onPartialTranscript?.(msg.text);
                if (msg.type === 'final')     this.onFinalTranscript?.(msg.text);
                if (msg.type === 'error')     window.showToast?.('Audio error: ' + msg.message);
            }
        };

        return new Promise((resolve, reject) => {
            ws.onopen = () => { this._ws = ws; resolve(); };
            ws.onerror = (e) => { this._ws = null; reject(e); };
        });
    }

    _sendJson(obj) {
        if (this._ws?.readyState === WebSocket.OPEN) this._ws.send(JSON.stringify(obj));
    }

    async speak(text) {
        await this.connect();
        this._sendJson({ type: 'tts', text });
    }

    async startRecording() {
        const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this._mediaStream = mediaStream;
        this._sampleRate = 16000;

        // Sample audio at 16kHz for sherpa-ai
        const ctx = new AudioContext();
        const source = ctx.createMediaStreamSource(mediaStream);
        const processor = ctx.createScriptProcessor(4096, 1, 1);
        this._audioCtx = ctx;

        let buffer = [];
        const FRAME_SAMPLES = Math.round(this._sampleRate * 0.05); // 50ms chunks

        processor.onaudioprocess = (e) => {
            const input = e.inputBuffer.getChannelData(0);
            for (let i = 0; i < input.length; i++) buffer.push(input[i]);

            while (buffer.length >= FRAME_SAMPLES) {
                const chunk = buffer.splice(0, FRAME_SAMPLES);
                // Send as base64-encoded JSON frame (uniform protocol)
                let base64 = '';
                for (let j = 0; j < chunk.length; j += 128) {
                    const slice = chunk.slice(j, j + 128);
                    base64 += btoa(String.fromCharCode(...slice));
                }
                this._sendJson({ type: 'audio_frame', data: base64 });
            }
        };

        source.connect(processor); // Don't connect to destination — we only need input
    }

    async stopRecording() {
        // Wait briefly for any in-flight results
        await new Promise(r => setTimeout(r, 200));
        if (this._audioCtx) { this._audioCtx.close(); this._audioCtx = null; }
        this._mediaStream?.getTracks().forEach(t => t.stop());
    }

    // ── PCM playback queue ────────────────────────────────────────

    _drainPCM() {
        if (this._pcmBuffer.length === 0) return;

        const buffer = this._pcmBuffer.shift();
        if (!buffer || buffer.length < 78) return; // WAV header is 44 bytes + min audio

        try {
            if (!this._audioCtx) this._audioCtx = new AudioContext();
            const arrayBuf = new ArrayBuffer(buffer.length);
            new Uint8Array(arrayBuf).set(buffer);
            this._audioCtx.decodeAudioData(arrayBuf, (audioBuffer) => {
                const source = this._audioCtx.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(this._audioCtx.destination);
                source.start(0);
            });
        } catch(e) { /* decode failed — skip chunk */ }
    }
}

window._audioStream = new AudioStream();

// ── Playback state ────────────────────────────────────────────────
let _audioEl = null;
let _onLoadedMetadata = null, _onTimeUpdate = null, _onEnded = null;

function formatTime(sec) {
    if (!sec || isNaN(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
}


// ── Audio playback engine ────────────────────────────────────────
function playAudioFromUrl(url) {
    // Stop existing playback — detach handlers to prevent stale fire
    if (_audioEl) { _audioEl.pause(); URL.revokeObjectURL(_audioEl.src); }
    if (_audioEl && typeof _audioEl.removeEventListener === 'function') {
        if (_onLoadedMetadata) _audioEl.removeEventListener('loadedmetadata', _onLoadedMetadata);
        if (_onTimeUpdate)     _audioEl.removeEventListener('timeupdate',     _onTimeUpdate);
        if (_onEnded)          _audioEl.removeEventListener('ended',         _onEnded);
    }

    _audioEl = new Audio(url);
    const bar  = document.getElementById('audio-playback-bar');
    const prog = document.getElementById('audio-progress');
    const durEl = document.getElementById('audio-duration');
    const currEl = document.getElementById('audio-current');

    if (!bar || !prog) return;
    bar.classList.remove('hidden');

    _onLoadedMetadata = () => { durEl.textContent = formatTime(_audioEl.duration); prog.max = 100; };
    _onTimeUpdate     = () => {
        if (!_audioEl?.duration) return;
        const pct = (_audioEl.currentTime / _audioEl.duration) * 100;
        prog.value = pct; currEl.textContent = formatTime(_audioEl.currentTime);
    };
    _onEnded = () => { _audioEl = null; bar.classList.add('hidden'); };

    _audioEl.addEventListener('loadedmetadata', _onLoadedMetadata);
    _audioEl.addEventListener('timeupdate',     _onTimeUpdate);
    _audioEl.addEventListener('ended',         _onEnded);

    // Bind playback controls once on first call
    if (!playAudioFromUrl._controlsBound) {
        playAudioFromUrl._controlsBound = true;
        prog.addEventListener('input', () => {
            if (!_audioEl?.duration) return;
            _audioEl.currentTime = (prog.value / 100) * _audioEl.duration;
        });
        document.getElementById('btn-pause-audio')?.addEventListener('click', () => {
            if (!_audioEl) return;
            _audioEl.paused ? _audioEl.play() : _audioEl.pause();
        });
        document.getElementById('btn-stop-audio')?.addEventListener('click', () => {
            if (!_audioEl) return;
            _audioEl.pause(); _audioEl = null; prog.value = 0;
        });
    }

    _audioEl.play();
}

// ── TTS action (uses streaming AudioStream when available) ──────
async function sendTTS(text) {
    if (!text?.trim()) return;

    try {
        await window._audioStream.speak(text.trim());
    } catch(e) {
        // Fallback: original batch POST
        const chatModelSel = document.getElementById('chat-model-select');
        const modelName = chatModelSel?.value || window.CFG?.DEFAULT_TTS;
        const statusEl = document.getElementById('chat-status');
        if (statusEl) statusEl.textContent = 'Synthesizing...';

        try {
            const resp = await fetch(window.location.origin + '/v1/audio/speech', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: modelName, input: text.trim() }),
            });
            if (!resp.ok) throw new Error('TTS API ' + resp.status);
            const blob = await resp.blob();
            playAudioFromUrl(URL.createObjectURL(blob));
        } catch(e2) {
            if (statusEl) { statusEl.textContent = 'TTS error: ' + e2.message; setTimeout(() => { statusEl.textContent = ''; }, 3000); }
            window.showToast?.('TTS error: ' + e2.message);
        }
    }
}

// ── ASR action ───────────────────────────────────────────────────
async function sendASR(file) {
    const modelSel = document.getElementById('asr-model-select');
    const modelName = modelSel?.value || window.CFG?.DEFAULT_WHISPER;
    const resultDiv = document.getElementById('asr-result-display');

    if (!file) { window.showToast?.('No audio file selected'); return; }
    if (!resultDiv) return;

    resultDiv.innerHTML = '<div class="asr-status">Transcribing...</div>';

    try {
        const formData = new FormData();
        formData.append('model', modelName);
        formData.append('file', file);
        const resp = await fetch(window.location.origin + '/v1/audio/transcriptions', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error('ASR API ' + resp.status);
        const data = await resp.json();
        resultDiv.innerHTML = '<div class="asr-text">' + window.esc(data.text || '') + '</div>' +
            (data.language ? '<div class="asr-lang">Language: ' + data.language + '</div>' : '');
    } catch(e) {
        resultDiv.innerHTML = '<div class="asr-status" style="color:var(--red)">Error: ' + e.message + '</div>';
    }
}

// ── Expose on window for app.js and widget.js ─────────────────────
window._audio = { playAudioFromUrl, sendTTS, sendASR };
Object.defineProperty(window, 'audioEl', { get: () => _audioEl, configurable: true });
