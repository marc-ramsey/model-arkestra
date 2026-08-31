/* widget-audio.js — TTS, ASR, audio playback */

// ── Playback state ────────────────────────────────────────────────
let _audioEl = null;
let _onLoadedMetadata = null, _onTimeUpdate = null, _onEnded = null;

function formatTime(sec) {
    if (!sec || isNaN(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
}

function escapeHtml(s) {
    return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
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

// ── TTS action ───────────────────────────────────────────────────
async function sendTTS() {
    const textEl = document.getElementById('f-chat-input');
    if (!textEl?.value.trim()) return;
    const statusEl = document.getElementById('chat-status');
    statusEl.textContent = 'Synthesizing...';

    try {
        const chatModelSel = document.getElementById('chat-model-select');
        const modelName = chatModelSel?.value || window.CFG?.DEFAULT_TTS;
        const resp = await fetch(window.location.origin + '/v1/audio/speech', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelName, input: textEl.value.trim() }),
        });
        if (!resp.ok) throw new Error('TTS API ' + resp.status);
        const blob = await resp.blob();
        playAudioFromUrl(URL.createObjectURL(blob));
    } catch(e) {
        statusEl.textContent = 'TTS error: ' + e.message;
        setTimeout(() => { statusEl.textContent = ''; }, 3000);
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
        resultDiv.innerHTML = '<div class="asr-text">' + escapeHtml(data.text || '') + '</div>' +
            (data.language ? '<div class="asr-lang">Language: ' + data.language + '</div>' : '');
    } catch(e) {
        resultDiv.innerHTML = '<div class="asr-status" style="color:var(--red)">Error: ' + e.message + '</div>';
    }
}

// ── Mic recording (called from AudioTranscriber renderer) ─────────
let _micRecorder = null;

function wireMicBtn() {
    const micBtn = document.getElementById('btn-record-audio');
    if (!micBtn) return;

    let isRecording = false;
    micBtn.addEventListener('click', async () => {
        if (!isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const recorder = new MediaRecorder(stream);
                const chunks = [];
                isRecording = true;
                micBtn.textContent = '⏹ Stop';
                micBtn.classList.add('recording');
                recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
                recorder.onstop = async () => {
                    stream.getTracks().forEach(t => t.stop());
                    isRecording = false; micBtn.textContent = '⏺ Mic'; micBtn.classList.remove('recording');
                    await sendASR(new Blob(chunks, { type: 'audio/webm' }));
                };
                recorder.start(); _micRecorder = recorder;
            } catch { window.showToast?.('Mic access denied'); }
        } else { _micRecorder?.stop(); }
    });
}

// ── ASR upload wiring (called from AudioTranscriber renderer) ─────
function wireAsrUpload() {
    const fileInput = document.getElementById('asr-file-input');
    const uploadBtn = document.getElementById('btn-upload-audio');
    if (uploadBtn && fileInput) { uploadBtn.addEventListener('click', () => fileInput.click()); }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) sendASR(file);
        });
    }

    const dropArea = document.querySelector('.asr-upload-area');
    if (!dropArea) return;

    ['dragenter','dragover'].forEach(evt => {
        dropArea.addEventListener(evt, (e) => { e.preventDefault(); dropArea.style.borderColor = 'var(--accent)'; });
    });
    ['dragleave','dragend'].forEach(evt => {
        dropArea.addEventListener(evt, () => { dropArea.style.borderColor = ''; });
    });
    dropArea.addEventListener('drop', (e) => {
        e.preventDefault(); dropArea.style.borderColor = '';
        const file = e.dataTransfer.files[0];
        if (file?.type.startsWith('audio/')) sendASR(file);
        else window.showToast?.('Please drop an audio file');
    });
}

// ── Expose on window for app.js and widget.js ─────────────────────
window._audio = { playAudioFromUrl, sendTTS, sendASR };
Object.defineProperty(window, 'audioEl', { get: () => _audioEl, configurable: true });
