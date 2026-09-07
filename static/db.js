/* db.js — IndexedDB wrapper for ModelArkestra client settings */

const DB_NAME = 'arkestra';
const DB_VERSION = 1;

let _dbPromise = null;

function open() {
    if (_dbPromise) return _dbPromise;
    _dbPromise = new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains('settings')) {
                db.createObjectStore('settings', { keyPath: 'modelId' });
            }
            if (!db.objectStoreNames.contains('layout_state')) {
                db.createObjectStore('layout_state', { keyPath: 'id' });
            }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
    return _dbPromise;
}

async function get(store, key) {
    const db = await open();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(store, 'readonly');
        const os = tx.objectStore(store);
        const req = os.get(key);
        req.onsuccess = () => resolve(req.result || null);
        req.onerror = () => reject(req.error);
    });
}

async function put(store, obj) {
    const db = await open();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(store, 'readwrite');
        const os = tx.objectStore(store);
        const req = os.put(obj);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function del(store, key) {
    const db = await open();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(store, 'readwrite');
        const os = tx.objectStore(store);
        const req = os.delete(key);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function getSettings(modelId) {
    return get('settings', modelId);
}

async function saveSettings(modelId, data) {
    return put('settings', { modelId, ...data });
}

async function clearSettings(modelId) {
    return del('settings', modelId);
}

window.arkestraDB = { getSettings, saveSettings, clearSettings };
