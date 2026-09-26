/* db.js — IndexedDB persistence for chat conversations.

Record: { id, model, name, params, history[], updated }
Key: numeric id (globally unique — id is minted per open tab,
and reopen reuses the stored id).
*/

const DB_NAME = 'arkestra';
const DB_VERSION = 1;
const STORE = 'conversations';

let _dbPromise = null;

function _open() {
    if (!_dbPromise) {
        _dbPromise = new Promise((resolve, reject) => {
            const req = indexedDB.open(DB_NAME, DB_VERSION);
            req.onupgradeneeded = () => {
                const db = req.result;
                if (!db.objectStoreNames.contains(STORE)) {
                    const os = db.createObjectStore(STORE, { keyPath: 'id' });
                    os.createIndex('by_updated', 'updated');
                }
            };
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });
    }
    return _dbPromise;
}

async function _tx(mode, fn) {
    const db = await _open();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, mode);
        const os = tx.objectStore(STORE);
        const req = fn(os);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

const db = {
    /* All conversations, newest-updated first. */
    async list() {
        const all = await _tx('readonly', os => os.getAll());
        return all.sort((a, b) => (b.updated || 0) - (a.updated || 0));
    },

    async get(id) {
        return (await _tx('readonly', os => os.get(id))) || null;
    },

    async put(rec) {
        await _tx('readwrite', os => os.put({ ...rec, updated: Date.now() }));
        return rec;
    },

    async del(id) {
        await _tx('readwrite', os => os.delete(id));
    },
};

window.db = db;
