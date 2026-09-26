/* app.js — init: load models, restore last context, wire chrome. */

(async () => {
    // Model list — one fetch at boot; model select shows it to every context.
    try {
        store.models = await api.listModels();
    } catch {
        showToast('Could not reach server — model list empty.');
    }

    // Restore the last open context (id only; history comes from IndexedDB).
    const last = Number(localStorage.getItem('arkestra-last-context') || 0);
    if (last) {
        const saved = await db.get(last);
        if (saved) {
            const rec = { ...saved, streaming: false, error: null, modelDead: false, abort: null };
            store.contexts.set(rec.id, rec);
            store.setActive(rec.id);
        }
    }

    if (store.activeId === null && store.contexts.size > 0) {
        store.setActive([...store.contexts.keys()].sort((a, b) => a - b)[0]);
    }

    // Chrome: + opens a fresh context, ⋯ opens the saved list.
    document.getElementById('btn-new').addEventListener('click', () => newContext());
    document.getElementById('btn-reopen').addEventListener('click', openSavedList);

    renderTabs();
    renderPane();
})();
