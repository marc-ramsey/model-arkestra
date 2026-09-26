/* store.js — single app state. Mutations go through here; render reads it. */

const store = {
    /* Open contexts, keyed by id. Record:
     *   { id, model, name, named, params, history[], streaming, error,
     *     modelDead, abort }
     * history[0] is the system message (always present). */
    contexts: new Map(),
    activeId: null,
    models: [],   // from GET /api/models: {name, model, size}[]

    nextId() {
        let max = 0;
        for (const c of this.contexts.values()) if (c.id > max) max = c.id;
        return max + 1;
    },

    active() { return this.contexts.get(this.activeId) || null; },

    setActive(id) {
        if (id === null) {
            this.activeId = null;
            localStorage.removeItem('arkestra-last-context');
            return;
        }
        if (!this.contexts.has(id)) return;
        this.activeId = id;
        localStorage.setItem('arkestra-last-context', String(id));
    },

    modelNames() { return this.models.map(m => m.name); },
};

window.store = store;
