/* Reliable chat message persistence (localStorage)
 * - Saves bubbles when Persist chat is on
 * - Restores after refresh
 * - Syncs with multi-session store (rabbit-sessions-v1)
 * - Status indicator + quota-safe saves + export all sessions
 */
(function () {
  const HISTORY_KEY = "rabbit-dashboard-history-v1";
  const SESSIONS_KEY = "rabbit-sessions-v1";
  const PREF_KEY = "rabbit-settings-v1";
  const chatEl = document.getElementById("chat");
  const emptyState = document.getElementById("empty-state");
  const persistEl = document.getElementById("persist-chat");
  if (!chatEl) return;

  let restoring = false;
  let saveTimer = null;
  let restoredOnce = false;
  let lastSavedAt = 0;
  let lastCount = 0;

  // Status pill next to composer tools
  let statusEl = document.getElementById("persist-status");
  if (!statusEl) {
    const tools = document.querySelector(".composer-tools");
    statusEl = document.createElement("span");
    statusEl.id = "persist-status";
    statusEl.className = "persist-status";
    statusEl.title = "Message persistence status";
    if (tools) tools.appendChild(statusEl);
    else document.body.appendChild(statusEl);
  }

  function setStatus(text, kind) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.dataset.kind = kind || "ok";
  }

  function persistEnabled() {
    if (persistEl) return !!persistEl.checked;
    try {
      const raw = localStorage.getItem(PREF_KEY);
      if (!raw) return true;
      const s = JSON.parse(raw);
      if (typeof s.persist_chat === "boolean") return s.persist_chat;
    } catch (e) {}
    return true;
  }

  function readMessages() {
    return Array.from(chatEl.querySelectorAll(".bubble")).map(function (div) {
      const role = div.classList.contains("user") ? "user" : "bot";
      const body = div.querySelector(".body");
      let content = "";
      if (body) {
        content =
          body.dataset && body.dataset.raw != null
            ? body.dataset.raw
            : body.textContent || "";
      }
      const timeEl = div.querySelector(".msg-time");
      const ts =
        (timeEl && timeEl.dataset && timeEl.dataset.ms && +timeEl.dataset.ms) ||
        Date.now();
      return { role: role, content: content, ts: ts };
    });
  }

  function formatClock(ts) {
    try {
      const d = new Date(ts || Date.now());
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch (e) {
      return "";
    }
  }

  function inject(role, text, ts) {
    if (emptyState) emptyState.style.display = "none";
    const div = document.createElement("div");
    div.className = "bubble " + (role === "user" ? "user" : "bot");
    const meta = document.createElement("div");
    meta.className = "meta";
    const clock = formatClock(ts);
    meta.textContent =
      (role === "user" ? "you" : "rabbit") + (clock ? " \u00b7 " + clock : "");
    const body = document.createElement("div");
    body.className = "body";
    body.dataset.raw = text || "";
    body.textContent = text || "";
    div.appendChild(meta);
    div.appendChild(body);
    chatEl.appendChild(div);
  }

  function clearDom() {
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      b.remove();
    });
    chatEl.querySelectorAll(".day-sep").forEach(function (b) {
      b.remove();
    });
    if (emptyState) emptyState.style.display = "";
  }

  function loadSessions() {
    try {
      const raw = localStorage.getItem(SESSIONS_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch (e) {
      return null;
    }
  }

  function saveSessionsMessages(msgs) {
    try {
      let st = loadSessions();
      if (!st || !st.sessions) {
        st = { active: "s1", sessions: { s1: { name: "Chat 1", messages: [] } } };
      }
      const id = st.active || "s1";
      if (!st.sessions[id]) st.sessions[id] = { name: id, messages: [] };
      st.sessions[id].messages = msgs;
      localStorage.setItem(SESSIONS_KEY, JSON.stringify(st));
    } catch (e) {
      console.warn("[persist] sessions save", e);
      throw e;
    }
  }

  function saveNow() {
    if (restoring) return;
    if (!persistEnabled()) {
      setStatus("persist off", "off");
      return;
    }
    const msgs = readMessages().slice(-120);
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(msgs));
      saveSessionsMessages(msgs);
      lastSavedAt = Date.now();
      lastCount = msgs.length;
      setStatus("saved · " + lastCount + " msg", "ok");
    } catch (e) {
      console.warn("[persist] save failed", e);
      setStatus("storage full", "err");
      if (window.rabbitToast) {
        rabbitToast("Could not save chat (storage full?). Export JSON as backup.", "err");
      }
    }
  }

  function scheduleSave() {
    if (restoring) return;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveNow, 200);
  }

  function messagesFromStore() {
    const st = loadSessions();
    if (st && st.sessions && st.active && st.sessions[st.active]) {
      const m = st.sessions[st.active].messages;
      if (Array.isArray(m) && m.length) return m;
    }
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      if (!raw) return [];
      const arr = JSON.parse(raw);
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function restore() {
    if (restoredOnce) return;
    if (!persistEnabled()) {
      setStatus("persist off", "off");
      restoredOnce = true;
      return;
    }
    if (chatEl.querySelector(".bubble")) {
      restoredOnce = true;
      scheduleSave();
      return;
    }
    const msgs = messagesFromStore();
    if (!msgs.length) {
      restoredOnce = true;
      setStatus("no saved chat", "off");
      return;
    }
    restoring = true;
    msgs.forEach(function (m) {
      inject(m.role === "user" ? "user" : "bot", m.content || "", m.ts);
    });
    restoring = false;
    restoredOnce = true;
    chatEl.scrollTop = chatEl.scrollHeight;
    lastCount = msgs.length;
    lastSavedAt = Date.now();
    setStatus("restored · " + msgs.length + " msg", "ok");
    if (window.rabbitToast) {
      rabbitToast("Restored " + msgs.length + " messages", "ok");
    }
  }

  function wipeStorage() {
    try {
      localStorage.removeItem(HISTORY_KEY);
    } catch (e) {}
    try {
      const st = loadSessions();
      if (st && st.sessions && st.active && st.sessions[st.active]) {
        st.sessions[st.active].messages = [];
        localStorage.setItem(SESSIONS_KEY, JSON.stringify(st));
      }
    } catch (e) {}
    lastCount = 0;
    setStatus(persistEnabled() ? "cleared" : "persist off", "off");
  }

  function exportAllSessions() {
    const st = loadSessions() || {
      active: "s1",
      sessions: { s1: { name: "Chat 1", messages: readMessages() } },
    };
    // refresh active from DOM
    try {
      const id = st.active || "s1";
      if (!st.sessions[id]) st.sessions[id] = { name: id, messages: [] };
      st.sessions[id].messages = readMessages().slice(-120);
    } catch (e) {}
    const payload = {
      exportedAt: new Date().toISOString(),
      active: st.active,
      sessions: st.sessions,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "rabbit-sessions-" + Date.now() + ".json";
    a.click();
    setTimeout(function () {
      URL.revokeObjectURL(a.href);
    }, 1000);
    if (window.rabbitToast) rabbitToast("Exported all sessions", "ok");
  }

  // Observe chat DOM
  const obs = new MutationObserver(function () {
    if (restoring) return;
    scheduleSave();
  });
  obs.observe(chatEl, { childList: true, subtree: true, characterData: true });

  if (persistEl) {
    persistEl.addEventListener("change", function () {
      if (!persistEl.checked) {
        wipeStorage();
        if (window.rabbitToast) rabbitToast("Chat persistence off", "ok");
      } else {
        saveNow();
        if (window.rabbitToast) rabbitToast("Chat persistence on", "ok");
      }
    });
  }

  const clearBtn = document.getElementById("clear-chat");
  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      setTimeout(function () {
        if (!chatEl.querySelector(".bubble")) wipeStorage();
        else scheduleSave();
      }, 50);
    });
  }

  // Export-all: long-press or secondary control near Export JSON
  const exportBtn = document.getElementById("export-chat");
  if (exportBtn && !exportBtn.dataset.sessionsBound) {
    exportBtn.dataset.sessionsBound = "1";
    exportBtn.title = (exportBtn.title || "Export") + " · Shift+click = all sessions";
    exportBtn.addEventListener(
      "click",
      function (ev) {
        if (ev.shiftKey) {
          ev.preventDefault();
          ev.stopImmediatePropagation();
          exportAllSessions();
        }
      },
      true
    );
  }

  // Keyboard: Ctrl+Shift+S force save, Ctrl+Shift+E export sessions
  document.addEventListener("keydown", function (ev) {
    if (!(ev.ctrlKey || ev.metaKey) || !ev.shiftKey) return;
    if (ev.key === "S" || ev.key === "s") {
      ev.preventDefault();
      saveNow();
      if (window.rabbitToast) rabbitToast("Chat saved", "ok");
    }
    if (ev.key === "E" || ev.key === "e") {
      ev.preventDefault();
      exportAllSessions();
    }
  });

  window.addEventListener("beforeunload", function () {
    if (persistEnabled()) saveNow();
  });

  function tryRestore() {
    restore();
  }
  if (document.readyState === "complete") {
    setTimeout(tryRestore, 300);
  } else {
    window.addEventListener("load", function () {
      setTimeout(tryRestore, 300);
    });
  }
  const prevReady = window.rabbitOnAppReady;
  window.rabbitOnAppReady = function () {
    if (typeof prevReady === "function") {
      try {
        prevReady();
      } catch (e) {}
    }
    setTimeout(tryRestore, 200);
  };

  // Refresh status clock occasionally
  setInterval(function () {
    if (!persistEnabled()) {
      setStatus("persist off", "off");
      return;
    }
    if (!lastSavedAt) return;
    const sec = Math.round((Date.now() - lastSavedAt) / 1000);
    if (sec < 5) setStatus("saved · " + lastCount + " msg", "ok");
    else if (sec < 60) setStatus("saved " + sec + "s ago · " + lastCount, "ok");
    else setStatus("saved " + Math.round(sec / 60) + "m ago · " + lastCount, "ok");
  }, 5000);

  window.rabbitPersist = {
    save: saveNow,
    restore: restore,
    read: readMessages,
    wipe: wipeStorage,
    exportAll: exportAllSessions,
  };

  setStatus(persistEnabled() ? "ready" : "persist off", persistEnabled() ? "ok" : "off");
})();
