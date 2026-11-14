/* Session extras: auto-title, duplicate, import-all, persist toggle */
(function () {
  const STORE = "rabbit-sessions-v1";
  const tabsEl = document.getElementById("session-tabs");
  const chatEl = document.getElementById("chat");
  const importBtn = document.getElementById("import-chat");
  const importFile = document.getElementById("import-file");
  if (!chatEl) return;

  function loadStore() {
    try {
      const raw = localStorage.getItem(STORE);
      if (!raw) return { active: "s1", sessions: { s1: { name: "Chat 1", messages: [] } } };
      return JSON.parse(raw);
    } catch (e) {
      return { active: "s1", sessions: { s1: { name: "Chat 1", messages: [] } } };
    }
  }

  function saveStore(st) {
    try {
      localStorage.setItem(STORE, JSON.stringify(st));
    } catch (e) {
      if (window.rabbitToast) rabbitToast("Could not save sessions", "err");
    }
  }

  function firstUserText() {
    const bubbles = chatEl.querySelectorAll(".bubble.user .body");
    if (!bubbles.length) return "";
    const b = bubbles[0];
    const t =
      b.dataset && b.dataset.raw != null ? b.dataset.raw : b.textContent || "";
    return (t || "").trim().replace(/\s+/g, " ").slice(0, 36);
  }

  function autoTitle() {
    const title = firstUserText();
    if (!title) return;
    const st = loadStore();
    const id = st.active;
    if (!id || !st.sessions[id]) return;
    const cur = st.sessions[id].name || "";
    // Only auto-rename default "Chat N" names
    if (cur && !/^Chat\s*\d*$/i.test(cur)) return;
    st.sessions[id].name = title + (title.length >= 36 ? "\u2026" : "");
    saveStore(st);
    // Refresh tab labels if present
    if (tabsEl) {
      tabsEl.querySelectorAll(".session-tab").forEach(function (btn) {
        if (btn.dataset && btn.dataset.id === id) {
          const label = btn.querySelector(".tab-label") || btn;
          // keep close button intact if structure is complex
          if (btn.childNodes.length <= 1) {
            const count = (st.sessions[id].messages || []).length;
            btn.textContent = st.sessions[id].name + (count ? " (" + count + ")" : "");
          }
        }
      });
    }
  }

  // Auto-title after messages settle
  let titleTimer = null;
  const obs = new MutationObserver(function () {
    clearTimeout(titleTimer);
    titleTimer = setTimeout(autoTitle, 400);
  });
  obs.observe(chatEl, { childList: true, subtree: true });

  function readDomMessages() {
    return Array.from(chatEl.querySelectorAll(".bubble")).map(function (div) {
      const role = div.classList.contains("user") ? "user" : "bot";
      const body = div.querySelector(".body");
      const content =
        body && body.dataset && body.dataset.raw != null
          ? body.dataset.raw
          : body
            ? body.textContent
            : "";
      return { role: role, content: content || "", ts: Date.now() };
    });
  }

  function duplicateActive() {
    const st = loadStore();
    const srcId = st.active;
    const src = st.sessions[srcId];
    if (!src) return;
    // capture current DOM into source first
    src.messages = readDomMessages().slice(-120);
    const n = Object.keys(st.sessions).length + 1;
    const id = "s" + Date.now().toString(36);
    st.sessions[id] = {
      name: (src.name || "Chat") + " copy",
      messages: (src.messages || []).map(function (m) {
        return { role: m.role, content: m.content, ts: m.ts || Date.now() };
      }),
    };
    st.active = id;
    saveStore(st);
    if (window.rabbitToast) rabbitToast("Session duplicated", "ok");
    // soft reload of chat via sessions module: clear and inject
    location.reload();
  }

  function importSessionsPayload(data) {
    if (!data || typeof data !== "object") return false;
    if (!data.sessions || typeof data.sessions !== "object") return false;
    const st = loadStore();
    const incoming = data.sessions;
    let added = 0;
    Object.keys(incoming).forEach(function (key) {
      const sess = incoming[key];
      if (!sess || !Array.isArray(sess.messages)) return;
      const id = "s" + Date.now().toString(36) + "_" + added;
      st.sessions[id] = {
        name: (sess.name || key || "Imported").slice(0, 40),
        messages: sess.messages.slice(-120).map(function (m) {
          return {
            role: m.role === "user" ? "user" : "bot",
            content: m.content || "",
            ts: m.ts || Date.now(),
          };
        }),
      };
      added++;
    });
    if (!added) return false;
    // switch to last imported
    const ids = Object.keys(st.sessions);
    st.active = ids[ids.length - 1];
    saveStore(st);
    if (window.rabbitToast) rabbitToast("Imported " + added + " session(s)", "ok");
    location.reload();
    return true;
  }

  // Shift+click Import → treat file as full sessions export when possible
  if (importFile && !importFile.dataset.sessionsImport) {
    importFile.dataset.sessionsImport = "1";
    importFile.addEventListener("change", function () {
      const file = importFile.files && importFile.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function () {
        try {
          const text = String(reader.result || "");
          if (!text.trim().startsWith("{")) return; // let other handlers deal with MD/plain
          const data = JSON.parse(text);
          if (data.sessions) {
            importSessionsPayload(data);
          }
        } catch (e) {
          /* ignore — existing import may handle */
        }
      };
      reader.readAsText(file);
    });
  }

  // Keyboard shortcuts
  document.addEventListener("keydown", function (ev) {
    if (!(ev.ctrlKey || ev.metaKey) || !ev.shiftKey) return;
    // Ctrl+Shift+D already theme — use Ctrl+Shift+U for duplicate
    if (ev.key === "U" || ev.key === "u") {
      ev.preventDefault();
      duplicateActive();
      return;
    }
    if (ev.key === "P" || ev.key === "p") {
      ev.preventDefault();
      const el = document.getElementById("persist-chat");
      if (el) {
        el.checked = !el.checked;
        el.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }
  });

  // Add Duplicate control near + Session
  const newBtn = document.getElementById("session-new");
  if (newBtn && !document.getElementById("session-dup")) {
    const dup = document.createElement("button");
    dup.type = "button";
    dup.id = "session-dup";
    dup.className = "btn-session-new";
    dup.title = "Duplicate session (Ctrl+Shift+U)";
    dup.textContent = "Dup";
    dup.addEventListener("click", function () {
      duplicateActive();
    });
    newBtn.parentNode.insertBefore(dup, newBtn.nextSibling);
  }

  window.rabbitSessionsExtra = {
    autoTitle: autoTitle,
    duplicate: duplicateActive,
    importAll: importSessionsPayload,
  };
})();
