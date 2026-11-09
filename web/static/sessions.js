/* Multi-session chat tabs (localStorage) */
(function () {
  const STORE = "rabbit-sessions-v1";
  const tabsEl = document.getElementById("session-tabs");
  const newBtn = document.getElementById("session-new");
  const chatEl = document.getElementById("chat");
  const emptyState = document.getElementById("empty-state");
  if (!tabsEl || !chatEl) return;

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
    } catch (e) {}
  }

  let state = loadStore();

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
      return { role: role, content: content };
    });
  }

  function clearChatDom() {
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      b.remove();
    });
    if (emptyState) emptyState.style.display = "";
  }

  function inject(role, text) {
    if (emptyState) emptyState.style.display = "none";
    const div = document.createElement("div");
    div.className = "bubble " + role;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = role === "user" ? "you" : "rabbit";
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = text || "";
    div.appendChild(meta);
    div.appendChild(body);
    chatEl.appendChild(div);
  }

  function persistActive() {
    const id = state.active;
    if (!state.sessions[id]) return;
    state.sessions[id].messages = readDomMessages().slice(-80);
    saveStore(state);
  }

  function loadActive() {
    const sess = state.sessions[state.active];
    clearChatDom();
    if (!sess || !sess.messages || !sess.messages.length) return;
    sess.messages.forEach(function (m) {
      inject(m.role === "user" ? "user" : "bot", m.content || "");
    });
    chatEl.scrollTop = chatEl.scrollHeight;
    if (window.rabbitFormatMarkdown) setTimeout(window.rabbitFormatMarkdown, 50);
  }

  function renderTabs() {
    tabsEl.innerHTML = "";
    Object.keys(state.sessions).forEach(function (id) {
      const s = state.sessions[id];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "session-tab" + (id === state.active ? " active" : "");
      btn.appendChild(document.createTextNode(s.name || id));
      if (Object.keys(state.sessions).length > 1) {
        const x = document.createElement("span");
        x.className = "x";
        x.textContent = "\u00d7";
        x.title = "Delete session";
        x.addEventListener("click", function (ev) {
          ev.stopPropagation();
          if (!confirm("Delete " + (s.name || id) + "?")) return;
          persistActive();
          delete state.sessions[id];
          if (state.active === id) {
            state.active = Object.keys(state.sessions)[0];
            loadActive();
          }
          saveStore(state);
          renderTabs();
        });
        btn.appendChild(x);
      }
      btn.addEventListener("click", function () {
        if (id === state.active) return;
        persistActive();
        state.active = id;
        saveStore(state);
        loadActive();
        renderTabs();
      });
      btn.addEventListener("dblclick", function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        const name = prompt("Rename session", s.name || id);
        if (name && name.trim()) {
          state.sessions[id].name = name.trim().slice(0, 40);
          saveStore(state);
          renderTabs();
        }
      });
      tabsEl.appendChild(btn);
    });
  }

  if (newBtn) {
    newBtn.addEventListener("click", function () {
      persistActive();
      const n = Object.keys(state.sessions).length + 1;
      const id = "s" + Date.now();
      state.sessions[id] = { name: "Chat " + n, messages: [] };
      state.active = id;
      saveStore(state);
      loadActive();
      renderTabs();
      if (window.rabbitToast) rabbitToast("New session", "ok");
    });
  }

  const obs = new MutationObserver(function () {
    persistActive();
  });
  obs.observe(chatEl, { childList: true, subtree: true, characterData: true });

  setTimeout(function () {
    if (!chatEl.querySelector(".bubble")) loadActive();
    else persistActive();
    renderTabs();
  }, 500);
})();
