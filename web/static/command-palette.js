/* Ctrl+K command palette + recent prompts */
(function () {
  const RECENT_KEY = "rabbit-recent-prompts";
  const promptEl = document.getElementById("prompt");
  const formEl = document.getElementById("form");
  if (!promptEl) return;

  function loadRecent() {
    try {
      const raw = localStorage.getItem(RECENT_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function pushRecent(text) {
    const t = (text || "").trim();
    if (!t) return;
    let list = loadRecent().filter(function (x) {
      return x !== t;
    });
    list.unshift(t);
    list = list.slice(0, 20);
    try {
      localStorage.setItem(RECENT_KEY, JSON.stringify(list));
    } catch (e) {}
  }

  if (formEl) {
    formEl.addEventListener("submit", function () {
      pushRecent(promptEl.value);
    });
  }

  // Build palette DOM
  let overlay = document.getElementById("cmd-palette");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "cmd-palette";
    overlay.className = "cmd-palette";
    overlay.hidden = true;
    overlay.innerHTML =
      '<div class="cmd-palette-card" role="dialog" aria-label="Command palette">' +
      '<input type="search" id="cmd-palette-input" class="cmd-palette-input" placeholder="Type a command or search recent…" autocomplete="off" />' +
      '<ul id="cmd-palette-list" class="cmd-palette-list"></ul>' +
      '<p class="cmd-palette-hint"><kbd>↑</kbd><kbd>↓</kbd> · <kbd>Enter</kbd> run · <kbd>Esc</kbd> close</p>' +
      "</div>";
    document.body.appendChild(overlay);
  }

  const input = document.getElementById("cmd-palette-input");
  const listEl = document.getElementById("cmd-palette-list");
  let items = [];
  let active = 0;

  function commands() {
    const base = [
      { id: "new", label: "New session", run: function () { const b = document.getElementById("session-new"); if (b) b.click(); } },
      { id: "dup", label: "Duplicate session", run: function () { if (window.rabbitSessionsExtra) rabbitSessionsExtra.duplicate(); } },
      { id: "clear", label: "Clear chat", run: function () { const b = document.getElementById("clear-chat"); if (b) b.click(); } },
      { id: "regen", label: "Regenerate last", run: function () { const b = document.getElementById("regen-last"); if (b) b.click(); } },
      { id: "export", label: "Export JSON (this chat)", run: function () { const b = document.getElementById("export-chat"); if (b) b.click(); } },
      { id: "export-all", label: "Export all sessions", run: function () { if (window.rabbitPersist) rabbitPersist.exportAll(); } },
      { id: "save", label: "Force save chat", run: function () { if (window.rabbitPersist) rabbitPersist.save(); } },
      { id: "theme", label: "Toggle dark mode", run: function () { const b = document.getElementById("theme-toggle"); if (b) b.click(); } },
      { id: "persist", label: "Toggle persist chat", run: function () { const el = document.getElementById("persist-chat"); if (el) { el.checked = !el.checked; el.dispatchEvent(new Event("change", { bubbles: true })); } } },
      { id: "filter-all", label: "Filter: All messages", run: function () { if (window.rabbitChatFilters) rabbitChatFilters.apply("all"); } },
      { id: "filter-you", label: "Filter: Your messages", run: function () { if (window.rabbitChatFilters) rabbitChatFilters.apply("user"); } },
      { id: "filter-bot", label: "Filter: Rabbit replies", run: function () { if (window.rabbitChatFilters) rabbitChatFilters.apply("bot"); } },
      { id: "help", label: "Open help", run: function () { const b = document.getElementById("help-open"); if (b) b.click(); } },
      { id: "focus", label: "Focus composer", run: function () { promptEl.focus(); } },
    ];
    loadRecent().forEach(function (text, i) {
      base.push({
        id: "recent-" + i,
        label: "Recent: " + text.slice(0, 60) + (text.length > 60 ? "…" : ""),
        recent: text,
        run: function () {
          promptEl.value = text;
          promptEl.focus();
          promptEl.dispatchEvent(new Event("input", { bubbles: true }));
        },
      });
    });
    return base;
  }

  function render() {
    const q = (input.value || "").trim().toLowerCase();
    items = commands().filter(function (c) {
      if (!q) return true;
      return c.label.toLowerCase().indexOf(q) >= 0 || (c.recent && c.recent.toLowerCase().indexOf(q) >= 0);
    });
    active = 0;
    listEl.innerHTML = "";
    items.forEach(function (c, i) {
      const li = document.createElement("li");
      li.className = "cmd-palette-item" + (i === 0 ? " active" : "");
      li.textContent = c.label;
      li.dataset.idx = String(i);
      li.addEventListener("click", function () {
        runIdx(i);
      });
      listEl.appendChild(li);
    });
    if (!items.length) {
      const li = document.createElement("li");
      li.className = "cmd-palette-item muted";
      li.textContent = "No matches";
      listEl.appendChild(li);
    }
  }

  function setActive(i) {
    if (!items.length) return;
    active = (i + items.length) % items.length;
    listEl.querySelectorAll(".cmd-palette-item").forEach(function (el, idx) {
      el.classList.toggle("active", idx === active);
    });
    const el = listEl.children[active];
    if (el && el.scrollIntoView) el.scrollIntoView({ block: "nearest" });
  }

  function runIdx(i) {
    const c = items[i];
    close();
    if (c && typeof c.run === "function") c.run();
  }

  function open() {
    overlay.hidden = false;
    input.value = "";
    render();
    setTimeout(function () {
      input.focus();
    }, 10);
  }

  function close() {
    overlay.hidden = true;
  }

  input.addEventListener("input", render);
  input.addEventListener("keydown", function (ev) {
    if (ev.key === "ArrowDown") {
      ev.preventDefault();
      setActive(active + 1);
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      setActive(active - 1);
    } else if (ev.key === "Enter") {
      ev.preventDefault();
      runIdx(active);
    } else if (ev.key === "Escape") {
      ev.preventDefault();
      close();
    }
  });

  overlay.addEventListener("click", function (ev) {
    if (ev.target === overlay) close();
  });

  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && (ev.key === "k" || ev.key === "K")) {
      ev.preventDefault();
      if (overlay.hidden) open();
      else close();
    }
    if (ev.key === "Escape" && !overlay.hidden) {
      close();
    }
  });

  window.rabbitCommandPalette = { open: open, close: close };
})();
