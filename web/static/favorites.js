/* Favorite prompts + session cycle shortcuts + palette helpers */
(function () {
  const FAV_KEY = "rabbit-favorite-prompts";
  const RECENT_KEY = "rabbit-recent-prompts";
  const promptEl = document.getElementById("prompt");
  const formEl = document.getElementById("form");
  const chatEl = document.getElementById("chat");
  if (!promptEl) return;

  function loadList(key) {
    try {
      const raw = localStorage.getItem(key);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function saveList(key, arr) {
    try {
      localStorage.setItem(key, JSON.stringify(arr.slice(0, 30)));
    } catch (e) {}
  }

  function loadFavs() {
    return loadList(FAV_KEY);
  }

  function toggleFav(text) {
    const t = (text || "").trim();
    if (!t) return false;
    let list = loadFavs();
    const i = list.indexOf(t);
    if (i >= 0) {
      list.splice(i, 1);
      saveList(FAV_KEY, list);
      if (window.rabbitToast) rabbitToast("Removed favorite", "ok");
      return false;
    }
    list.unshift(t);
    saveList(FAV_KEY, list);
    if (window.rabbitToast) rabbitToast("Added favorite", "ok");
    return true;
  }

  function starCurrent() {
    return toggleFav(promptEl.value);
  }

  // Star button near save-prompt
  const tools = document.querySelector(".composer-tools");
  if (tools && !document.getElementById("star-prompt")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "star-prompt";
    btn.className = "linkish";
    btn.title = "Favorite current text (Ctrl+Shift+F)";
    btn.textContent = "★ Favorite";
    btn.addEventListener("click", starCurrent);
    tools.appendChild(btn);
  }

  function chatStats() {
    if (!chatEl) return { msgs: 0, user: 0, bot: 0, chars: 0 };
    const bubbles = chatEl.querySelectorAll(".bubble");
    let user = 0,
      bot = 0,
      chars = 0;
    bubbles.forEach(function (b) {
      if (b.classList.contains("user")) user++;
      else bot++;
      const body = b.querySelector(".body");
      const t =
        body && body.dataset && body.dataset.raw != null
          ? body.dataset.raw
          : body
            ? body.textContent
            : "";
      chars += (t || "").length;
    });
    return { msgs: bubbles.length, user: user, bot: bot, chars: chars };
  }

  function copyAllMarkdown() {
    if (!chatEl) return;
    const lines = [];
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      if (b.style.display === "none") return;
      const role = b.classList.contains("user") ? "User" : "Rabbit";
      const body = b.querySelector(".body");
      const t =
        body && body.dataset && body.dataset.raw != null
          ? body.dataset.raw
          : body
            ? body.textContent
            : "";
      lines.push("### " + role + "\n\n" + (t || "") + "\n");
    });
    const md = lines.join("\n");
    if (!md.trim()) {
      if (window.rabbitToast) rabbitToast("Nothing to copy", "ok");
      return;
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(md).then(function () {
        if (window.rabbitToast) rabbitToast("Copied chat as Markdown", "ok");
      });
    } else {
      const ta = document.createElement("textarea");
      ta.value = md;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      if (window.rabbitToast) rabbitToast("Copied chat as Markdown", "ok");
    }
  }

  function cycleSession(dir) {
    const tabs = document.querySelectorAll("#session-tabs .session-tab");
    if (!tabs.length) return;
    let idx = -1;
    tabs.forEach(function (t, i) {
      if (t.classList.contains("active")) idx = i;
    });
    if (idx < 0) idx = 0;
    const next = (idx + dir + tabs.length) % tabs.length;
    tabs[next].click();
  }

  document.addEventListener("keydown", function (ev) {
    // Ctrl+Shift+F favorite current prompt
    if ((ev.ctrlKey || ev.metaKey) && ev.shiftKey && (ev.key === "F" || ev.key === "f")) {
      ev.preventDefault();
      starCurrent();
      return;
    }
    // Alt+[ / Alt+] cycle sessions
    if (ev.altKey && !ev.ctrlKey && !ev.metaKey) {
      if (ev.key === "[" || ev.code === "BracketLeft") {
        ev.preventDefault();
        cycleSession(-1);
      }
      if (ev.key === "]" || ev.code === "BracketRight") {
        ev.preventDefault();
        cycleSession(1);
      }
    }
  });

  // Extend command palette when available
  function patchPalette() {
    if (!window.rabbitCommandPalette) return;
    const origOpen = window.rabbitCommandPalette.open;
    // Inject into commands by monkey-patching list build is hard;
    // instead register extras globally for palette to pick up if reloaded.
    window.rabbitFavorites = {
      list: loadFavs,
      toggle: toggleFav,
      starCurrent: starCurrent,
      stats: chatStats,
      copyAllMarkdown: copyAllMarkdown,
      cycleSession: cycleSession,
    };
  }
  patchPalette();
  window.rabbitFavorites = {
    list: loadFavs,
    toggle: toggleFav,
    starCurrent: starCurrent,
    stats: chatStats,
    copyAllMarkdown: copyAllMarkdown,
    cycleSession: cycleSession,
  };
})();
