/* Multi-select bubbles + unread divider + /slash commands in composer */
(function () {
  const chatEl = document.getElementById("chat");
  const promptEl = document.getElementById("prompt");
  if (!chatEl) return;

  // --- Multi-select ---
  let selectMode = false;
  const selected = new Set();

  function updateSelectBar() {
    let bar = document.getElementById("select-bar");
    if (!selectMode || !selected.size) {
      if (bar) bar.hidden = true;
      return;
    }
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "select-bar";
      bar.className = "select-bar";
      bar.innerHTML =
        '<span id="select-count">0</span> selected' +
        '<button type="button" id="select-copy">Copy</button>' +
        '<button type="button" id="select-clear">Clear</button>' +
        '<button type="button" id="select-exit">Done</button>';
      const tools = document.querySelector(".composer-tools");
      if (tools) tools.parentNode.insertBefore(bar, tools);
      else chatEl.parentNode.insertBefore(bar, chatEl.nextSibling);
      bar.addEventListener("click", function (ev) {
        const id = ev.target && ev.target.id;
        if (id === "select-copy") copySelected();
        if (id === "select-clear") {
          selected.clear();
          paintSelection();
          updateSelectBar();
        }
        if (id === "select-exit") setSelectMode(false);
      });
    }
    bar.hidden = false;
    const c = document.getElementById("select-count");
    if (c) c.textContent = String(selected.size);
  }

  function paintSelection() {
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      b.classList.toggle("msg-selected", selected.has(b));
    });
  }

  function setSelectMode(on) {
    selectMode = on;
    document.documentElement.classList.toggle("select-mode", on);
    if (!on) {
      selected.clear();
      paintSelection();
    }
    updateSelectBar();
    if (window.rabbitToast) rabbitToast(on ? "Select mode on — click bubbles" : "Select mode off", "ok");
  }

  function copySelected() {
    const parts = [];
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      if (!selected.has(b)) return;
      const role = b.classList.contains("user") ? "You" : "Rabbit";
      const body = b.querySelector(".body");
      const text =
        body && body.dataset && body.dataset.raw != null
          ? body.dataset.raw
          : body
            ? body.textContent
            : "";
      parts.push("**" + role + "**\n" + (text || "").trim());
    });
    const md = parts.join("\n\n");
    if (!md) return;
    const done = function () {
      if (window.rabbitToast) rabbitToast("Copied " + selected.size + " messages", "ok");
    };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(md).then(done);
    else {
      const ta = document.createElement("textarea");
      ta.value = md;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      done();
    }
  }

  chatEl.addEventListener("click", function (ev) {
    if (!selectMode) return;
    const b = ev.target.closest(".bubble");
    if (!b || !chatEl.contains(b)) return;
    ev.preventDefault();
    ev.stopPropagation();
    if (selected.has(b)) selected.delete(b);
    else selected.add(b);
    paintSelection();
    updateSelectBar();
  });

  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.shiftKey && (ev.key === "m" || ev.key === "M")) {
      const tag = (ev.target && ev.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      ev.preventDefault();
      setSelectMode(!selectMode);
    }
  });

  const tools = document.querySelector(".composer-tools");
  if (tools && !document.getElementById("select-mode-btn")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "select-mode-btn";
    btn.className = "linkish";
    btn.textContent = "Select";
    btn.title = "Multi-select messages (Ctrl+Shift+M)";
    btn.addEventListener("click", function () {
      setSelectMode(!selectMode);
    });
    tools.appendChild(btn);
  }

  // --- Unread divider: mark last-read on blur / leave ---
  const READ_KEY = "rabbit-last-read-mark";
  function placeUnreadDivider() {
    const old = chatEl.querySelector(".unread-divider");
    if (old) old.remove();
    let mark;
    try {
      mark = localStorage.getItem(READ_KEY);
    } catch (e) {}
    if (!mark) return;
    const bubbles = Array.from(chatEl.querySelectorAll(".bubble"));
    let after = null;
    for (let i = 0; i < bubbles.length; i++) {
      const b = bubbles[i];
      const id = b.id || b.dataset.mid || "";
      if (id && id === mark) after = b;
    }
    if (!after || after === bubbles[bubbles.length - 1]) return;
    const div = document.createElement("div");
    div.className = "unread-divider";
    div.textContent = "New messages";
    if (after.nextSibling) after.parentNode.insertBefore(div, after.nextSibling);
    else after.parentNode.appendChild(div);
  }

  function markRead() {
    const bots = chatEl.querySelectorAll(".bubble");
    const last = bots[bots.length - 1];
    if (!last) return;
    if (!last.id && !last.dataset.mid) {
      last.dataset.mid = "m" + Date.now();
    }
    const id = last.id || last.dataset.mid;
    try {
      localStorage.setItem(READ_KEY, id);
    } catch (e) {}
    const d = chatEl.querySelector(".unread-divider");
    if (d) d.remove();
  }

  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden") markRead();
    else placeUnreadDivider();
  });
  window.addEventListener("blur", markRead);
  window.addEventListener("focus", placeUnreadDivider);
  new MutationObserver(placeUnreadDivider).observe(chatEl, { childList: true });

  // --- Slash commands in composer ---
  const SLASH = {
    "/clear": function () {
      const b = document.getElementById("clear-chat");
      if (b) b.click();
    },
    "/new": function () {
      const b = document.getElementById("session-new");
      if (b) b.click();
    },
    "/regen": function () {
      const b = document.getElementById("regen-last");
      if (b) b.click();
    },
    "/export": function () {
      const b = document.getElementById("export-md");
      if (b) b.click();
    },
    "/copy": function () {
      const b = document.getElementById("copy-last");
      if (b) b.click();
    },
    "/help": function () {
      const b = document.getElementById("help-open");
      if (b) b.click();
    },
    "/snip": function () {
      if (window.rabbitSnippets) rabbitSnippets.toggle();
    },
    "/notes": function () {
      if (window.rabbitNotes) rabbitNotes.toggle();
    },
    "/focus": function () {
      document.documentElement.classList.toggle("focus-mode");
    },
  };

  if (promptEl) {
    promptEl.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter" || ev.shiftKey) return;
      const v = (promptEl.value || "").trim();
      if (!v.startsWith("/")) return;
      const cmd = v.split(/\s+/)[0].toLowerCase();
      if (!SLASH[cmd]) return;
      ev.preventDefault();
      promptEl.value = "";
      promptEl.dispatchEvent(new Event("input", { bubbles: true }));
      SLASH[cmd]();
      if (window.rabbitToast) rabbitToast("Command " + cmd, "ok");
    });
  }

  window.rabbitSelectSlash = {
    setSelectMode: setSelectMode,
    markRead: markRead,
  };
})();
