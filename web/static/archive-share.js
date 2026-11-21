/* Archive sessions + focus search + share transcript as Markdown */
(function () {
  const ARCH_KEY = "rabbit-archived-sessions";
  const tabsHost = document.getElementById("session-tabs");
  const chatEl = document.getElementById("chat");

  function loadArch() {
    try {
      const raw = localStorage.getItem(ARCH_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function saveArch(arr) {
    try {
      localStorage.setItem(ARCH_KEY, JSON.stringify(arr));
    } catch (e) {}
  }

  function tabId(tab) {
    return tab.dataset.sessionId || tab.dataset.id || tab.getAttribute("data-id") || tab.textContent.trim();
  }

  function applyArchiveVisibility() {
    if (!tabsHost) return;
    const arch = loadArch();
    const showArchived = document.documentElement.classList.contains("show-archived");
    tabsHost.querySelectorAll(".session-tab").forEach(function (tab) {
      const id = tabId(tab);
      const isArch = arch.indexOf(id) >= 0;
      tab.classList.toggle("session-archived", isArch);
      if (isArch && !showArchived) tab.style.display = "none";
      else if (tab.style.display === "none" && !isArch) {
        /* leave filter-driven hide alone if session-filter set it */
      }
      if (isArch && showArchived) tab.style.display = "";
      if (!isArch && !document.getElementById("session-filter")) tab.style.display = "";
    });
  }

  function toggleArchiveActive() {
    if (!tabsHost) return;
    const active = tabsHost.querySelector(".session-tab.active");
    if (!active) {
      if (window.rabbitToast) rabbitToast("No active session", "ok");
      return;
    }
    const id = tabId(active);
    let arch = loadArch();
    const i = arch.indexOf(id);
    if (i >= 0) {
      arch.splice(i, 1);
      if (window.rabbitToast) rabbitToast("Session unarchived", "ok");
    } else {
      arch.push(id);
      if (window.rabbitToast) rabbitToast("Session archived (hidden)", "ok");
    }
    saveArch(arch);
    applyArchiveVisibility();
  }

  // Context menu extension via right-click already exists — add archive via Ctrl+Shift+A
  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.shiftKey && (ev.key === "a" || ev.key === "A")) {
      const tag = (ev.target && ev.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      ev.preventDefault();
      toggleArchiveActive();
    }
    // / focuses chat search when not in input
    if (ev.key === "/" && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
      const tag = (ev.target && ev.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      const search = document.getElementById("chat-search");
      if (search) {
        ev.preventDefault();
        search.focus();
        search.select && search.select();
      }
    }
  });

  // Toggle show archived in session bar
  const bar = document.querySelector(".session-bar");
  if (bar && !document.getElementById("show-archived")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "show-archived";
    btn.className = "btn-show-archived";
    btn.title = "Show archived sessions";
    btn.textContent = "Archived";
    bar.appendChild(btn);
    btn.addEventListener("click", function () {
      document.documentElement.classList.toggle("show-archived");
      btn.classList.toggle("active", document.documentElement.classList.contains("show-archived"));
      applyArchiveVisibility();
    });
  }

  if (tabsHost) {
    new MutationObserver(applyArchiveVisibility).observe(tabsHost, { childList: true, subtree: true });
    applyArchiveVisibility();
  }

  // Share transcript as Markdown
  function transcriptMarkdown() {
    if (!chatEl) return "";
    const lines = ["# Rabbit transcript", "", "_Exported " + new Date().toISOString() + "_", ""];
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      const role = b.classList.contains("user") ? "You" : "Rabbit";
      const body = b.querySelector(".body");
      const text =
        body && body.dataset && body.dataset.raw != null
          ? body.dataset.raw
          : body
            ? body.textContent
            : b.textContent || "";
      lines.push("## " + role, "", text.trim(), "");
    });
    return lines.join("\n");
  }

  function shareTranscript() {
    const md = transcriptMarkdown();
    if (!md || md.length < 40) {
      if (window.rabbitToast) rabbitToast("Nothing to share", "ok");
      return;
    }
    const doneCopy = function () {
      if (window.rabbitToast) rabbitToast("Transcript copied as Markdown", "ok");
    };
    if (navigator.share) {
      navigator
        .share({ title: "Rabbit transcript", text: md })
        .catch(function () {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(md).then(doneCopy);
          }
        });
      return;
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(md).then(doneCopy);
    } else {
      const ta = document.createElement("textarea");
      ta.value = md;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      doneCopy();
    }
  }

  const tools = document.querySelector(".composer-tools");
  if (tools && !document.getElementById("share-transcript")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "share-transcript";
    btn.className = "linkish";
    btn.textContent = "Share MD";
    btn.title = "Copy / share full transcript as Markdown";
    btn.addEventListener("click", shareTranscript);
    tools.appendChild(btn);
  }

  // Also on session context menu if present
  const menu = document.getElementById("session-ctx-menu");
  if (menu && !menu.querySelector('[data-act="archive"]')) {
    const a = document.createElement("button");
    a.type = "button";
    a.dataset.act = "archive";
    a.textContent = "Archive / unarchive";
    menu.appendChild(a);
    const s = document.createElement("button");
    s.type = "button";
    s.dataset.act = "share";
    s.textContent = "Share transcript MD";
    menu.appendChild(s);
    menu.addEventListener("click", function (ev) {
      const b = ev.target.closest("button[data-act]");
      if (!b) return;
      if (b.dataset.act === "archive") toggleArchiveActive();
      if (b.dataset.act === "share") shareTranscript();
    });
  }

  window.rabbitArchiveShare = {
    toggleArchiveActive: toggleArchiveActive,
    shareTranscript: shareTranscript,
    transcriptMarkdown: transcriptMarkdown,
  };
})();
