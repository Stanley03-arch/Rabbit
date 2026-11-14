/* Chat role filters + session soft-delete undo */
(function () {
  const chatEl = document.getElementById("chat");
  const tabsEl = document.getElementById("session-tabs");
  if (!chatEl) return;

  const FILTER_KEY = "rabbit-msg-filter";

  // --- Role filter bar ---
  let bar = document.getElementById("msg-filter-bar");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "msg-filter-bar";
    bar.className = "msg-filter-bar";
    bar.innerHTML =
      '<button type="button" class="msg-filter active" data-filter="all">All</button>' +
      '<button type="button" class="msg-filter" data-filter="user">You</button>' +
      '<button type="button" class="msg-filter" data-filter="bot">Rabbit</button>' +
      '<span class="msg-filter-count" id="msg-filter-count"></span>';
    const chips = document.getElementById("chips");
    if (chips && chips.parentNode) {
      chips.parentNode.insertBefore(bar, chips.nextSibling);
    } else {
      chatEl.parentNode.insertBefore(bar, chatEl);
    }
  }

  function loadFilter() {
    try {
      return localStorage.getItem(FILTER_KEY) || "all";
    } catch (e) {
      return "all";
    }
  }

  function applyFilter(mode) {
    mode = mode || "all";
    try {
      localStorage.setItem(FILTER_KEY, mode);
    } catch (e) {}
    bar.querySelectorAll(".msg-filter").forEach(function (btn) {
      btn.classList.toggle("active", btn.dataset.filter === mode);
    });
    let visible = 0;
    let total = 0;
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      total++;
      const isUser = b.classList.contains("user");
      const show =
        mode === "all" ||
        (mode === "user" && isUser) ||
        (mode === "bot" && !isUser);
      b.style.display = show ? "" : "none";
      if (show) visible++;
    });
    // Hide day seps when filtering
    chatEl.querySelectorAll(".day-sep").forEach(function (sep) {
      sep.style.display = mode === "all" ? "" : "none";
    });
    const countEl = document.getElementById("msg-filter-count");
    if (countEl) {
      countEl.textContent =
        mode === "all" ? total + " msgs" : visible + " / " + total;
    }
  }

  bar.addEventListener("click", function (ev) {
    const btn = ev.target.closest(".msg-filter");
    if (!btn) return;
    applyFilter(btn.dataset.filter);
  });

  const obs = new MutationObserver(function () {
    applyFilter(loadFilter());
  });
  obs.observe(chatEl, { childList: true });
  applyFilter(loadFilter());

  // Keyboard: 1 all, 2 you, 3 rabbit (outside inputs)
  document.addEventListener("keydown", function (ev) {
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (ev.key === "1") applyFilter("all");
    if (ev.key === "2") applyFilter("user");
    if (ev.key === "3") applyFilter("bot");
  });

  // --- Soft-delete session with undo ---
  if (tabsEl && !tabsEl.dataset.softDelete) {
    tabsEl.dataset.softDelete = "1";
    tabsEl.addEventListener(
      "click",
      function (ev) {
        const x = ev.target.closest && ev.target.closest(".session-x, .tab-close");
        // sessions.js uses a small × button — catch generic close on tab
        const closeBtn =
          ev.target.closest("button") &&
          (ev.target.textContent || "").trim() === "×"
            ? ev.target
            : null;
        const btn = x || closeBtn;
        if (!btn) return;
        // Let the original handler run; we snapshot just before via capture isn't easy.
        // Instead: on successful delete (tab gone), we can't undo without snapshot.
        // Snapshot store on any × mousedown.
      },
      true
    );

    let lastSnapshot = null;
    tabsEl.addEventListener(
      "mousedown",
      function (ev) {
        const t = (ev.target.textContent || "").trim();
        if (t !== "×" && t !== "x" && t !== "✕") return;
        try {
          lastSnapshot = localStorage.getItem("rabbit-sessions-v1");
        } catch (e) {
          lastSnapshot = null;
        }
      },
      true
    );

    tabsEl.addEventListener("click", function (ev) {
      const t = (ev.target.textContent || "").trim();
      if (t !== "×" && t !== "x" && t !== "✕") return;
      if (!lastSnapshot) return;
      const snap = lastSnapshot;
      lastSnapshot = null;
      setTimeout(function () {
        if (!window.rabbitToast) return;
        rabbitToast("Session removed — click to undo", "ok");
        const host = document.getElementById("toast-host");
        if (!host) return;
        const last = host.lastElementChild;
        if (!last) return;
        last.style.cursor = "pointer";
        const undo = function () {
          last.removeEventListener("click", undo);
          try {
            localStorage.setItem("rabbit-sessions-v1", snap);
            location.reload();
          } catch (e) {}
        };
        last.addEventListener("click", undo);
        setTimeout(function () {
          last.removeEventListener("click", undo);
        }, 6000);
      }, 30);
    });
  }

  window.rabbitChatFilters = { apply: applyFilter, current: loadFilter };
})();
