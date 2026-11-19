/* Capture recent errors + retry last failure + status strip */
(function () {
  const KEY = "rabbit-error-log-v1";
  const MAX = 30;
  let lastPayload = null;

  function load() {
    try {
      const raw = localStorage.getItem(KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function save(arr) {
    try {
      localStorage.setItem(KEY, JSON.stringify(arr.slice(0, MAX)));
    } catch (e) {}
  }

  function pushErr(msg, detail) {
    const entry = {
      t: Date.now(),
      msg: String(msg || "Error").slice(0, 200),
      detail: detail ? String(detail).slice(0, 500) : "",
    };
    const list = load();
    list.unshift(entry);
    save(list);
    renderStrip();
    return entry;
  }

  // Status strip under session bar
  let strip = document.getElementById("status-strip");
  if (!strip) {
    strip = document.createElement("div");
    strip.id = "status-strip";
    strip.className = "status-strip";
    strip.hidden = true;
    const bar = document.querySelector(".session-bar");
    if (bar && bar.parentNode) bar.parentNode.insertBefore(strip, bar.nextSibling);
    else document.body.appendChild(strip);
  }

  function renderStrip() {
    const list = load();
    if (!list.length) {
      strip.hidden = true;
      strip.innerHTML = "";
      return;
    }
    strip.hidden = false;
    const latest = list[0];
    const when = new Date(latest.t).toLocaleTimeString();
    strip.innerHTML =
      '<span class="status-strip-msg" title="' +
      escapeAttr(latest.detail || latest.msg) +
      '">⚠ ' +
      escapeHtml(latest.msg) +
      " · " +
      when +
      "</span>" +
      '<button type="button" class="status-strip-btn" id="retry-last-fail">Retry</button>' +
      '<button type="button" class="status-strip-btn" id="clear-err-log">Dismiss</button>' +
      '<button type="button" class="status-strip-btn" id="show-err-log">Log (' +
      list.length +
      ")</button>";
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
  function escapeAttr(s) {
    return escapeHtml(s).replace(/"/g, "&quot;");
  }

  strip.addEventListener("click", function (ev) {
    const id = ev.target && ev.target.id;
    if (id === "clear-err-log") {
      save([]);
      renderStrip();
    }
    if (id === "show-err-log") {
      const lines = load()
        .map(function (e) {
          return new Date(e.t).toISOString() + "  " + e.msg + (e.detail ? " — " + e.detail : "");
        })
        .join("\n");
      const blob = new Blob([lines || "(empty)"], { type: "text/plain" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "rabbit-errors.txt";
      a.click();
      setTimeout(function () {
        URL.revokeObjectURL(a.href);
      }, 1000);
    }
    if (id === "retry-last-fail") {
      retryLast();
    }
  });

  function retryLast() {
    const promptEl = document.getElementById("prompt");
    const formEl = document.getElementById("form");
    if (lastPayload && promptEl && formEl) {
      promptEl.value = lastPayload;
      promptEl.dispatchEvent(new Event("input", { bubbles: true }));
      if (typeof formEl.requestSubmit === "function") formEl.requestSubmit();
      else formEl.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
      if (window.rabbitToast) rabbitToast("Retrying last prompt", "ok");
      return;
    }
    // fallback: regenerate
    const b = document.getElementById("regen-last");
    if (b) b.click();
    else if (window.rabbitToast) rabbitToast("Nothing to retry", "ok");
  }

  // Track last submitted prompt
  const formEl = document.getElementById("form");
  const promptEl = document.getElementById("prompt");
  if (formEl && promptEl) {
    formEl.addEventListener("submit", function () {
      const t = (promptEl.value || "").trim();
      if (t) lastPayload = t;
    });
  }

  // Hook window errors
  window.addEventListener("error", function (ev) {
    pushErr(ev.message || "Script error", ev.filename || "");
  });
  window.addEventListener("unhandledrejection", function (ev) {
    const r = ev.reason;
    pushErr("Promise rejection", r && r.message ? r.message : String(r || ""));
  });

  // Observe status pill / conn for failure keywords
  const pill = document.getElementById("status-pill");
  const conn = document.getElementById("conn-status");
  function watchText(el) {
    if (!el) return;
    let last = "";
    new MutationObserver(function () {
      const t = (el.textContent || "").trim();
      if (t === last) return;
      last = t;
      const low = t.toLowerCase();
      if (
        low.indexOf("error") >= 0 ||
        low.indexOf("fail") >= 0 ||
        low.indexOf("offline") >= 0 ||
        low.indexOf("timeout") >= 0
      ) {
        pushErr(t, el.id || "");
      }
    }).observe(el, { childList: true, characterData: true, subtree: true });
  }
  watchText(pill);
  watchText(conn);

  // Public API for other modules
  window.rabbitErrorLog = {
    push: pushErr,
    list: load,
    clear: function () {
      save([]);
      renderStrip();
    },
    retryLast: retryLast,
    setLastPayload: function (t) {
      lastPayload = t;
    },
  };

  renderStrip();
})();
