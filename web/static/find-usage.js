/* Search hit ring + copy plain text + daily usage counter */
(function () {
  const USAGE_KEY = "rabbit-daily-usage";
  const chatEl = document.getElementById("chat");
  const searchEl = document.getElementById("chat-search");

  // Strengthen search hits with a ring on the parent bubble
  function ringHits() {
    if (!chatEl) return;
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      const hit = b.querySelector(".msg-hit, mark, .search-hit");
      b.classList.toggle("has-search-hit", !!hit);
    });
  }
  if (chatEl) {
    new MutationObserver(ringHits).observe(chatEl, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["class"],
    });
    if (searchEl) {
      searchEl.addEventListener("input", function () {
        setTimeout(ringHits, 50);
      });
    }
    ringHits();
  }

  // Copy plain text action on bubbles
  function bubbleText(b) {
    const body = b.querySelector(".body");
    if (body && body.dataset && body.dataset.raw != null) return body.dataset.raw;
    return body ? body.textContent : b.textContent || "";
  }

  function copyPlain(b) {
    const text = bubbleText(b).trim();
    if (!text) return;
    const done = function () {
      if (window.rabbitToast) rabbitToast("Copied plain text", "ok");
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done);
    } else {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      done();
    }
  }

  function enhance(b) {
    if (!b || b.dataset.plainReady === "1") return;
    b.dataset.plainReady = "1";
    const host = b.querySelector(".bubble-actions") || b;
    if (host.querySelector(".plain-copy-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "bubble-act plain-copy-btn";
    btn.textContent = "Txt";
    btn.title = "Copy plain text";
    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      copyPlain(b);
    });
    host.appendChild(btn);
  }

  function scan() {
    if (!chatEl) return;
    chatEl.querySelectorAll(".bubble").forEach(enhance);
  }
  if (chatEl) {
    new MutationObserver(scan).observe(chatEl, { childList: true, subtree: true });
    scan();
  }

  // Daily usage counter (local only)
  function todayKey() {
    const d = new Date();
    return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
  }

  function loadUsage() {
    try {
      const raw = localStorage.getItem(USAGE_KEY);
      const obj = raw ? JSON.parse(raw) : {};
      return obj && typeof obj === "object" ? obj : {};
    } catch (e) {
      return {};
    }
  }

  function saveUsage(obj) {
    try {
      localStorage.setItem(USAGE_KEY, JSON.stringify(obj));
    } catch (e) {}
  }

  function bumpUsage() {
    const u = loadUsage();
    const k = todayKey();
    u[k] = (u[k] || 0) + 1;
    // keep last 14 days
    const keys = Object.keys(u).sort();
    while (keys.length > 14) {
      delete u[keys.shift()];
    }
    saveUsage(u);
    renderUsage();
  }

  let usageEl = document.getElementById("daily-usage");
  if (!usageEl) {
    const met = document.querySelector(".panel h3");
    // place under session metrics panel
    const panels = document.querySelectorAll(".panel");
    let target = null;
    panels.forEach(function (p) {
      const h = p.querySelector("h3");
      if (h && /session metrics/i.test(h.textContent || "")) target = p;
    });
    usageEl = document.createElement("div");
    usageEl.id = "daily-usage";
    usageEl.className = "kv";
    usageEl.innerHTML = "<span>today</span><b id=\"daily-usage-n\">0</b>";
    if (target) target.appendChild(usageEl);
  }

  function renderUsage() {
    const u = loadUsage();
    const n = u[todayKey()] || 0;
    const el = document.getElementById("daily-usage-n");
    if (el) el.textContent = String(n);
  }

  const formEl = document.getElementById("form");
  if (formEl) {
    formEl.addEventListener("submit", function () {
      bumpUsage();
    });
  }
  renderUsage();

  // Ctrl+Shift+C copies last bot plain text
  document.addEventListener("keydown", function (ev) {
    if (!(ev.ctrlKey || ev.metaKey) || !ev.shiftKey) return;
    if (ev.key !== "c" && ev.key !== "C") return;
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (!chatEl) return;
    const bots = chatEl.querySelectorAll(".bubble.bot");
    const last = bots[bots.length - 1];
    if (!last) return;
    ev.preventDefault();
    copyPlain(last);
  });

  window.rabbitFindUsage = {
    bumpUsage: bumpUsage,
    copyPlain: copyPlain,
  };
})();
