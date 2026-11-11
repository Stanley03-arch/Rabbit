/* Rabbit chat UX extras */
(function () {
  const chatEl = document.getElementById("chat");
  const btn = document.getElementById("scroll-bottom");
  const promptEl = document.getElementById("prompt");
  const formEl = document.getElementById("form");
  if (!chatEl) return;

  const COLLAPSE_AT = 420;
  const WORD_KEY = "rabbit-show-word-count";
  const CHAR_KEY = "rabbit-show-char-count";
  const TIME_KEY = "rabbit-show-timestamps";
  const REL_KEY = "rabbit-relative-timestamps";
  const ROLE_KEY = "rabbit-show-roles";
  const LINK_KEY = "rabbit-linkify";

  const showWordEl = document.getElementById("show-word-count");
  const showCharEl = document.getElementById("show-char-count");
  const showTimeEl = document.getElementById("show-timestamps");
  const relTimeEl = document.getElementById("relative-timestamps");
  const showRoleEl = document.getElementById("show-roles");
  const linkifyEl = document.getElementById("linkify-urls");

  function loadPref(key, defaultOn) {
    try {
      const v = localStorage.getItem(key);
      if (v === null) return defaultOn;
      return v === "1";
    } catch (e) {
      return defaultOn;
    }
  }

  function applyVisibility() {
    const showWord = loadPref(WORD_KEY, true);
    const showChar = loadPref(CHAR_KEY, true);
    const showTime = loadPref(TIME_KEY, true);
    const relTime = loadPref(REL_KEY, false);
    const showRole = loadPref(ROLE_KEY, true);
    const linkify = loadPref(LINK_KEY, true);
    document.body.classList.toggle("hide-word-count", !showWord);
    document.body.classList.toggle("hide-char-count", !showChar);
    document.body.classList.toggle("hide-timestamps", !showTime);
    document.body.classList.toggle("hide-roles", !showRole);
    if (showWordEl) showWordEl.checked = showWord;
    if (showCharEl) showCharEl.checked = showChar;
    if (showTimeEl) showTimeEl.checked = showTime;
    if (relTimeEl) relTimeEl.checked = relTime;
    if (showRoleEl) showRoleEl.checked = showRole;
    if (linkifyEl) linkifyEl.checked = linkify;
  }

  applyVisibility();

  function bindToggle(el, key, label, after) {
    if (!el) return;
    el.addEventListener("change", function () {
      const on = el.checked;
      try {
        localStorage.setItem(key, on ? "1" : "0");
      } catch (e) {}
      applyVisibility();
      if (typeof after === "function") after();
      if (window.rabbitToast) rabbitToast(label + (on ? " on" : " off"), "ok");
    });
  }

  bindToggle(showWordEl, WORD_KEY, "Word count");
  bindToggle(showCharEl, CHAR_KEY, "Character count");
  bindToggle(showTimeEl, TIME_KEY, "Timestamps");
  bindToggle(relTimeEl, REL_KEY, "Relative time", function () {
    chatEl.querySelectorAll(".bubble").forEach(refreshTimeDisplay);
  });
  bindToggle(showRoleEl, ROLE_KEY, "Role labels");
  bindToggle(linkifyEl, LINK_KEY, "Clickable links", function () {
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      const body = b.querySelector(".body");
      if (!body) return;
      const raw = body.dataset.raw != null ? body.dataset.raw : body.textContent || "";
      renderBody(body, raw);
    });
  });

  function nearBottom() {
    return chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 80;
  }

  function syncBtn() {
    if (!btn) return;
    btn.hidden = !(chatEl.scrollHeight > chatEl.clientHeight + 40 && !nearBottom());
  }

  chatEl.addEventListener("scroll", syncBtn, { passive: true });
  window.addEventListener("resize", syncBtn);
  if (btn) {
    btn.addEventListener("click", function () {
      chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" });
      setTimeout(syncBtn, 300);
    });
  }

  function bubbleText(b) {
    const body = b.querySelector(".body");
    if (!body) return "";
    if (body.dataset && body.dataset.raw != null) return body.dataset.raw;
    return body.textContent || "";
  }

  function wordCount(text) {
    const t = (text || "").trim();
    if (!t) return 0;
    return t.split(/\s+/).length;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function linkifyHtml(text) {
    const esc = escapeHtml(text);
    return esc.replace(
      /(https?:\/\/[^\s<]+[^.,;:!?\s<])/g,
      '<a class="msg-link" href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
    );
  }

  function renderBody(body, raw) {
    if (body.dataset.raw == null) body.dataset.raw = raw;
    if (loadPref(LINK_KEY, true)) {
      body.innerHTML = linkifyHtml(raw).replace(/\n/g, "<br>");
    } else {
      body.textContent = raw;
    }
  }

  function formatRelative(ms) {
    const sec = Math.round((Date.now() - ms) / 1000);
    if (sec < 10) return "just now";
    if (sec < 60) return sec + "s ago";
    const min = Math.round(sec / 60);
    if (min < 60) return min + "m ago";
    const hr = Math.round(min / 60);
    if (hr < 24) return hr + "h ago";
    return Math.round(hr / 24) + "d ago";
  }

  function refreshTimeDisplay(b) {
    const time = b.querySelector(".msg-time");
    if (!time) return;
    const abs = time.dataset.abs || time.textContent.replace(/^\s*[\u00b7]\s*/, "").trim();
    if (!time.dataset.abs) time.dataset.abs = abs;
    if (!time.dataset.ms) {
      const m = abs.match(/(\d{1,2}):(\d{2})(?::(\d{2}))?/);
      if (m) {
        const d = new Date();
        d.setHours(+m[1], +m[2], m[3] ? +m[3] : 0, 0);
        time.dataset.ms = String(d.getTime());
      } else {
        time.dataset.ms = String(Date.now());
      }
    }
    time.textContent =
      " \u00b7 " + (loadPref(REL_KEY, false) ? formatRelative(+time.dataset.ms) : time.dataset.abs);
  }

  function ensureMeta(b) {
    const meta = b.querySelector(".meta");
    if (!meta) return;
    const text = bubbleText(b);

    if (!meta.querySelector(".msg-role")) {
      const raw = (meta.textContent || "").trim();
      const timeMatch = raw.match(/(\d{1,2}:\d{2}(?::\d{2})?)/);
      let rolePart = b.classList.contains("user") ? "you" : "rabbit";
      let timePart = "";
      if (timeMatch) {
        timePart = timeMatch[1];
        const parts = raw.split(/\s*[\u00b7|]\s*/);
        rolePart = (parts[0] || rolePart).trim() || rolePart;
      }
      meta.innerHTML = "";
      const role = document.createElement("span");
      role.className = "msg-role";
      role.textContent = rolePart;
      meta.appendChild(role);
      if (timePart) {
        const time = document.createElement("span");
        time.className = "msg-time";
        time.dataset.abs = timePart;
        time.dataset.ms = String(Date.now());
        time.textContent = " \u00b7 " + timePart;
        meta.appendChild(time);
      }
    }
    refreshTimeDisplay(b);

    let count = meta.querySelector(".msg-count");
    if (!count) {
      count = document.createElement("span");
      count.className = "msg-count";
      meta.appendChild(document.createTextNode(" \u00b7 "));
      meta.appendChild(count);
    }
    const n = wordCount(text);
    count.textContent = n + (n === 1 ? " word" : " words");

    let chars = meta.querySelector(".msg-chars");
    if (!chars) {
      chars = document.createElement("span");
      chars.className = "msg-chars";
      meta.appendChild(document.createTextNode(" \u00b7 "));
      meta.appendChild(chars);
    }
    chars.textContent = text.length + (text.length === 1 ? " char" : " chars");
  }

  function ensureCollapse(b) {
    const body = b.querySelector(".body");
    if (!body) return;
    const text = bubbleText(b);
    if (text.length < COLLAPSE_AT) {
      body.classList.remove("collapsed");
      const t = b.querySelector(".collapse-toggle");
      if (t) t.remove();
      return;
    }
    if (!b.querySelector(".collapse-toggle")) {
      body.classList.add("collapsed");
      const t = document.createElement("button");
      t.type = "button";
      t.className = "collapse-toggle";
      t.textContent = "Show more";
      t.addEventListener("click", function (ev) {
        ev.stopPropagation();
        const on = body.classList.toggle("collapsed");
        t.textContent = on ? "Show more" : "Show less";
      });
      b.appendChild(t);
    }
  }

  function dayKeyFromBubble(b) {
    const time = b.querySelector(".msg-time");
    const ms = time && time.dataset.ms ? +time.dataset.ms : Date.now();
    const d = new Date(ms);
    return d.getFullYear() + "-" + (d.getMonth() + 1) + "-" + d.getDate();
  }

  function formatDayLabel(key) {
    const parts = key.split("-").map(Number);
    const d = new Date(parts[0], parts[1] - 1, parts[2]);
    const today = new Date();
    const tKey =
      today.getFullYear() + "-" + (today.getMonth() + 1) + "-" + today.getDate();
    const y = new Date(today);
    y.setDate(y.getDate() - 1);
    const yKey = y.getFullYear() + "-" + (y.getMonth() + 1) + "-" + y.getDate();
    if (key === tKey) return "Today";
    if (key === yKey) return "Yesterday";
    return d.toLocaleDateString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
    });
  }

  function ensureDateSeparators() {
    chatEl.querySelectorAll(".day-sep").forEach(function (el) {
      el.remove();
    });
    let lastKey = null;
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      const key = dayKeyFromBubble(b);
      if (key !== lastKey) {
        const sep = document.createElement("div");
        sep.className = "day-sep";
        sep.textContent = formatDayLabel(key);
        b.parentNode.insertBefore(sep, b);
        lastKey = key;
      }
    });
  }

  function addAct(actions, cls, label, title, handler) {
    if (actions.querySelector("." + cls)) return;
    const el = document.createElement("button");
    el.type = "button";
    el.className = "bubble-act " + cls;
    el.textContent = label;
    el.title = title;
    el.addEventListener("click", function (ev) {
      ev.stopPropagation();
      handler();
    });
    actions.appendChild(el);
  }

  function ensureActions() {
    ensureDateSeparators();
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      ensureMeta(b);
      ensureCollapse(b);

      const body = b.querySelector(".body");
      if (body && body.dataset.raw != null) {
        if (!body.querySelector("a.msg-link") && loadPref(LINK_KEY, true)) {
          renderBody(body, body.dataset.raw);
        }
      } else if (body) {
        body.dataset.raw = body.textContent || "";
        if (loadPref(LINK_KEY, true)) renderBody(body, body.dataset.raw);
      }

      if (b.classList.contains("user") && !b.dataset.dblEdit) {
        b.dataset.dblEdit = "1";
        b.addEventListener("dblclick", function () {
          if (!promptEl) return;
          promptEl.value = bubbleText(b);
          promptEl.focus();
          promptEl.dispatchEvent(new Event("input", { bubbles: true }));
          if (window.rabbitToast) rabbitToast("Double-click: loaded into composer", "ok");
        });
      }

      let actions = b.querySelector(".bubble-actions");
      if (!actions) {
        actions = document.createElement("div");
        actions.className = "bubble-actions";
        b.appendChild(actions);
      }

      addAct(actions, "copy-btn", "Copy", "Copy message", function () {
        const raw = bubbleText(b);
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(raw).then(function () {
            if (window.rabbitToast) rabbitToast("Copied", "ok");
          });
        }
      });

      addAct(actions, "quote-btn", "Quote", "Quote into composer", function () {
        if (!promptEl) return;
        const raw = bubbleText(b).trim();
        const quoted = raw
          .split("\n")
          .map(function (line) {
            return "> " + line;
          })
          .join("\n");
        const cur = promptEl.value.trim();
        promptEl.value = cur ? cur + "\n\n" + quoted + "\n\n" : quoted + "\n\n";
        promptEl.focus();
        promptEl.dispatchEvent(new Event("input", { bubbles: true }));
        if (window.rabbitToast) rabbitToast("Quoted into composer", "ok");
      });

      if (b.classList.contains("user")) {
        addAct(actions, "edit-btn", "Edit", "Edit in composer", function () {
          if (!promptEl) return;
          promptEl.value = bubbleText(b);
          promptEl.focus();
          promptEl.dispatchEvent(new Event("input", { bubbles: true }));
          if (window.rabbitToast) rabbitToast("Loaded into composer", "ok");
        });
        addAct(actions, "resend-btn", "Resend", "Send again", function () {
          if (!promptEl || !formEl) return;
          promptEl.value = bubbleText(b);
          promptEl.dispatchEvent(new Event("input", { bubbles: true }));
          if (typeof formEl.requestSubmit === "function") formEl.requestSubmit();
          else formEl.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
        });
      }

      addAct(actions, "del-btn", "\u2715", "Remove from view", function () {
        b.remove();
        if (window.rabbitToast) rabbitToast("Message removed", "ok");
        syncBtn();
      });
    });
  }

  const statusPill = document.getElementById("status-pill");
  function syncStreamingClass() {
    const label = (statusPill && statusPill.textContent) || "";
    const busy = /generat|stream|busy|run/i.test(label);
    chatEl.querySelectorAll(".bubble.bot").forEach(function (b) {
      b.classList.remove("streaming");
    });
    if (busy) {
      const bots = chatEl.querySelectorAll(".bubble.bot");
      if (bots.length) bots[bots.length - 1].classList.add("streaming");
    }
  }
  if (statusPill) {
    new MutationObserver(syncStreamingClass).observe(statusPill, {
      childList: true,
      characterData: true,
      subtree: true,
    });
  }

  document.addEventListener("keydown", function (ev) {
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const bubbles = Array.from(chatEl.querySelectorAll(".bubble"));
    if (!bubbles.length) return;
    let idx = bubbles.findIndex(function (b) {
      return b.classList.contains("msg-focus");
    });
    if (ev.key === "j" || ev.key === "J") {
      ev.preventDefault();
      idx = Math.min(bubbles.length - 1, idx + 1);
    } else if (ev.key === "k" || ev.key === "K") {
      ev.preventDefault();
      idx = Math.max(0, idx < 0 ? bubbles.length - 1 : idx - 1);
    } else {
      return;
    }
    bubbles.forEach(function (b) {
      b.classList.remove("msg-focus");
    });
    const target = bubbles[idx];
    if (target) {
      target.classList.add("msg-focus");
      target.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  });

  setInterval(function () {
    if (!loadPref(REL_KEY, false)) return;
    chatEl.querySelectorAll(".bubble").forEach(refreshTimeDisplay);
  }, 30000);

  new MutationObserver(function () {
    syncBtn();
    ensureActions();
    syncStreamingClass();
  }).observe(chatEl, { childList: true, subtree: true });

  setTimeout(function () {
    ensureActions();
    syncBtn();
  }, 600);
})();
