/* Chat UX: scroll, copy, edit, resend, quote, delete, collapse, counts, timestamps, caret */
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
  const showWordEl = document.getElementById("show-word-count");
  const showCharEl = document.getElementById("show-char-count");
  const showTimeEl = document.getElementById("show-timestamps");

  function loadPref(key, defaultOn) {
    try {
      const v = localStorage.getItem(key);
      if (v === null) return defaultOn;
      return v === "1";
    } catch (e) {
      return defaultOn;
    }
  }

  function applyCountVisibility() {
    const showWord = loadPref(WORD_KEY, true);
    const showChar = loadPref(CHAR_KEY, true);
    const showTime = loadPref(TIME_KEY, true);
    document.body.classList.toggle("hide-word-count", !showWord);
    document.body.classList.toggle("hide-char-count", !showChar);
    document.body.classList.toggle("hide-timestamps", !showTime);
    if (showWordEl) showWordEl.checked = showWord;
    if (showCharEl) showCharEl.checked = showChar;
    if (showTimeEl) showTimeEl.checked = showTime;
  }

  applyCountVisibility();

  function bindToggle(el, key, label) {
    if (!el) return;
    el.addEventListener("change", function () {
      const on = el.checked;
      try {
        localStorage.setItem(key, on ? "1" : "0");
      } catch (e) {}
      applyCountVisibility();
      if (window.rabbitToast) rabbitToast(label + (on ? " shown" : " hidden"), "ok");
    });
  }
  bindToggle(showWordEl, WORD_KEY, "Word count");
  bindToggle(showCharEl, CHAR_KEY, "Character count");
  bindToggle(showTimeEl, TIME_KEY, "Timestamps");

  function nearBottom() {
    return chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 80;
  }

  function syncBtn() {
    if (!btn) return;
    const show = chatEl.scrollHeight > chatEl.clientHeight + 40 && !nearBottom();
    btn.hidden = !show;
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

  function ensureMeta(b) {
    let meta = b.querySelector(".meta");
    if (!meta) return;
    const text = bubbleText(b);

    if (!meta.querySelector(".msg-time") && !meta.querySelector(".msg-role")) {
      const raw = (meta.textContent || "").trim();
      const timeMatch = raw.match(/(\d{1,2}:\d{2}(?::\d{2})?)/);
      let rolePart = raw;
      let timePart = "";
      if (timeMatch) {
        timePart = timeMatch[1];
        rolePart = raw.replace(timePart, "").replace(/\s*\u00b7\s*$/, "").replace(/^\s*\u00b7\s*/, "").trim();
        rolePart = rolePart.replace(/\s*\u00b7\s*$/, "").trim() || raw.split("\u00b7")[0].trim();
      }
      meta.innerHTML = "";
      const role = document.createElement("span");
      role.className = "msg-role";
      role.textContent = rolePart || (b.classList.contains("user") ? "you" : "rabbit");
      meta.appendChild(role);
      if (timePart) {
        const time = document.createElement("span");
        time.className = "msg-time";
        time.textContent = " \u00b7 " + timePart;
        meta.appendChild(time);
      }
    }

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
    const c = (text || "").length;
    chars.textContent = c + (c === 1 ? " char" : " chars");
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
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      ensureMeta(b);
      ensureCollapse(b);

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

  const obs = new MutationObserver(function () {
    syncBtn();
    ensureActions();
    syncStreamingClass();
  });
  obs.observe(chatEl, { childList: true, subtree: true });

  setTimeout(function () {
    ensureActions();
    syncBtn();
  }, 600);
})();
