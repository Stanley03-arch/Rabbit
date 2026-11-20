/* Pin messages strip + copy all code + offline draft backup */
(function () {
  const PIN_KEY = "rabbit-pinned-msgs-v1";
  const DRAFT_KEY = "rabbit-composer-draft-backup";
  const chatEl = document.getElementById("chat");
  const promptEl = document.getElementById("prompt");
  let strip = document.getElementById("pinned-strip");

  function loadPins() {
    try {
      const raw = localStorage.getItem(PIN_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function savePins(arr) {
    try {
      localStorage.setItem(PIN_KEY, JSON.stringify(arr.slice(0, 40)));
    } catch (e) {}
  }

  function bubbleText(b) {
    const body = b.querySelector(".body");
    if (body && body.dataset && body.dataset.raw != null) return body.dataset.raw;
    return body ? body.textContent : b.textContent || "";
  }

  function renderPins() {
    if (!strip) {
      strip = document.createElement("div");
      strip.id = "pinned-strip";
      strip.className = "pinned-strip";
      const chips = document.getElementById("chips");
      if (chips && chips.parentNode) chips.parentNode.insertBefore(strip, chips.nextSibling);
      else if (chatEl) chatEl.parentNode.insertBefore(strip, chatEl);
    }
    const pins = loadPins();
    if (!pins.length) {
      strip.hidden = true;
      strip.innerHTML = "";
      return;
    }
    strip.hidden = false;
    strip.innerHTML =
      '<span class="pinned-label">Pinned</span>' +
      pins
        .map(function (p, i) {
          const label = (p.text || "").replace(/\s+/g, " ").slice(0, 48);
          return (
            '<button type="button" class="pin-chip" data-i="' +
            i +
            '" title="' +
            escapeAttr(p.text || "") +
            '">' +
            escapeHtml(label) +
            (label.length >= 48 ? "…" : "") +
            "</button>" +
            '<button type="button" class="pin-x" data-x="' +
            i +
            '" title="Unpin">×</button>'
          );
        })
        .join("");
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

  if (strip || document.body) {
    document.body.addEventListener("click", function (ev) {
      const chip = ev.target.closest(".pin-chip");
      if (chip && strip && strip.contains(chip)) {
        const pins = loadPins();
        const p = pins[+chip.dataset.i];
        if (p && promptEl) {
          promptEl.value = p.text || "";
          promptEl.focus();
          promptEl.dispatchEvent(new Event("input", { bubbles: true }));
        }
        return;
      }
      const x = ev.target.closest(".pin-x");
      if (x && strip && strip.contains(x)) {
        const pins = loadPins();
        pins.splice(+x.dataset.x, 1);
        savePins(pins);
        renderPins();
      }
    });
  }

  function enhanceBubble(b) {
    if (!b || b.dataset.pinReady === "1") return;
    b.dataset.pinReady = "1";
    const host = b.querySelector(".bubble-actions") || b;
    if (host.querySelector(".pin-msg-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "bubble-act pin-msg-btn";
    btn.textContent = "Pin";
    btn.title = "Pin this message";
    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      const text = bubbleText(b).trim();
      if (!text) return;
      const pins = loadPins().filter(function (p) {
        return p.text !== text;
      });
      pins.unshift({ text: text, t: Date.now() });
      savePins(pins);
      renderPins();
      if (window.rabbitToast) rabbitToast("Pinned", "ok");
    });
    host.appendChild(btn);
  }

  function scan() {
    if (!chatEl) return;
    chatEl.querySelectorAll(".bubble").forEach(enhanceBubble);
  }
  if (chatEl) {
    new MutationObserver(scan).observe(chatEl, { childList: true, subtree: true });
    scan();
  }
  renderPins();

  // Copy all code blocks in chat
  function copyAllCode() {
    if (!chatEl) return;
    const blocks = chatEl.querySelectorAll("pre code, pre");
    const parts = [];
    blocks.forEach(function (el) {
      const t = (el.textContent || "").trim();
      if (t && parts.indexOf(t) < 0) parts.push(t);
    });
    if (!parts.length) {
      if (window.rabbitToast) rabbitToast("No code blocks", "ok");
      return;
    }
    const text = parts.join("\n\n---\n\n");
    const done = function () {
      if (window.rabbitToast) rabbitToast("Copied " + parts.length + " code block(s)", "ok");
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

  const tools = document.querySelector(".composer-tools");
  if (tools && !document.getElementById("copy-all-code")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "copy-all-code";
    btn.className = "linkish";
    btn.textContent = "Copy all code";
    btn.title = "Copy every code block in this chat";
    btn.addEventListener("click", copyAllCode);
    tools.appendChild(btn);
  }

  // Offline / crash draft backup
  if (promptEl) {
    try {
      const saved = localStorage.getItem(DRAFT_KEY);
      if (saved && !promptEl.value) {
        promptEl.value = saved;
        promptEl.dispatchEvent(new Event("input", { bubbles: true }));
      }
    } catch (e) {}
    let timer;
    promptEl.addEventListener("input", function () {
      clearTimeout(timer);
      timer = setTimeout(function () {
        try {
          localStorage.setItem(DRAFT_KEY, promptEl.value || "");
        } catch (e) {}
      }, 300);
    });
    const formEl = document.getElementById("form");
    if (formEl) {
      formEl.addEventListener("submit", function () {
        try {
          localStorage.removeItem(DRAFT_KEY);
        } catch (e) {}
      });
    }
  }

  window.rabbitPins = {
    list: loadPins,
    copyAllCode: copyAllCode,
    render: renderPins,
  };
})();
