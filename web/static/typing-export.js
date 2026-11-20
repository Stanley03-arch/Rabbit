/* Streaming typing indicator + export reply as .md/.txt + side peek */
(function () {
  const chatEl = document.getElementById("chat");
  if (!chatEl) return;

  // Typing indicator while .streaming exists
  let tip = document.getElementById("typing-indicator");
  if (!tip) {
    tip = document.createElement("div");
    tip.id = "typing-indicator";
    tip.className = "typing-indicator";
    tip.hidden = true;
    tip.innerHTML = '<span class="typing-dots" aria-hidden="true"></span> Rabbit is writing…';
    const composer = document.querySelector(".composer");
    if (composer && composer.parentNode) {
      composer.parentNode.insertBefore(tip, composer);
    } else {
      document.body.appendChild(tip);
    }
  }

  function syncTyping() {
    const streaming = chatEl.querySelector(".bubble.streaming, .bubble.bot.streaming");
    tip.hidden = !streaming;
  }
  new MutationObserver(syncTyping).observe(chatEl, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["class"],
  });
  syncTyping();

  // Export single reply as file
  function exportBubble(b) {
    const body = b.querySelector(".body");
    const text =
      body && body.dataset && body.dataset.raw != null
        ? body.dataset.raw
        : body
          ? body.textContent
          : b.textContent || "";
    const role = b.classList.contains("user") ? "you" : "rabbit";
    const name = "rabbit-" + role + "-" + Date.now() + ".md";
    const blob = new Blob([text], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(function () {
      URL.revokeObjectURL(a.href);
    }, 1000);
    if (window.rabbitToast) rabbitToast("Downloaded " + name, "ok");
  }

  function enhance(b) {
    if (!b || b.dataset.exportReady === "1") return;
    b.dataset.exportReady = "1";
    const host = b.querySelector(".bubble-actions") || b;
    if (host.querySelector(".export-msg-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "bubble-act export-msg-btn";
    btn.textContent = "Save";
    btn.title = "Download this message as .md";
    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      exportBubble(b);
    });
    host.appendChild(btn);
  }

  function scan() {
    chatEl.querySelectorAll(".bubble").forEach(enhance);
  }
  new MutationObserver(scan).observe(chatEl, { childList: true, subtree: true });
  scan();

  // Side peek: float last bot reply in a small panel (toggle)
  let peek = document.getElementById("side-peek");
  if (!peek) {
    peek = document.createElement("div");
    peek.id = "side-peek";
    peek.className = "side-peek";
    peek.hidden = true;
    peek.innerHTML =
      '<header><strong>Last reply</strong><button type="button" id="side-peek-close" aria-label="Close">×</button></header>' +
      '<div class="side-peek-body" id="side-peek-body"></div>';
    document.body.appendChild(peek);
    document.getElementById("side-peek-close").addEventListener("click", function () {
      peek.hidden = true;
    });
  }

  function updatePeek() {
    if (peek.hidden) return;
    const bots = chatEl.querySelectorAll(".bubble.bot:not(.streaming)");
    const last = bots[bots.length - 1];
    const body = document.getElementById("side-peek-body");
    if (!body) return;
    if (!last) {
      body.textContent = "(no reply yet)";
      return;
    }
    const src = last.querySelector(".body");
    body.textContent = src ? src.textContent : last.textContent;
  }

  function togglePeek() {
    peek.hidden = !peek.hidden;
    updatePeek();
  }

  const top = document.querySelector(".top-right");
  if (top && !document.getElementById("side-peek-open")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "side-peek-open";
    btn.className = "btn-peek";
    btn.title = "Peek last reply (Ctrl+Shift+P)";
    btn.textContent = "Peek";
    top.insertBefore(btn, top.firstChild);
    btn.addEventListener("click", togglePeek);
  }

  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.shiftKey && (ev.key === "p" || ev.key === "P")) {
      // avoid conflict with pin session Ctrl+Alt+P
      ev.preventDefault();
      togglePeek();
    }
  });

  new MutationObserver(function () {
    updatePeek();
  }).observe(chatEl, { childList: true, subtree: true });

  window.rabbitTypingExport = {
    exportBubble: exportBubble,
    togglePeek: togglePeek,
  };
})();
