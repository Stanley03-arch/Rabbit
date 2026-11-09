/* Markdown-lite bot bubbles, chat search */
(function () {
  const chatEl = document.getElementById("chat");
  const searchEl = document.getElementById("chat-search");
  if (!chatEl) return;

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderMarkdown(raw) {
    let s = escapeHtml(raw || "");
    s = s.replace(/```([\s\S]*?)```/g, function (_, code) {
      return '<pre class="md-code"><code>' + code.replace(/^\n/, "") + "</code></pre>";
    });
    s = s.replace(/`([^`]+)`/g, '<code class="md-inline">$1</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^\*])\*([^*]+)\*/g, "$1<em>$2</em>");
    s = s.replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    s = s.replace(/\n/g, "<br>");
    return s;
  }

  function formatBubble(body) {
    if (!body || body.dataset.md === "1") return;
    if (body.closest(".bubble.user")) return;
    const raw = body.dataset.raw != null ? body.dataset.raw : body.textContent;
    body.dataset.raw = raw;
    body.innerHTML = renderMarkdown(raw);
    body.dataset.md = "1";
  }

  function formatAll() {
    chatEl.querySelectorAll(".bubble.bot .body").forEach(formatBubble);
  }

  const statusPill = document.getElementById("status-pill");
  if (statusPill) {
    let last = statusPill.textContent || "";
    const so = new MutationObserver(function () {
      const now = statusPill.textContent || "";
      if (/generat|busy|stream/i.test(last) && /ready|ok|idle/i.test(now)) {
        setTimeout(formatAll, 30);
      }
      last = now;
    });
    so.observe(statusPill, { childList: true, characterData: true, subtree: true });
  }

  const obs = new MutationObserver(function (muts) {
    muts.forEach(function (m) {
      m.addedNodes.forEach(function (n) {
        if (n.nodeType !== 1) return;
        if (n.classList && n.classList.contains("bubble") && n.classList.contains("bot")) {
          const body = n.querySelector(".body");
          setTimeout(function () {
            if (body && body.textContent && body.textContent !== "\u2026") formatBubble(body);
          }, 400);
        }
      });
    });
  });
  obs.observe(chatEl, { childList: true });

  setTimeout(formatAll, 600);

  if (searchEl) {
    searchEl.addEventListener("input", function () {
      const q = (searchEl.value || "").trim().toLowerCase();
      chatEl.querySelectorAll(".bubble").forEach(function (b) {
        if (!q) {
          b.style.display = "";
          return;
        }
        const t = (b.textContent || "").toLowerCase();
        b.style.display = t.indexOf(q) >= 0 ? "" : "none";
      });
    });
  }

  window.rabbitFormatMarkdown = formatAll;
})();
