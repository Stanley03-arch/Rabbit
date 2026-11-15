/* Lightweight ``` fence → <pre> so code Copy works; safe text only */
(function () {
  const chatEl = document.getElementById("chat");
  if (!chatEl) return;

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderFences(text) {
    const raw = text || "";
    if (raw.indexOf("```") < 0) return null;
    const parts = raw.split(/```/);
    let html = "";
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 0) {
        // prose
        const prose = escapeHtml(parts[i]).replace(/\n/g, "<br>");
        html += prose;
      } else {
        // code: optional lang on first line
        let block = parts[i];
        let lang = "";
        const nl = block.indexOf("\n");
        if (nl >= 0) {
          const first = block.slice(0, nl).trim();
          if (/^[a-zA-Z0-9_+.-]+$/.test(first)) {
            lang = first;
            block = block.slice(nl + 1);
          }
        }
        // trim one trailing newline common in fences
        if (block.endsWith("\n")) block = block.slice(0, -1);
        html +=
          '<pre class="md-pre"' +
          (lang ? ' data-lang="' + escapeHtml(lang) + '"' : "") +
          "><code>" +
          escapeHtml(block) +
          "</code></pre>";
      }
    }
    return html;
  }

  function enhance(body) {
    if (!body || body.dataset.mdLite === "1") return;
    const raw =
      body.dataset.raw != null ? body.dataset.raw : body.textContent || "";
    if (raw.indexOf("```") < 0) {
      body.dataset.mdLite = "1";
      return;
    }
    // don't reprocess while streaming
    const bubble = body.closest(".bubble");
    if (bubble && bubble.classList.contains("streaming")) return;
    const html = renderFences(raw);
    if (!html) {
      body.dataset.mdLite = "1";
      return;
    }
    body.dataset.raw = raw;
    body.innerHTML = html;
    body.dataset.mdLite = "1";
    // allow code-copy to re-scan
    if (bubble) delete bubble.dataset.codeCopy;
  }

  function scan() {
    chatEl.querySelectorAll(".bubble .body").forEach(enhance);
  }

  const obs = new MutationObserver(function () {
    scan();
  });
  obs.observe(chatEl, { childList: true, subtree: true });
  scan();
  // late pass after stream ends
  setInterval(scan, 1500);
})();
