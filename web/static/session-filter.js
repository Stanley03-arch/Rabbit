/* Filter session tabs by name + density cycle + paste-as-code */
(function () {
  const bar = document.querySelector(".session-bar");
  const tabsHost = document.getElementById("session-tabs");
  const promptEl = document.getElementById("prompt");

  // Session filter input
  if (bar && tabsHost && !document.getElementById("session-filter")) {
    const input = document.createElement("input");
    input.type = "search";
    input.id = "session-filter";
    input.className = "session-filter";
    input.placeholder = "Filter sessions…";
    input.autocomplete = "off";
    input.title = "Filter session tabs by name";
    // place before search or after new session
    const search = document.getElementById("chat-search");
    if (search) bar.insertBefore(input, search);
    else bar.appendChild(input);

    input.addEventListener("input", function () {
      const q = (input.value || "").trim().toLowerCase();
      tabsHost.querySelectorAll(".session-tab").forEach(function (tab) {
        const name = (tab.textContent || "").toLowerCase();
        tab.style.display = !q || name.indexOf(q) >= 0 ? "" : "none";
      });
    });
  }

  // Ctrl+Shift+D is theme in some builds — use Ctrl+Alt+D for density cycle
  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.altKey && (ev.key === "d" || ev.key === "D")) {
      ev.preventDefault();
      if (!window.rabbitDensity) return;
      const order = ["comfortable", "compact", "dense"];
      const cur = rabbitDensity.current();
      const i = order.indexOf(cur);
      const next = order[(i + 1) % order.length];
      rabbitDensity.apply(next);
      if (window.rabbitToast) rabbitToast("Density: " + next, "ok");
    }
  });

  // Paste as code: Ctrl+Shift+V wraps clipboard/selection as fenced block
  document.addEventListener("keydown", function (ev) {
    if (!(ev.ctrlKey || ev.metaKey) || !ev.shiftKey) return;
    if (ev.key !== "V" && ev.key !== "v") return;
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag !== "TEXTAREA" && tag !== "INPUT") return;
    if (ev.target !== promptEl) return;
    // Let native paste happen first? Instead intercept and wrap
    ev.preventDefault();
    const doWrap = function (text) {
      const t = String(text || "").replace(/\s+$/, "");
      if (!t) return;
      const fence = "```\n" + t + "\n```";
      const start = promptEl.selectionStart || 0;
      const end = promptEl.selectionEnd || 0;
      const val = promptEl.value || "";
      promptEl.value = val.slice(0, start) + fence + val.slice(end);
      const pos = start + fence.length;
      promptEl.setSelectionRange(pos, pos);
      promptEl.dispatchEvent(new Event("input", { bubbles: true }));
      if (window.rabbitToast) rabbitToast("Pasted as code block", "ok");
    };
    if (navigator.clipboard && navigator.clipboard.readText) {
      navigator.clipboard.readText().then(doWrap).catch(function () {
        // fallback: wrap current selection if any
        const sel = (promptEl.value || "").slice(
          promptEl.selectionStart || 0,
          promptEl.selectionEnd || 0
        );
        if (sel) doWrap(sel);
        else if (window.rabbitToast) rabbitToast("Clipboard blocked — select text then Ctrl+Shift+V", "err");
      });
    } else {
      const sel = (promptEl.value || "").slice(
        promptEl.selectionStart || 0,
        promptEl.selectionEnd || 0
      );
      if (sel) doWrap(sel);
    }
  });
})();
