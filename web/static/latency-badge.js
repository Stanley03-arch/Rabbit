/* Live latency badge near status pill + copy last reply helper */
(function () {
  const latencyEl = document.getElementById("latency");
  const tpsEl = document.getElementById("tps");
  const topRight = document.querySelector(".top-right");
  if (!topRight) return;

  let badge = document.getElementById("latency-badge");
  if (!badge) {
    badge = document.createElement("span");
    badge.id = "latency-badge";
    badge.className = "latency-badge";
    badge.title = "Last generation latency / speed";
    badge.textContent = "—";
    const pill = document.getElementById("status-pill");
    if (pill && pill.parentNode) pill.parentNode.insertBefore(badge, pill);
    else topRight.appendChild(badge);
  }

  function sync() {
    const lat = latencyEl ? (latencyEl.textContent || "").trim() : "";
    const tps = tpsEl ? (tpsEl.textContent || "").trim() : "";
    if ((!lat || lat === "—") && (!tps || tps === "—")) {
      badge.textContent = "—";
      badge.dataset.kind = "idle";
      return;
    }
    const parts = [];
    if (lat && lat !== "—") parts.push(lat);
    if (tps && tps !== "—") parts.push(tps);
    badge.textContent = parts.join(" · ");
    badge.dataset.kind = "ok";
  }

  if (latencyEl) {
    new MutationObserver(sync).observe(latencyEl, {
      childList: true,
      characterData: true,
      subtree: true,
    });
  }
  if (tpsEl) {
    new MutationObserver(sync).observe(tpsEl, {
      childList: true,
      characterData: true,
      subtree: true,
    });
  }
  setInterval(sync, 1000);
  sync();

  function copyLastReply() {
    const chat = document.getElementById("chat");
    if (!chat) return;
    const bots = chat.querySelectorAll(".bubble.bot");
    const last = bots[bots.length - 1];
    if (!last) {
      if (window.rabbitToast) rabbitToast("No bot reply yet", "ok");
      return;
    }
    const body = last.querySelector(".body");
    const text =
      body && body.dataset && body.dataset.raw != null
        ? body.dataset.raw
        : body
          ? body.textContent
          : "";
    const t = (text || "").trim();
    if (!t) return;
    const done = function () {
      if (window.rabbitToast) rabbitToast("Last reply copied", "ok");
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(t).then(done);
    } else {
      const ta = document.createElement("textarea");
      ta.value = t;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      done();
    }
  }

  function exportShortcuts() {
    const text = [
      "Rabbit keyboard shortcuts",
      "========================",
      "Ctrl+K          Command palette",
      "Ctrl+Shift+F    Favorite current prompt",
      "Ctrl+Shift+S    Force save chat",
      "Ctrl+Shift+E    Export all sessions",
      "Ctrl+Shift+P    Toggle persist",
      "Ctrl+Shift+U    Duplicate session",
      "Ctrl+Shift+D    Toggle dark mode",
      "Ctrl+M          Toggle mic",
      "Ctrl+/- / 0     UI scale / reset",
      "Alt+[ / ]       Previous / next session",
      "1 / 2 / 3       Filter All / You / Rabbit",
      "j / k           Focus prev / next message",
      "c / q / e / d   Copy / quote / edit / download focused",
      "r               Regenerate",
      "n               New session",
      "/               Search",
      "?               Help",
      "",
      "Also: send while busy queues; Clear is undoable via toast;",
      "drop text files into the window; click sidebar panel titles to collapse.",
      "",
    ].join("\n");
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "rabbit-shortcuts.txt";
    a.click();
    setTimeout(function () {
      URL.revokeObjectURL(a.href);
    }, 1000);
    if (window.rabbitToast) rabbitToast("Shortcuts exported", "ok");
  }

  // Expose for palette
  window.rabbitLatencyBadge = { sync: sync, copyLastReply: copyLastReply, exportShortcuts: exportShortcuts };

  // Patch palette commands if it rebuilds from global hooks — inject via open wrapper
  const prev = window.rabbitCommandPalette;
  if (prev && typeof prev.open === "function") {
    // palette already lists fixed commands; add a small extra menu via key
  }

  document.addEventListener("keydown", function (ev) {
    if (!(ev.ctrlKey || ev.metaKey) || !ev.shiftKey) return;
    if (ev.key === "C" || ev.key === "c") {
      // avoid clashing with browser hard refresh habits; only when not in input
      const tag = (ev.target && ev.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      // Ctrl+Shift+C often used by devtools — skip if risk
      // use Ctrl+Shift+L for last reply instead
    }
    if (ev.key === "L" || ev.key === "l") {
      const tag = (ev.target && ev.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      ev.preventDefault();
      copyLastReply();
    }
    if (ev.key === "H" || ev.key === "h") {
      const tag = (ev.target && ev.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      ev.preventDefault();
      exportShortcuts();
    }
  });
})();
