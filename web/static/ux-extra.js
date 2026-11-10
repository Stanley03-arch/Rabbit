/* Scroll-to-bottom + per-bubble copy */
(function () {
  const chatEl = document.getElementById("chat");
  const btn = document.getElementById("scroll-bottom");
  if (!chatEl) return;

  function nearBottom() {
    return chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 80;
  }

  function syncBtn() {
    if (!btn) return;
    const show = chatEl.scrollHeight > chatEl.clientHeight + 40 && !nearBottom();
    btn.hidden = !show;
  }

  chatEl.addEventListener("scroll", syncBtn, { passive: true });
  const obs = new MutationObserver(function () {
    syncBtn();
    ensureCopyButtons();
  });
  obs.observe(chatEl, { childList: true, subtree: true });
  window.addEventListener("resize", syncBtn);

  if (btn) {
    btn.addEventListener("click", function () {
      chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" });
      setTimeout(syncBtn, 300);
    });
  }

  function ensureCopyButtons() {
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      let actions = b.querySelector(".bubble-actions");
      if (!actions) {
        actions = document.createElement("div");
        actions.className = "bubble-actions";
        b.appendChild(actions);
      }
      if (actions.querySelector(".copy-btn")) return;
      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "bubble-act copy-btn";
      copy.textContent = "Copy";
      copy.title = "Copy message";
      copy.addEventListener("click", function (ev) {
        ev.stopPropagation();
        const body = b.querySelector(".body");
        const raw =
          body && body.dataset && body.dataset.raw != null
            ? body.dataset.raw
            : body
              ? body.textContent
              : "";
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(raw || "").then(function () {
            if (window.rabbitToast) rabbitToast("Copied", "ok");
          });
        }
      });
      actions.appendChild(copy);
    });
  }

  setTimeout(function () {
    ensureCopyButtons();
    syncBtn();
  }, 600);
})();
