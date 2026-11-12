/* Extra chat UX: regen, new session, jump-top, composer warn */
(function () {
  const chatEl = document.getElementById("chat");
  const promptEl = document.getElementById("prompt");
  const countEl = document.getElementById("composer-count");
  if (!chatEl) return;

  // Jump to top FAB
  let topBtn = document.getElementById("scroll-top");
  if (!topBtn) {
    topBtn = document.createElement("button");
    topBtn.type = "button";
    topBtn.id = "scroll-top";
    topBtn.className = "btn-scroll-top";
    topBtn.title = "Jump to top";
    topBtn.textContent = "\u2191";
    topBtn.hidden = true;
    document.body.appendChild(topBtn);
  }

  function syncTopBtn() {
    const show = chatEl.scrollTop > 200;
    topBtn.hidden = !show;
  }
  chatEl.addEventListener("scroll", syncTopBtn, { passive: true });
  topBtn.addEventListener("click", function () {
    chatEl.scrollTo({ top: 0, behavior: "smooth" });
  });
  syncTopBtn();

  // Composer length warning (amber > 1500, red > 4000)
  function updateComposerWarn() {
    if (!promptEl || !countEl) return;
    const n = (promptEl.value || "").length;
    countEl.classList.remove("warn", "danger");
    if (n > 4000) countEl.classList.add("danger");
    else if (n > 1500) countEl.classList.add("warn");
  }
  if (promptEl) {
    promptEl.addEventListener("input", updateComposerWarn);
    updateComposerWarn();
  }

  document.addEventListener("keydown", function (ev) {
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;

    if (ev.key === "r" || ev.key === "R") {
      const regen = document.getElementById("regen-last");
      if (regen) {
        ev.preventDefault();
        regen.click();
        if (window.rabbitToast) rabbitToast("Regenerate", "ok");
      }
      return;
    }
    if (ev.key === "n" || ev.key === "N") {
      const neu = document.getElementById("session-new");
      if (neu) {
        ev.preventDefault();
        neu.click();
        if (window.rabbitToast) rabbitToast("New session", "ok");
      }
      return;
    }
    if (ev.key === "Home") {
      ev.preventDefault();
      chatEl.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    if (ev.key === "End") {
      ev.preventDefault();
      chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" });
      return;
    }
  });
})();
