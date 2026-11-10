/* Scroll-to-bottom, copy, edit, resend, delete message */
(function () {
  const chatEl = document.getElementById("chat");
  const btn = document.getElementById("scroll-bottom");
  const promptEl = document.getElementById("prompt");
  const formEl = document.getElementById("form");
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

  function ensureActions() {
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      let actions = b.querySelector(".bubble-actions");
      if (!actions) {
        actions = document.createElement("div");
        actions.className = "bubble-actions";
        b.appendChild(actions);
      }

      if (!actions.querySelector(".copy-btn")) {
        const copy = document.createElement("button");
        copy.type = "button";
        copy.className = "bubble-act copy-btn";
        copy.textContent = "Copy";
        copy.title = "Copy message";
        copy.addEventListener("click", function (ev) {
          ev.stopPropagation();
          const raw = bubbleText(b);
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(raw).then(function () {
              if (window.rabbitToast) rabbitToast("Copied", "ok");
            });
          }
        });
        actions.appendChild(copy);
      }

      if (b.classList.contains("user") && !actions.querySelector(".edit-btn")) {
        const edit = document.createElement("button");
        edit.type = "button";
        edit.className = "bubble-act edit-btn";
        edit.textContent = "Edit";
        edit.title = "Edit in composer";
        edit.addEventListener("click", function (ev) {
          ev.stopPropagation();
          if (promptEl) {
            promptEl.value = bubbleText(b);
            promptEl.focus();
            promptEl.dispatchEvent(new Event("input", { bubbles: true }));
            if (window.rabbitToast) rabbitToast("Loaded into composer", "ok");
          }
        });
        actions.appendChild(edit);

        const resend = document.createElement("button");
        resend.type = "button";
        resend.className = "bubble-act resend-btn";
        resend.textContent = "Resend";
        resend.title = "Send again";
        resend.addEventListener("click", function (ev) {
          ev.stopPropagation();
          if (!promptEl || !formEl) return;
          promptEl.value = bubbleText(b);
          promptEl.dispatchEvent(new Event("input", { bubbles: true }));
          if (typeof formEl.requestSubmit === "function") formEl.requestSubmit();
          else formEl.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
        });
        actions.appendChild(resend);
      }

      if (!actions.querySelector(".del-btn")) {
        const del = document.createElement("button");
        del.type = "button";
        del.className = "bubble-act del-btn";
        del.textContent = "\u2715";
        del.title = "Remove from view";
        del.addEventListener("click", function (ev) {
          ev.stopPropagation();
          b.remove();
          if (window.rabbitToast) rabbitToast("Message removed", "ok");
          syncBtn();
        });
        actions.appendChild(del);
      }
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
