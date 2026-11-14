/* Message queue while generating + undo clear */
(function () {
  const chatEl = document.getElementById("chat");
  const formEl = document.getElementById("form");
  const promptEl = document.getElementById("prompt");
  const sendBtn = document.getElementById("send");
  const clearBtn = document.getElementById("clear-chat");
  const statusPill = document.getElementById("status-pill");
  if (!formEl || !promptEl) return;

  const queue = [];
  let queueEl = document.getElementById("msg-queue-bar");
  if (!queueEl) {
    queueEl = document.createElement("div");
    queueEl.id = "msg-queue-bar";
    queueEl.className = "msg-queue-bar";
    queueEl.hidden = true;
    const tools = document.querySelector(".composer-tools");
    if (tools && tools.parentNode) {
      tools.parentNode.insertBefore(queueEl, tools);
    } else {
      document.body.appendChild(queueEl);
    }
  }

  function isBusy() {
    if (statusPill) {
      const t = (statusPill.textContent || "").toLowerCase();
      if (t.indexOf("generat") >= 0 || t.indexOf("stream") >= 0 || t.indexOf("busy") >= 0 || t === "running") {
        return true;
      }
    }
    if (document.querySelector(".bubble.bot.streaming")) return true;
    const stop = document.getElementById("stop");
    if (stop && !stop.hidden) return true;
    return false;
  }

  function renderQueue() {
    if (!queue.length) {
      queueEl.hidden = true;
      queueEl.innerHTML = "";
      return;
    }
    queueEl.hidden = false;
    queueEl.innerHTML =
      '<span class="msg-queue-label">Queued (' +
      queue.length +
      ")</span>" +
      queue
        .map(function (t, i) {
          return (
            '<span class="msg-queue-item" data-i="' +
            i +
            '" title="Click to remove">' +
            escapeHtml(t.slice(0, 48)) +
            (t.length > 48 ? "…" : "") +
            " ×</span>"
          );
        })
        .join("");
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  queueEl.addEventListener("click", function (ev) {
    const item = ev.target.closest(".msg-queue-item");
    if (!item) return;
    const i = +item.dataset.i;
    if (!isNaN(i)) {
      queue.splice(i, 1);
      renderQueue();
    }
  });

  function flushNext() {
    if (!queue.length) return;
    if (isBusy()) return;
    const next = queue.shift();
    renderQueue();
    promptEl.value = next;
    promptEl.dispatchEvent(new Event("input", { bubbles: true }));
    // let the app pick up the value then submit
    setTimeout(function () {
      if (typeof formEl.requestSubmit === "function") formEl.requestSubmit();
      else formEl.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    }, 30);
  }

  // Intercept submit when busy → queue
  formEl.addEventListener(
    "submit",
    function (ev) {
      const text = (promptEl.value || "").trim();
      if (!text) return;
      if (!isBusy()) return; // normal send
      ev.preventDefault();
      ev.stopImmediatePropagation();
      queue.push(text);
      promptEl.value = "";
      promptEl.dispatchEvent(new Event("input", { bubbles: true }));
      renderQueue();
      if (window.rabbitToast) rabbitToast("Queued (" + queue.length + ")", "ok");
    },
    true
  );

  // Watch for idle to flush queue
  const obs = new MutationObserver(function () {
    setTimeout(flushNext, 120);
  });
  if (statusPill) obs.observe(statusPill, { childList: true, characterData: true, subtree: true });
  if (chatEl) obs.observe(chatEl, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });
  setInterval(flushNext, 800);

  // --- Undo clear ---
  let clearSnapshot = null;
  if (clearBtn) {
    clearBtn.addEventListener(
      "mousedown",
      function () {
        if (!chatEl) return;
        const msgs = Array.from(chatEl.querySelectorAll(".bubble")).map(function (div) {
          const role = div.classList.contains("user") ? "user" : "bot";
          const body = div.querySelector(".body");
          const content =
            body && body.dataset && body.dataset.raw != null
              ? body.dataset.raw
              : body
                ? body.textContent
                : "";
          return { role: role, content: content || "" };
        });
        clearSnapshot = msgs.length ? msgs : null;
      },
      true
    );

    clearBtn.addEventListener("click", function () {
      if (!clearSnapshot || !clearSnapshot.length) return;
      const snap = clearSnapshot;
      clearSnapshot = null;
      setTimeout(function () {
        if (chatEl && chatEl.querySelector(".bubble")) return; // wasn't cleared
        if (!window.rabbitToast) return;
        rabbitToast("Chat cleared — click to undo", "ok");
        const host = document.getElementById("toast-host");
        if (!host) return;
        const last = host.lastElementChild;
        if (!last) return;
        last.style.cursor = "pointer";
        const undo = function () {
          last.removeEventListener("click", undo);
          const empty = document.getElementById("empty-state");
          if (empty) empty.style.display = "none";
          snap.forEach(function (m) {
            const div = document.createElement("div");
            div.className = "bubble " + (m.role === "user" ? "user" : "bot");
            const meta = document.createElement("div");
            meta.className = "meta";
            meta.textContent = m.role === "user" ? "you" : "rabbit";
            const body = document.createElement("div");
            body.className = "body";
            body.dataset.raw = m.content || "";
            body.textContent = m.content || "";
            div.appendChild(meta);
            div.appendChild(body);
            chatEl.appendChild(div);
          });
          if (window.rabbitPersist) rabbitPersist.save();
          if (window.rabbitToast) rabbitToast("Restored " + snap.length + " messages", "ok");
        };
        last.addEventListener("click", undo);
        setTimeout(function () {
          last.removeEventListener("click", undo);
        }, 8000);
      }, 80);
    });
  }

  // Reading time helper for favorites/stats
  const prevStats = window.rabbitFavorites && window.rabbitFavorites.stats;
  if (window.rabbitFavorites && prevStats) {
    window.rabbitFavorites.stats = function () {
      const s = prevStats();
      const words = Math.max(1, Math.round((s.chars || 0) / 5));
      s.readingMin = Math.max(1, Math.round(words / 200));
      return s;
    };
  }

  window.rabbitMsgQueue = {
    list: function () {
      return queue.slice();
    },
    clear: function () {
      queue.length = 0;
      renderQueue();
    },
  };
})();
