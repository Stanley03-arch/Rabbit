/* Rabbit toasts, double-click copy, hide FAB */
(function () {
  const SETTINGS_KEY = "rabbit-dashboard-v1";
  const host = document.getElementById("toast-host");
  const hideFabEl = document.getElementById("hide-fab");
  const fab = document.getElementById("voice-toggle-fab");
  const chatEl = document.getElementById("chat");

  window.rabbitToast = function (msg, kind) {
    if (!host) return;
    const el = document.createElement("div");
    el.className = "toast" + (kind ? " toast-" + kind : "");
    el.textContent = msg;
    host.appendChild(el);
    requestAnimationFrame(function () {
      el.classList.add("show");
    });
    setTimeout(function () {
      el.classList.remove("show");
      setTimeout(function () {
        el.remove();
      }, 280);
    }, 2200);
  };

  function applyHideFab(on) {
    if (fab) fab.style.display = on ? "none" : "";
  }

  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) {
      const s = JSON.parse(raw);
      if (typeof s.hide_fab === "boolean" && hideFabEl) {
        hideFabEl.checked = s.hide_fab;
        applyHideFab(s.hide_fab);
      }
    }
  } catch (e) {}

  if (hideFabEl) {
    hideFabEl.addEventListener("change", function () {
      applyHideFab(hideFabEl.checked);
      try {
        const raw = localStorage.getItem(SETTINGS_KEY);
        const s = raw ? JSON.parse(raw) : {};
        s.hide_fab = hideFabEl.checked;
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
      } catch (e) {}
      if (window.rabbitToast) {
        rabbitToast(hideFabEl.checked ? "Floating mic hidden" : "Floating mic shown", "ok");
      }
    });
  }

  if (chatEl) {
    chatEl.addEventListener("dblclick", function (ev) {
      const bubble = ev.target.closest && ev.target.closest(".bubble");
      if (!bubble) return;
      const body = bubble.querySelector(".body");
      const text = body ? body.textContent : bubble.textContent;
      if (!text) return;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          if (window.rabbitToast) rabbitToast("Copied message", "ok");
        });
      }
    });
  }

  ["export-chat", "export-md", "clear-chat"].forEach(function (id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("click", function () {
      const map = {
        "export-chat": "Exported JSON",
        "export-md": "Exported Markdown",
        "clear-chat": "Chat cleared",
      };
      setTimeout(function () {
        if (window.rabbitToast) rabbitToast(map[id] || "Done", "ok");
      }, 50);
    });
  });

  const mic = document.getElementById("mic");
  if (mic) {
    let last = mic.classList.contains("listening");
    const obs = new MutationObserver(function () {
      const on = mic.classList.contains("listening");
      if (on !== last) {
        last = on;
        if (window.rabbitToast) rabbitToast(on ? "Listening\u2026" : "Mic off", on ? "warn" : "ok");
      }
    });
    obs.observe(mic, { attributes: true, attributeFilter: ["class"] });
  }
})();
