/* Polish: focus click, d download, clear confirm, reduce motion */
(function () {
  const chatEl = document.getElementById("chat");
  if (!chatEl) return;

  const MOTION_KEY = "rabbit-reduce-motion";
  const motionEl = document.getElementById("reduce-motion");

  function loadPref(key, def) {
    try {
      const v = localStorage.getItem(key);
      if (v === null) return def;
      return v === "1";
    } catch (e) {
      return def;
    }
  }

  function applyMotion() {
    const on = loadPref(MOTION_KEY, false);
    document.documentElement.classList.toggle("reduce-motion", on);
    if (motionEl) motionEl.checked = on;
  }
  applyMotion();
  if (motionEl) {
    motionEl.addEventListener("change", function () {
      const on = motionEl.checked;
      try {
        localStorage.setItem(MOTION_KEY, on ? "1" : "0");
      } catch (e) {}
      applyMotion();
      if (window.rabbitToast) rabbitToast("Reduce motion " + (on ? "on" : "off"), "ok");
    });
  }

  // Click a bubble to focus it (for keyboard actions)
  chatEl.addEventListener("click", function (ev) {
    const b = ev.target.closest && ev.target.closest(".bubble");
    if (!b || !chatEl.contains(b)) return;
    // don't steal clicks from action buttons / links
    if (ev.target.closest(".bubble-actions, a, button, .collapse-toggle")) return;
    chatEl.querySelectorAll(".bubble.msg-focus").forEach(function (x) {
      x.classList.remove("msg-focus");
    });
    b.classList.add("msg-focus");
  });

  function focusedBubble() {
    return (
      chatEl.querySelector(".bubble.msg-focus") ||
      Array.from(chatEl.querySelectorAll(".bubble")).slice(-1)[0] ||
      null
    );
  }

  function bubbleText(b) {
    const body = b.querySelector(".body");
    if (!body) return "";
    if (body.dataset && body.dataset.raw != null) return body.dataset.raw;
    return body.textContent || "";
  }

  // Confirm clear when there are several messages
  const clearBtn = document.getElementById("clear-chat");
  if (clearBtn && !clearBtn.dataset.confirmBound) {
    clearBtn.dataset.confirmBound = "1";
    clearBtn.addEventListener(
      "click",
      function (ev) {
        const n = chatEl.querySelectorAll(".bubble").length;
        if (n >= 3) {
          const ok = window.confirm("Clear " + n + " messages from this view?");
          if (!ok) {
            ev.preventDefault();
            ev.stopImmediatePropagation();
          }
        }
      },
      true
    );
  }

  document.addEventListener("keydown", function (ev) {
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;

    if (ev.key === "d" || ev.key === "D") {
      const b = focusedBubble();
      if (!b) return;
      ev.preventDefault();
      const raw = bubbleText(b);
      const role = b.classList.contains("user") ? "user" : "assistant";
      const blob = new Blob([raw], { type: "text/plain;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "rabbit-" + role + "-" + Date.now() + ".txt";
      a.click();
      setTimeout(function () {
        URL.revokeObjectURL(a.href);
      }, 1000);
      if (window.rabbitToast) rabbitToast("Downloaded focused message", "ok");
      return;
    }

    if (ev.key === "?") {
      const open = document.getElementById("help-open");
      if (open) {
        ev.preventDefault();
        open.click();
      }
    }
  });
})();
