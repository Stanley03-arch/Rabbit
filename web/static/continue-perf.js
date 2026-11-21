/* Continue-from-here, optional perf HUD, composer height memory */
(function () {
  const HEIGHT_KEY = "rabbit-composer-height";
  const PERF_KEY = "rabbit-perf-hud";
  const chatEl = document.getElementById("chat");
  const promptEl = document.getElementById("prompt");
  const formEl = document.getElementById("form");

  // --- Continue from this user/bot turn into composer ---
  function bubbleText(b) {
    const body = b.querySelector(".body");
    if (body && body.dataset && body.dataset.raw != null) return body.dataset.raw;
    return body ? body.textContent : b.textContent || "";
  }

  function continueFrom(b) {
    if (!promptEl || !b) return;
    const text = bubbleText(b).trim();
    if (!text) return;
    const isUser = b.classList.contains("user");
    if (isUser) {
      promptEl.value = text;
    } else {
      // seed a follow-up that references the reply
      const snippet = text.length > 280 ? text.slice(0, 280) + "…" : text;
      promptEl.value = "Continue from this:\n\n> " + snippet.replace(/\n/g, "\n> ") + "\n\n";
    }
    promptEl.focus();
    promptEl.dispatchEvent(new Event("input", { bubbles: true }));
    const len = promptEl.value.length;
    try {
      promptEl.setSelectionRange(len, len);
    } catch (e) {}
    if (window.rabbitToast) rabbitToast(isUser ? "Loaded into composer" : "Continue prompt seeded", "ok");
  }

  function enhance(b) {
    if (!b || b.dataset.contReady === "1") return;
    b.dataset.contReady = "1";
    const host = b.querySelector(".bubble-actions") || b;
    if (host.querySelector(".continue-msg-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "bubble-act continue-msg-btn";
    btn.textContent = "Continue";
    btn.title = "Seed composer from this message";
    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      continueFrom(b);
    });
    host.appendChild(btn);
  }

  function scan() {
    if (!chatEl) return;
    chatEl.querySelectorAll(".bubble").forEach(enhance);
  }
  if (chatEl) {
    new MutationObserver(scan).observe(chatEl, { childList: true, subtree: true });
    scan();
  }

  // --- Composer height memory ---
  if (promptEl) {
    try {
      const h = parseInt(localStorage.getItem(HEIGHT_KEY), 10);
      if (h >= 48 && h <= 320) promptEl.style.height = h + "px";
    } catch (e) {}

    let ro;
    if (window.ResizeObserver) {
      ro = new ResizeObserver(function (entries) {
        const h = Math.round(entries[0].contentRect.height);
        if (h >= 48 && h <= 320) {
          try {
            localStorage.setItem(HEIGHT_KEY, String(h));
          } catch (e) {}
        }
      });
      ro.observe(promptEl);
    }
  }

  // --- Optional perf HUD (fps + heap if available) ---
  let hud = document.getElementById("perf-hud");
  if (!hud) {
    hud = document.createElement("div");
    hud.id = "perf-hud";
    hud.className = "perf-hud";
    hud.hidden = true;
    hud.textContent = "— fps";
    document.body.appendChild(hud);
  }

  let frames = 0;
  let lastT = performance.now();
  let rafId = 0;

  function tick(t) {
    frames++;
    if (t - lastT >= 1000) {
      const fps = Math.round((frames * 1000) / (t - lastT));
      let mem = "";
      try {
        if (performance.memory) {
          mem = " · " + Math.round(performance.memory.usedJSHeapSize / 1048576) + " MB";
        }
      } catch (e) {}
      hud.textContent = fps + " fps" + mem;
      frames = 0;
      lastT = t;
    }
    if (!hud.hidden) rafId = requestAnimationFrame(tick);
  }

  function setPerf(on) {
    hud.hidden = !on;
    try {
      localStorage.setItem(PERF_KEY, on ? "1" : "0");
    } catch (e) {}
    if (on) {
      frames = 0;
      lastT = performance.now();
      rafId = requestAnimationFrame(tick);
    } else if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = 0;
    }
  }

  try {
    if (localStorage.getItem(PERF_KEY) === "1") setPerf(true);
  } catch (e) {}

  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.shiftKey && (ev.key === "f" || ev.key === "F")) {
      const tag = (ev.target && ev.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      ev.preventDefault();
      setPerf(hud.hidden);
      if (window.rabbitToast) rabbitToast(hud.hidden ? "Perf HUD off" : "Perf HUD on", "ok");
    }
  });

  // Toggle in help-adjacent area
  const help = document.getElementById("help-open");
  if (help && !document.getElementById("perf-toggle")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "perf-toggle";
    btn.className = "btn-perf";
    btn.title = "Toggle FPS HUD (Ctrl+Shift+F)";
    btn.textContent = "FPS";
    help.parentNode.insertBefore(btn, help);
    btn.addEventListener("click", function () {
      setPerf(hud.hidden);
    });
  }

  window.rabbitContinuePerf = {
    continueFrom: continueFrom,
    setPerf: setPerf,
  };
})();
