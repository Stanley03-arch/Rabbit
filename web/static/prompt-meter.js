/* Prompt length meter + reuse last user + header metrics chip */
(function () {
  const promptEl = document.getElementById("prompt");
  const tokensEl = document.getElementById("tokens");
  const formEl = document.getElementById("form");
  const chatEl = document.getElementById("chat");

  // Length meter under composer tools
  let meter = document.getElementById("prompt-meter");
  if (!meter) {
    meter = document.createElement("div");
    meter.id = "prompt-meter";
    meter.className = "prompt-meter";
    meter.innerHTML = '<div class="prompt-meter-fill" id="prompt-meter-fill"></div>';
    const tools = document.querySelector(".composer-tools");
    if (tools) tools.insertAdjacentElement("afterend", meter);
    else if (formEl) formEl.insertAdjacentElement("beforebegin", meter);
  }
  const fill = document.getElementById("prompt-meter-fill");

  function updateMeter() {
    if (!promptEl || !fill) return;
    const chars = (promptEl.value || "").length;
    const est = Math.ceil(chars / 4);
    const budget = tokensEl ? parseInt(tokensEl.value, 10) || 48 : 48;
    // scale: full bar at 2x generation budget (prompt pressure)
    const ratio = Math.min(1, est / Math.max(1, budget * 2));
    fill.style.width = (ratio * 100).toFixed(1) + "%";
    fill.classList.remove("warn", "over");
    if (est >= budget) fill.classList.add("over");
    else if (est >= budget * 0.7) fill.classList.add("warn");
  }

  if (promptEl) {
    promptEl.addEventListener("input", updateMeter);
    if (tokensEl) {
      tokensEl.addEventListener("input", updateMeter);
      tokensEl.addEventListener("change", updateMeter);
    }
    updateMeter();
  }

  // Reuse last user message into composer
  function lastUserText() {
    if (!chatEl) return "";
    const users = chatEl.querySelectorAll(".bubble.user");
    const last = users[users.length - 1];
    if (!last) return "";
    const body = last.querySelector(".body");
    if (body && body.dataset && body.dataset.raw != null) return body.dataset.raw;
    return body ? body.textContent : last.textContent || "";
  }

  function reuseLastUser() {
    const t = lastUserText().trim();
    if (!t) {
      if (window.rabbitToast) rabbitToast("No user message yet", "ok");
      return;
    }
    if (!promptEl) return;
    promptEl.value = t;
    promptEl.focus();
    promptEl.dispatchEvent(new Event("input", { bubbles: true }));
    if (window.rabbitToast) rabbitToast("Loaded last user message", "ok");
  }

  const tools = document.querySelector(".composer-tools");
  if (tools && !document.getElementById("reuse-last-user")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "reuse-last-user";
    btn.className = "linkish";
    btn.textContent = "Reuse last";
    btn.title = "Put last user message back in the composer (Ctrl+Shift+U)";
    btn.addEventListener("click", reuseLastUser);
    tools.appendChild(btn);
  }

  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.shiftKey && (ev.key === "u" || ev.key === "U")) {
      ev.preventDefault();
      reuseLastUser();
    }
  });

  // Header metrics chip — compact live summary
  const top = document.querySelector(".top-right");
  let chip = document.getElementById("metrics-chip");
  if (top && !chip) {
    chip = document.createElement("button");
    chip.type = "button";
    chip.id = "metrics-chip";
    chip.className = "metrics-chip";
    chip.title = "Session metrics (click to refresh stats)";
    chip.textContent = "—";
    top.insertBefore(chip, top.firstChild);
    chip.addEventListener("click", function () {
      const b = document.getElementById("refresh-stats");
      if (b) b.click();
    });
  }

  function syncChip() {
    if (!chip) return;
    const req = document.getElementById("met-req");
    const avg = document.getElementById("met-avg");
    const tps = document.getElementById("met-tps");
    const r = req ? req.textContent : "0";
    const a = avg ? avg.textContent : "—";
    const s = tps ? tps.textContent : "—";
    chip.textContent = r + " req · " + a + " · " + s;
  }

  ["met-req", "met-avg", "met-tps", "latency", "tps"].forEach(function (id) {
    const el = document.getElementById(id);
    if (!el) return;
    new MutationObserver(syncChip).observe(el, {
      childList: true,
      characterData: true,
      subtree: true,
    });
  });
  setInterval(syncChip, 2000);
  syncChip();

  window.rabbitPromptMeter = {
    update: updateMeter,
    reuseLastUser: reuseLastUser,
  };
})();
