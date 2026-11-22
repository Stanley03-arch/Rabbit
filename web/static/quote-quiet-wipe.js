/* Quote into composer + quiet-hours dim + local storage wipe */
(function () {
  const QUIET_KEY = "rabbit-quiet-hours";
  const chatEl = document.getElementById("chat");
  const promptEl = document.getElementById("prompt");

  function bubbleText(b) {
    const body = b.querySelector(".body");
    if (body && body.dataset && body.dataset.raw != null) return body.dataset.raw;
    return body ? body.textContent : b.textContent || "";
  }

  function quoteInto(b) {
    if (!promptEl || !b) return;
    const text = bubbleText(b).trim();
    if (!text) return;
    const snippet = text.length > 400 ? text.slice(0, 400) + "…" : text;
    const quoted = snippet
      .split("\n")
      .map(function (line) {
        return "> " + line;
      })
      .join("\n");
    const cur = promptEl.value || "";
    promptEl.value = (cur ? cur.replace(/\s+$/, "") + "\n\n" : "") + quoted + "\n\n";
    promptEl.focus();
    promptEl.dispatchEvent(new Event("input", { bubbles: true }));
    try {
      const len = promptEl.value.length;
      promptEl.setSelectionRange(len, len);
    } catch (e) {}
    if (window.rabbitToast) rabbitToast("Quoted into composer", "ok");
  }

  function enhance(b) {
    if (!b || b.dataset.quoteReady === "1") return;
    b.dataset.quoteReady = "1";
    const host = b.querySelector(".bubble-actions") || b;
    if (host.querySelector(".quote-msg-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "bubble-act quote-msg-btn";
    btn.textContent = "Quote";
    btn.title = "Append as blockquote in composer";
    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      quoteInto(b);
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

  // Quiet hours — dim chrome between local 22:00–07:00 if enabled
  function applyQuiet() {
    let on = false;
    try {
      on = localStorage.getItem(QUIET_KEY) === "1";
    } catch (e) {}
    if (!on) {
      document.documentElement.classList.remove("quiet-hours");
      return;
    }
    const h = new Date().getHours();
    const night = h >= 22 || h < 7;
    document.documentElement.classList.toggle("quiet-hours", night);
  }

  const controls = document.querySelector(".panel.controls");
  if (controls && !document.getElementById("quiet-hours")) {
    const label = document.createElement("label");
    label.className = "check";
    label.innerHTML = '<input type="checkbox" id="quiet-hours" /> Quiet hours (dim 22:00–07:00)';
    controls.appendChild(label);
    const el = document.getElementById("quiet-hours");
    try {
      el.checked = localStorage.getItem(QUIET_KEY) === "1";
    } catch (e) {}
    el.addEventListener("change", function () {
      try {
        localStorage.setItem(QUIET_KEY, el.checked ? "1" : "0");
      } catch (e) {}
      applyQuiet();
      if (window.rabbitToast)
        rabbitToast(el.checked ? "Quiet hours on" : "Quiet hours off", "ok");
    });
  }
  applyQuiet();
  setInterval(applyQuiet, 60000);

  // Wipe local Rabbit storage (keys starting with rabbit-)
  function wipeLocal() {
    if (!confirm("Clear all Rabbit local data (sessions, settings, snippets)? This cannot be undone.")) {
      return;
    }
    const keys = [];
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.indexOf("rabbit") === 0) keys.push(k);
      }
      keys.forEach(function (k) {
        localStorage.removeItem(k);
      });
    } catch (e) {}
    if (window.rabbitToast) rabbitToast("Local Rabbit data cleared (" + keys.length + " keys)", "ok");
    setTimeout(function () {
      location.reload();
    }, 600);
  }

  if (controls && !document.getElementById("wipe-local")) {
    const row = document.createElement("div");
    row.className = "btn-row";
    row.innerHTML =
      '<button type="button" class="btn-secondary dangerish" id="wipe-local">Wipe local data</button>';
    controls.appendChild(row);
    document.getElementById("wipe-local").addEventListener("click", wipeLocal);
  }

  window.rabbitQuoteQuiet = {
    quoteInto: quoteInto,
    applyQuiet: applyQuiet,
    wipeLocal: wipeLocal,
  };
})();
