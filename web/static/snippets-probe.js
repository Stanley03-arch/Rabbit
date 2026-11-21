/* Snippets library + Esc clears composer + lightweight latency probe */
(function () {
  const SNIP_KEY = "rabbit-snippets-v1";
  const promptEl = document.getElementById("prompt");
  const formEl = document.getElementById("form");

  function loadSnips() {
    try {
      const raw = localStorage.getItem(SNIP_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function saveSnips(arr) {
    try {
      localStorage.setItem(SNIP_KEY, JSON.stringify(arr.slice(0, 40)));
    } catch (e) {}
  }

  // Snippets panel
  if (!document.getElementById("snippets-panel")) {
    const panel = document.createElement("div");
    panel.id = "snippets-panel";
    panel.className = "snippets-panel";
    panel.hidden = true;
    panel.innerHTML =
      '<header><strong>Snippets</strong><button type="button" id="snip-close" aria-label="Close">×</button></header>' +
      '<div id="snip-list" class="snip-list"></div>' +
      '<div class="snip-actions">' +
      '<button type="button" id="snip-save-current" class="btn-secondary">Save composer</button>' +
      "</div>";
    document.body.appendChild(panel);

    document.getElementById("snip-close").addEventListener("click", function () {
      panel.hidden = true;
    });
    document.getElementById("snip-save-current").addEventListener("click", function () {
      const t = (promptEl && promptEl.value ? promptEl.value : "").trim();
      if (!t) {
        if (window.rabbitToast) rabbitToast("Composer empty", "ok");
        return;
      }
      const name = (t.split("\n")[0] || "Snippet").slice(0, 40);
      const list = loadSnips().filter(function (s) {
        return s.text !== t;
      });
      list.unshift({ name: name, text: t, t: Date.now() });
      saveSnips(list);
      renderSnips();
      if (window.rabbitToast) rabbitToast("Snippet saved", "ok");
    });
  }

  function renderSnips() {
    const host = document.getElementById("snip-list");
    if (!host) return;
    const list = loadSnips();
    if (!list.length) {
      host.innerHTML = '<p class="snip-empty">No snippets yet. Save from the composer.</p>';
      return;
    }
    host.innerHTML = list
      .map(function (s, i) {
        return (
          '<div class="snip-row">' +
          '<button type="button" class="snip-insert" data-i="' +
          i +
          '">' +
          escapeHtml(s.name || "Snippet") +
          "</button>" +
          '<button type="button" class="snip-del" data-x="' +
          i +
          '" title="Delete">×</button>' +
          "</div>"
        );
      })
      .join("");
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  document.body.addEventListener("click", function (ev) {
    const ins = ev.target.closest(".snip-insert");
    if (ins) {
      const s = loadSnips()[+ins.dataset.i];
      if (s && promptEl) {
        promptEl.value = s.text || "";
        promptEl.focus();
        promptEl.dispatchEvent(new Event("input", { bubbles: true }));
        document.getElementById("snippets-panel").hidden = true;
        if (window.rabbitToast) rabbitToast("Snippet inserted", "ok");
      }
      return;
    }
    const del = ev.target.closest(".snip-del");
    if (del) {
      const list = loadSnips();
      list.splice(+del.dataset.x, 1);
      saveSnips(list);
      renderSnips();
    }
  });

  function toggleSnips() {
    const p = document.getElementById("snippets-panel");
    if (!p) return;
    p.hidden = !p.hidden;
    if (!p.hidden) renderSnips();
  }

  const tools = document.querySelector(".composer-tools");
  if (tools && !document.getElementById("snip-open")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "snip-open";
    btn.className = "linkish";
    btn.textContent = "Snippets";
    btn.title = "Snippet library (Ctrl+Shift+S)";
    btn.addEventListener("click", toggleSnips);
    tools.appendChild(btn);
  }

  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.shiftKey && (ev.key === "s" || ev.key === "S")) {
      ev.preventDefault();
      toggleSnips();
    }
    // Esc clears composer when focused there (and not empty)
    if (ev.key === "Escape" && document.activeElement === promptEl) {
      if (promptEl.value) {
        ev.preventDefault();
        promptEl.value = "";
        promptEl.dispatchEvent(new Event("input", { bubbles: true }));
        if (window.rabbitToast) rabbitToast("Composer cleared", "ok");
      }
    }
  });

  // Latency probe — HEAD/GET /api/health or /health every 15s when online UI
  let probeEl = document.getElementById("latency-probe");
  if (!probeEl) {
    const top = document.querySelector(".top-right");
    probeEl = document.createElement("span");
    probeEl.id = "latency-probe";
    probeEl.className = "latency-probe";
    probeEl.title = "Round-trip to dashboard";
    probeEl.textContent = "— ms";
    if (top) top.insertBefore(probeEl, top.firstChild);
  }

  async function probe() {
    const paths = ["/api/health", "/health", "/"];
    const t0 = performance.now();
    let ok = false;
    for (let i = 0; i < paths.length; i++) {
      try {
        const ctrl = new AbortController();
        const timer = setTimeout(function () {
          ctrl.abort();
        }, 4000);
        const res = await fetch(paths[i], {
          method: "GET",
          cache: "no-store",
          signal: ctrl.signal,
        });
        clearTimeout(timer);
        if (res.ok || res.status < 500) {
          ok = true;
          break;
        }
      } catch (e) {}
    }
    const ms = Math.round(performance.now() - t0);
    probeEl.textContent = ok ? ms + " ms" : "offline";
    probeEl.classList.toggle("bad", !ok || ms > 800);
    probeEl.classList.toggle("warn", ok && ms > 300 && ms <= 800);
  }

  probe();
  setInterval(probe, 15000);

  window.rabbitSnippets = {
    toggle: toggleSnips,
    list: loadSnips,
    probe: probe,
  };
})();
