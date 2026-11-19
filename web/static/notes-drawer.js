/* Scratch notes drawer + gen presets + session tab drag reorder */
(function () {
  const NOTES_KEY = "rabbit-scratch-notes";
  const ORDER_KEY = "rabbit-session-order";

  // --- Notes drawer ---
  if (!document.getElementById("notes-drawer")) {
    const drawer = document.createElement("aside");
    drawer.id = "notes-drawer";
    drawer.className = "notes-drawer";
    drawer.hidden = true;
    drawer.innerHTML =
      '<header class="notes-head">' +
      "<strong>Scratch notes</strong>" +
      '<button type="button" id="notes-close" aria-label="Close">×</button>' +
      "</header>" +
      '<textarea id="notes-body" placeholder="Local notes — not sent to the model…"></textarea>' +
      '<div class="notes-foot"><button type="button" id="notes-to-prompt" class="btn-secondary">Insert into prompt</button></div>';
    document.body.appendChild(drawer);

    const body = document.getElementById("notes-body");
    try {
      body.value = localStorage.getItem(NOTES_KEY) || "";
    } catch (e) {}
    body.addEventListener("input", function () {
      try {
        localStorage.setItem(NOTES_KEY, body.value);
      } catch (e) {}
    });

    document.getElementById("notes-close").addEventListener("click", function () {
      drawer.hidden = true;
    });
    document.getElementById("notes-to-prompt").addEventListener("click", function () {
      const promptEl = document.getElementById("prompt");
      if (!promptEl) return;
      const t = body.value.trim();
      if (!t) return;
      promptEl.value = (promptEl.value ? promptEl.value + "\n\n" : "") + t;
      promptEl.dispatchEvent(new Event("input", { bubbles: true }));
      promptEl.focus();
      if (window.rabbitToast) rabbitToast("Notes inserted", "ok");
    });
  }

  function toggleNotes() {
    const d = document.getElementById("notes-drawer");
    if (!d) return;
    d.hidden = !d.hidden;
    if (!d.hidden) document.getElementById("notes-body").focus();
  }

  // Notes button in top-right
  const top = document.querySelector(".top-right");
  if (top && !document.getElementById("notes-open")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "notes-open";
    btn.className = "btn-notes";
    btn.title = "Scratch notes (Ctrl+Shift+N)";
    btn.textContent = "Notes";
    top.insertBefore(btn, top.firstChild);
    btn.addEventListener("click", toggleNotes);
  }

  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.shiftKey && (ev.key === "n" || ev.key === "N")) {
      ev.preventDefault();
      toggleNotes();
    }
  });

  // --- Generation presets ---
  const controls = document.querySelector(".panel.controls");
  if (controls && !document.getElementById("gen-presets")) {
    const row = document.createElement("div");
    row.id = "gen-presets";
    row.className = "gen-presets";
    row.innerHTML =
      '<span class="gen-presets-label">Presets</span>' +
      '<button type="button" data-tok="32" data-temp="0.3">Precise</button>' +
      '<button type="button" data-tok="48" data-temp="0.8">Balanced</button>' +
      '<button type="button" data-tok="96" data-temp="1.1">Creative</button>' +
      '<button type="button" data-tok="128" data-temp="0.7">Long</button>';
    const tokens = document.getElementById("tokens");
    if (tokens && tokens.closest("label")) {
      tokens.closest("label").insertAdjacentElement("beforebegin", row);
    } else {
      controls.insertBefore(row, controls.children[1] || null);
    }
    row.addEventListener("click", function (ev) {
      const b = ev.target.closest("button[data-tok]");
      if (!b) return;
      const tok = document.getElementById("tokens");
      const temp = document.getElementById("temp");
      if (tok) {
        tok.value = b.dataset.tok;
        tok.dispatchEvent(new Event("input", { bubbles: true }));
        tok.dispatchEvent(new Event("change", { bubbles: true }));
      }
      if (temp) {
        temp.value = b.dataset.temp;
        temp.dispatchEvent(new Event("input", { bubbles: true }));
        temp.dispatchEvent(new Event("change", { bubbles: true }));
      }
      if (window.rabbitToast)
        rabbitToast("Preset: " + b.textContent + " (" + b.dataset.tok + " tok, t=" + b.dataset.temp + ")", "ok");
    });
  }

  // --- Session tab drag reorder ---
  const tabsHost = document.getElementById("session-tabs");
  if (tabsHost) {
    let dragEl = null;

    function persistOrder() {
      const ids = [];
      tabsHost.querySelectorAll(".session-tab").forEach(function (t) {
        const id = t.dataset.sessionId || t.dataset.id || t.getAttribute("data-id") || t.textContent.trim();
        ids.push(id);
      });
      try {
        localStorage.setItem(ORDER_KEY, JSON.stringify(ids));
      } catch (e) {}
    }

    function applySavedOrder() {
      let order;
      try {
        order = JSON.parse(localStorage.getItem(ORDER_KEY) || "[]");
      } catch (e) {
        order = [];
      }
      if (!Array.isArray(order) || !order.length) return;
      const map = {};
      tabsHost.querySelectorAll(".session-tab").forEach(function (t) {
        const id = t.dataset.sessionId || t.dataset.id || t.getAttribute("data-id") || t.textContent.trim();
        map[id] = t;
      });
      order.forEach(function (id) {
        if (map[id]) tabsHost.appendChild(map[id]);
      });
    }

    tabsHost.addEventListener("dragstart", function (ev) {
      const tab = ev.target.closest(".session-tab");
      if (!tab) return;
      dragEl = tab;
      tab.classList.add("dragging");
      try {
        ev.dataTransfer.setData("text/plain", "session");
        ev.dataTransfer.effectAllowed = "move";
      } catch (e) {}
    });
    tabsHost.addEventListener("dragend", function () {
      if (dragEl) dragEl.classList.remove("dragging");
      dragEl = null;
      tabsHost.querySelectorAll(".session-tab").forEach(function (t) {
        t.classList.remove("drag-over");
      });
      persistOrder();
    });
    tabsHost.addEventListener("dragover", function (ev) {
      ev.preventDefault();
      const tab = ev.target.closest(".session-tab");
      if (!tab || tab === dragEl) return;
      tabsHost.querySelectorAll(".session-tab").forEach(function (t) {
        t.classList.toggle("drag-over", t === tab);
      });
      const rect = tab.getBoundingClientRect();
      const before = ev.clientX < rect.left + rect.width / 2;
      if (before) tabsHost.insertBefore(dragEl, tab);
      else tabsHost.insertBefore(dragEl, tab.nextSibling);
    });

    // make tabs draggable when created
    function markDraggable() {
      tabsHost.querySelectorAll(".session-tab").forEach(function (t) {
        if (!t.getAttribute("draggable")) t.setAttribute("draggable", "true");
      });
    }
    new MutationObserver(function () {
      markDraggable();
      applySavedOrder();
    }).observe(tabsHost, { childList: true });
    markDraggable();
    applySavedOrder();
  }

  window.rabbitNotes = { toggle: toggleNotes };
})();
