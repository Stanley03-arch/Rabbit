/* Summarize-selected into composer + session age labels + settings export/import */
(function () {
  const chatEl = document.getElementById("chat");
  const promptEl = document.getElementById("prompt");
  const tabsHost = document.getElementById("session-tabs");

  // Summarize selected bubbles (works with select-slash selection if present)
  function collectSelectedText() {
    const parts = [];
    const selected = chatEl ? chatEl.querySelectorAll(".bubble.msg-selected") : [];
    if (selected.length) {
      selected.forEach(function (b) {
        const role = b.classList.contains("user") ? "You" : "Rabbit";
        const body = b.querySelector(".body");
        const text =
          body && body.dataset && body.dataset.raw != null
            ? body.dataset.raw
            : body
              ? body.textContent
              : "";
        parts.push(role + ": " + (text || "").trim());
      });
      return parts.join("\n\n");
    }
    // fallback: last 6 bubbles
    if (!chatEl) return "";
    const all = Array.from(chatEl.querySelectorAll(".bubble")).slice(-6);
    all.forEach(function (b) {
      const role = b.classList.contains("user") ? "You" : "Rabbit";
      const body = b.querySelector(".body");
      const text =
        body && body.dataset && body.dataset.raw != null
          ? body.dataset.raw
          : body
            ? body.textContent
            : "";
      parts.push(role + ": " + (text || "").trim());
    });
    return parts.join("\n\n");
  }

  function summarizeIntoComposer() {
    if (!promptEl) return;
    const src = collectSelectedText().trim();
    if (!src) {
      if (window.rabbitToast) rabbitToast("Nothing to summarize", "ok");
      return;
    }
    const seed =
      "Summarize the following conversation briefly and list key points:\n\n" + src + "\n\nSummary:";
    promptEl.value = seed;
    promptEl.focus();
    promptEl.dispatchEvent(new Event("input", { bubbles: true }));
    if (window.rabbitToast) rabbitToast("Summary prompt ready — send when ready", "ok");
  }

  const tools = document.querySelector(".composer-tools");
  if (tools && !document.getElementById("summarize-sel")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "summarize-sel";
    btn.className = "linkish";
    btn.textContent = "Summarize";
    btn.title = "Build a summarize prompt from selected (or last) messages";
    btn.addEventListener("click", summarizeIntoComposer);
    tools.appendChild(btn);
  }

  // Wire into slash commands if select-slash loaded later
  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.shiftKey && (ev.key === "y" || ev.key === "Y")) {
      const tag = (ev.target && ev.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      ev.preventDefault();
      summarizeIntoComposer();
    }
  });

  // Session age labels — show relative time on tabs if data-updated or title has ISO
  function relative(ts) {
    const d = Date.now() - ts;
    if (d < 60000) return "now";
    if (d < 3600000) return Math.floor(d / 60000) + "m";
    if (d < 86400000) return Math.floor(d / 3600000) + "h";
    return Math.floor(d / 86400000) + "d";
  }

  function paintAges() {
    if (!tabsHost) return;
    tabsHost.querySelectorAll(".session-tab").forEach(function (tab) {
      let ts = parseInt(tab.dataset.updated || tab.dataset.ts || "0", 10);
      if (!ts) {
        // bump on click activity
        return;
      }
      let age = tab.querySelector(".tab-age");
      if (!age) {
        age = document.createElement("span");
        age.className = "tab-age";
        tab.appendChild(age);
      }
      age.textContent = relative(ts);
    });
  }

  if (tabsHost) {
    tabsHost.addEventListener("click", function (ev) {
      const tab = ev.target.closest(".session-tab");
      if (!tab) return;
      tab.dataset.updated = String(Date.now());
      paintAges();
    });
    new MutationObserver(paintAges).observe(tabsHost, { childList: true, subtree: true });
    setInterval(paintAges, 60000);
    paintAges();
  }

  // Settings export / import (rabbit-* localStorage subset of preferences)
  function exportSettings() {
    const out = {};
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (!k || k.indexOf("rabbit-") !== 0) continue;
        // skip bulky session payloads
        if (k.indexOf("session") >= 0 && k.indexOf("order") < 0 && k.indexOf("filter") < 0) continue;
        if (k.indexOf("history") >= 0 || k.indexOf("messages") >= 0 || k.indexOf("persist") >= 0) continue;
        out[k] = localStorage.getItem(k);
      }
    } catch (e) {}
    const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "rabbit-settings.json";
    a.click();
    setTimeout(function () {
      URL.revokeObjectURL(a.href);
    }, 1000);
    if (window.rabbitToast) rabbitToast("Settings exported", "ok");
  }

  function importSettings(file) {
    const reader = new FileReader();
    reader.onload = function () {
      try {
        const data = JSON.parse(reader.result);
        Object.keys(data).forEach(function (k) {
          if (k.indexOf("rabbit-") === 0) localStorage.setItem(k, data[k]);
        });
        if (window.rabbitToast) rabbitToast("Settings imported — reloading", "ok");
        setTimeout(function () {
          location.reload();
        }, 500);
      } catch (e) {
        if (window.rabbitToast) rabbitToast("Invalid settings file", "ok");
      }
    };
    reader.readAsText(file);
  }

  const controls = document.querySelector(".panel.controls");
  if (controls && !document.getElementById("export-settings")) {
    const row = document.createElement("div");
    row.className = "btn-row";
    row.innerHTML =
      '<button type="button" class="btn-secondary" id="export-settings">Export settings</button>' +
      '<button type="button" class="btn-secondary" id="import-settings">Import settings</button>' +
      '<input type="file" id="import-settings-file" accept="application/json,.json" hidden />';
    controls.appendChild(row);
    document.getElementById("export-settings").addEventListener("click", exportSettings);
    document.getElementById("import-settings").addEventListener("click", function () {
      document.getElementById("import-settings-file").click();
    });
    document.getElementById("import-settings-file").addEventListener("change", function (ev) {
      const f = ev.target.files && ev.target.files[0];
      if (f) importSettings(f);
      ev.target.value = "";
    });
  }

  // /summarize slash if composer
  if (promptEl) {
    promptEl.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter" || ev.shiftKey) return;
      const v = (promptEl.value || "").trim().toLowerCase();
      if (v !== "/summarize" && v !== "/sum") return;
      ev.preventDefault();
      promptEl.value = "";
      promptEl.dispatchEvent(new Event("input", { bubbles: true }));
      summarizeIntoComposer();
    });
  }

  window.rabbitSummarizeSettings = {
    summarizeIntoComposer: summarizeIntoComposer,
    exportSettings: exportSettings,
  };
})();
