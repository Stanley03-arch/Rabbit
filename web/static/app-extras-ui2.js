/* Rabbit extras-ui2: system presets, font scale, drag-drop import, Ctrl+K */
(function () {
  const SETTINGS_KEY = "rabbit-dashboard-v1";
  const systemEl = document.getElementById("system");
  const presetEl = document.getElementById("system-preset");
  const fontScaleEl = document.getElementById("font-scale");
  const chatEl = document.getElementById("chat");
  const promptEl = document.getElementById("prompt");
  const importFile = document.getElementById("import-file");

  function loadBits() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (!raw) return;
      const s = JSON.parse(raw);
      if (s.font_scale && fontScaleEl) {
        fontScaleEl.value = String(s.font_scale);
        applyFontScale(s.font_scale);
      }
    } catch (e) {}
  }

  function patchSettings(partial) {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      const s = raw ? JSON.parse(raw) : {};
      Object.assign(s, partial);
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    } catch (e) {}
  }

  function applyFontScale(v) {
    const n = Number(v) || 1;
    document.documentElement.style.setProperty("--ui-scale", String(n));
    document.body.style.fontSize = 16 * n + "px";
  }

  if (presetEl && systemEl) {
    presetEl.addEventListener("change", function () {
      if (!presetEl.value) return;
      systemEl.value = presetEl.value;
      systemEl.dispatchEvent(new Event("input", { bubbles: true }));
      if (typeof rabbitSaveSettings === "function") rabbitSaveSettings();
    });
  }

  if (fontScaleEl) {
    fontScaleEl.addEventListener("change", function () {
      applyFontScale(fontScaleEl.value);
      patchSettings({ font_scale: Number(fontScaleEl.value) });
    });
  }

  document.addEventListener("keydown", function (ev) {
    if (ev.ctrlKey && !ev.altKey && !ev.metaKey && (ev.key === "k" || ev.key === "K")) {
      if (promptEl) {
        ev.preventDefault();
        promptEl.focus();
        if (promptEl.select) promptEl.select();
      }
    }
  });

  function importText(text, name) {
    const trimmed = (text || "").trim();
    if (!trimmed) return;
    try {
      const data = JSON.parse(trimmed);
      const msgs = Array.isArray(data) ? data : data.messages || data.history || data.chat;
      if (Array.isArray(msgs) && msgs.length && chatEl) {
        const empty = document.getElementById("empty-state");
        if (empty) empty.style.display = "none";
        msgs.forEach(function (m) {
          const role = m.role === "user" || m.role === "human" ? "user" : "bot";
          const content = m.content || m.text || "";
          const div = document.createElement("div");
          div.className = "bubble " + role;
          const meta = document.createElement("div");
          meta.className = "meta";
          meta.textContent = (role === "user" ? "you" : "rabbit") + " \u00b7 import";
          const body = document.createElement("div");
          body.className = "body";
          body.textContent = content;
          div.appendChild(meta);
          div.appendChild(body);
          chatEl.appendChild(div);
        });
        chatEl.scrollTop = chatEl.scrollHeight;
        return;
      }
    } catch (e) {}
    if (/^##\s+/m.test(trimmed) && chatEl) {
      const empty = document.getElementById("empty-state");
      if (empty) empty.style.display = "none";
      const parts = trimmed.split(/^##\s+/m).slice(1);
      parts.forEach(function (block) {
        const nl = block.indexOf("\n");
        const title = (nl >= 0 ? block.slice(0, nl) : block).trim().toLowerCase();
        const bodyText = (nl >= 0 ? block.slice(nl + 1) : "").trim();
        if (!bodyText) return;
        const role =
          title.indexOf("you") >= 0 || title.indexOf("user") >= 0 ? "user" : "bot";
        const div = document.createElement("div");
        div.className = "bubble " + role;
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = (role === "user" ? "you" : "rabbit") + " \u00b7 import";
        const body = document.createElement("div");
        body.className = "body";
        body.textContent = bodyText;
        div.appendChild(meta);
        div.appendChild(body);
        chatEl.appendChild(div);
      });
      chatEl.scrollTop = chatEl.scrollHeight;
      return;
    }
    alert("Could not parse " + (name || "file") + " as chat JSON or Markdown.");
  }

  if (chatEl) {
    chatEl.addEventListener("dragover", function (ev) {
      ev.preventDefault();
      chatEl.classList.add("drop-target");
    });
    chatEl.addEventListener("dragleave", function () {
      chatEl.classList.remove("drop-target");
    });
    chatEl.addEventListener("drop", function (ev) {
      ev.preventDefault();
      chatEl.classList.remove("drop-target");
      const f = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = function () {
        importText(String(reader.result || ""), f.name);
      };
      reader.readAsText(f);
    });
  }

  if (importFile) {
    importFile.setAttribute(
      "accept",
      "application/json,.json,.md,text/markdown,text/plain"
    );
  }

  loadBits();
})();
