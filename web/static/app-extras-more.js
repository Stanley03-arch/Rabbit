/* Rabbit extras-more: MD export, sound, compact, Escape-stop */
(function () {
  const completionSoundEl = document.getElementById("completion-sound");
  const compactUiEl = document.getElementById("compact-ui");
  const exportMdBtn = document.getElementById("export-md");
  const stopBtn = document.getElementById("stop");
  const systemEl = document.getElementById("system");
  const chatEl = document.getElementById("chat");
  const SETTINGS_KEY = "rabbit-dashboard-v1";

  function applyCompact(on) {
    document.body.classList.toggle("compact", !!on);
  }

  function playCompletionSound() {
    if (!completionSoundEl || !completionSoundEl.checked) return;
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      const ctx = new Ctx();
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.type = "sine";
      o.frequency.value = 880;
      g.gain.value = 0.04;
      o.connect(g);
      g.connect(ctx.destination);
      o.start();
      g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.18);
      o.stop(ctx.currentTime + 0.2);
      setTimeout(function () {
        try { ctx.close(); } catch (e) {}
      }, 300);
    } catch (e) {}
  }

  function readBubbles() {
    if (!chatEl) return [];
    return Array.from(chatEl.querySelectorAll(".bubble")).map(function (div) {
      const role = div.classList.contains("user") ? "user" : "bot";
      const body = div.querySelector(".body");
      return { role: role, content: body ? body.textContent : "" };
    });
  }

  function exportMarkdown() {
    const lines = ["# Rabbit chat", "", "_Exported " + new Date().toISOString() + "_", ""];
    const sys = systemEl ? systemEl.value.trim() : "";
    if (sys) lines.push("## System", "", sys, "");
    readBubbles().forEach(function (m) {
      lines.push("## " + (m.role === "user" ? "You" : "Rabbit"), "", m.content || "", "");
    });
    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "rabbit-chat-" + Date.now() + ".md";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) {
      const s = JSON.parse(raw);
      if (typeof s.completion_sound === "boolean" && completionSoundEl)
        completionSoundEl.checked = s.completion_sound;
      if (typeof s.compact_ui === "boolean" && compactUiEl) {
        compactUiEl.checked = s.compact_ui;
        applyCompact(s.compact_ui);
      }
    }
  } catch (e) {}

  function patchSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      const s = raw ? JSON.parse(raw) : {};
      s.completion_sound = !!(completionSoundEl && completionSoundEl.checked);
      s.compact_ui = !!(compactUiEl && compactUiEl.checked);
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    } catch (e) {}
  }

  if (exportMdBtn) exportMdBtn.addEventListener("click", exportMarkdown);
  if (compactUiEl) {
    compactUiEl.addEventListener("change", function () {
      applyCompact(compactUiEl.checked);
      patchSettings();
    });
  }
  if (completionSoundEl) {
    completionSoundEl.addEventListener("change", patchSettings);
  }

  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape" && stopBtn && !stopBtn.hidden) stopBtn.click();
  });

  const statusPill = document.getElementById("status-pill");
  if (statusPill) {
    let last = statusPill.textContent || "";
    const so = new MutationObserver(function () {
      const now = statusPill.textContent || "";
      if (/generat|busy/i.test(last) && /ready|ok/i.test(now)) playCompletionSound();
      last = now;
    });
    so.observe(statusPill, { characterData: true, childList: true, subtree: true });
  }
})();
