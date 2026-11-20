/* What's new panel + session Alt+[/] nav + Enter-to-send toggle */
(function () {
  const ENTER_KEY = "rabbit-enter-to-send";
  const SEEN_KEY = "rabbit-changelog-seen";
  const VERSION = "2026-11-20";

  const NOTES = [
    "Scratch notes drawer (Ctrl+Shift+N)",
    "Gen presets: Precise / Balanced / Creative / Long",
    "Drag session tabs to reorder",
    "Error status strip + Retry last prompt",
    "Focus mode, reactions, message deep links",
    "Templates, density presets, paste-as-code",
  ];

  // Changelog panel
  if (!document.getElementById("changelog-panel")) {
    const panel = document.createElement("div");
    panel.id = "changelog-panel";
    panel.className = "changelog-panel";
    panel.hidden = true;
    panel.innerHTML =
      '<header><strong>What\'s new</strong><button type="button" id="changelog-close" aria-label="Close">×</button></header>' +
      "<ul>" +
      NOTES.map(function (n) {
        return "<li>" + n + "</li>";
      }).join("") +
      "</ul>" +
      '<p class="changelog-ver">UI build ' +
      VERSION +
      "</p>";
    document.body.appendChild(panel);
    document.getElementById("changelog-close").addEventListener("click", function () {
      panel.hidden = true;
      try {
        localStorage.setItem(SEEN_KEY, VERSION);
      } catch (e) {}
    });
  }

  function openChangelog() {
    const p = document.getElementById("changelog-panel");
    if (p) p.hidden = false;
  }

  // Badge / button near help
  const help = document.getElementById("help-open");
  if (help && !document.getElementById("changelog-open")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "changelog-open";
    btn.className = "btn-changelog";
    btn.title = "What's new";
    btn.textContent = "New";
    help.parentNode.insertBefore(btn, help);
    btn.addEventListener("click", openChangelog);
    try {
      if (localStorage.getItem(SEEN_KEY) !== VERSION) {
        btn.classList.add("has-update");
      }
    } catch (e) {}
  }

  // Auto-show once per version
  try {
    if (localStorage.getItem(SEEN_KEY) !== VERSION) {
      setTimeout(openChangelog, 800);
    }
  } catch (e) {}

  // Session navigation Alt+[ / Alt+]
  function cycleSession(dir) {
    const tabs = Array.from(document.querySelectorAll(".session-tab")).filter(function (t) {
      return t.style.display !== "none";
    });
    if (!tabs.length) return;
    let i = tabs.findIndex(function (t) {
      return t.classList.contains("active");
    });
    if (i < 0) i = 0;
    i = (i + dir + tabs.length) % tabs.length;
    tabs[i].click();
  }

  document.addEventListener("keydown", function (ev) {
    if (!ev.altKey) return;
    if (ev.key === "[" || ev.key === "PageUp") {
      ev.preventDefault();
      cycleSession(-1);
    } else if (ev.key === "]" || ev.key === "PageDown") {
      ev.preventDefault();
      cycleSession(1);
    }
  });

  // Enter to send (Shift+Enter newline) preference
  const tools = document.querySelector(".composer-tools");
  const promptEl = document.getElementById("prompt");
  const formEl = document.getElementById("form");
  if (tools && !document.getElementById("enter-to-send")) {
    const label = document.createElement("label");
    label.className = "check inline";
    label.innerHTML = '<input type="checkbox" id="enter-to-send" /> Enter to send';
    tools.appendChild(label);
    const el = document.getElementById("enter-to-send");
    try {
      el.checked = localStorage.getItem(ENTER_KEY) === "1";
    } catch (e) {}
    el.addEventListener("change", function () {
      try {
        localStorage.setItem(ENTER_KEY, el.checked ? "1" : "0");
      } catch (e) {}
      if (window.rabbitToast)
        rabbitToast(el.checked ? "Enter sends · Shift+Enter newline" : "Enter inserts newline", "ok");
    });
  }

  if (promptEl && formEl) {
    promptEl.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter") return;
      if (ev.shiftKey) return; // always allow Shift+Enter newline
      let on = false;
      try {
        on = localStorage.getItem(ENTER_KEY) === "1";
      } catch (e) {}
      const box = document.getElementById("enter-to-send");
      if (box) on = box.checked;
      if (!on) return;
      ev.preventDefault();
      if (typeof formEl.requestSubmit === "function") formEl.requestSubmit();
      else formEl.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });
  }

  window.rabbitChangelog = { open: openChangelog, cycleSession: cycleSession };
})();
