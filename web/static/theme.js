/* Rabbit theme: dark / light / system */
(function () {
  const KEY = "rabbit-theme";
  const selectEl = document.getElementById("theme-select");
  const btn = document.getElementById("theme-toggle");
  const icon = document.getElementById("theme-toggle-icon");

  function systemTheme() {
    try {
      return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    } catch (e) {
      return "dark";
    }
  }

  function resolve(mode) {
    if (mode === "system") return systemTheme();
    return mode === "light" ? "light" : "dark";
  }

  function getMode() {
    try {
      return localStorage.getItem(KEY) || "dark";
    } catch (e) {
      return "dark";
    }
  }

  function setMode(mode) {
    const m = mode === "light" || mode === "system" ? mode : "dark";
    try {
      localStorage.setItem(KEY, m);
    } catch (e) {}
    const resolved = resolve(m);
    document.documentElement.setAttribute("data-theme", resolved);
    if (selectEl) selectEl.value = m;
    if (icon) icon.textContent = resolved === "light" ? "\u2600\uFE0F" : "\uD83C\uDF19";
    if (btn) btn.title = "Theme: " + m + " (Ctrl+Shift+D)";
  }

  function cycle() {
    const cur = getMode();
    const next = cur === "dark" ? "light" : cur === "light" ? "system" : "dark";
    setMode(next);
    if (window.rabbitToast) {
      rabbitToast("Theme: " + next, "ok");
    }
  }

  setMode(getMode());

  if (selectEl) {
    selectEl.addEventListener("change", function () {
      setMode(selectEl.value);
    });
  }
  if (btn) {
    btn.addEventListener("click", cycle);
  }

  document.addEventListener("keydown", function (ev) {
    if (ev.ctrlKey && ev.shiftKey && (ev.key === "D" || ev.key === "d")) {
      ev.preventDefault();
      cycle();
    }
  });

  try {
    window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", function () {
      if (getMode() === "system") setMode("system");
    });
  } catch (e) {}
})();
