/* Rabbit theme: dark / light / system + Dark mode checkbox */
(function () {
  const KEY = "rabbit-theme";
  const selectEl = document.getElementById("theme-select");
  const btn = document.getElementById("theme-toggle");
  const icon = document.getElementById("theme-toggle-icon");
  const textEl = document.getElementById("theme-toggle-text");
  const darkCheck = document.getElementById("dark-mode-toggle");

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
    if (textEl) textEl.textContent = resolved === "light" ? "Light" : "Dark";
    if (btn) {
      btn.title = "Theme: " + m + " (Ctrl+Shift+D) \u2014 click for dark/light";
      btn.setAttribute("aria-pressed", resolved === "dark" ? "true" : "false");
    }
    if (darkCheck) darkCheck.checked = resolved === "dark";
  }

  function toggleDarkLight() {
    const resolved = resolve(getMode());
    setMode(resolved === "dark" ? "light" : "dark");
    if (window.rabbitToast) rabbitToast(resolved === "dark" ? "Light mode" : "Dark mode", "ok");
  }

  setMode(getMode());

  if (selectEl) {
    selectEl.addEventListener("change", function () {
      setMode(selectEl.value);
      if (window.rabbitToast) rabbitToast("Theme: " + selectEl.value, "ok");
    });
  }

  if (darkCheck) {
    darkCheck.addEventListener("change", function () {
      setMode(darkCheck.checked ? "dark" : "light");
      if (window.rabbitToast) rabbitToast(darkCheck.checked ? "Dark mode" : "Light mode", "ok");
    });
  }

  if (btn) {
    btn.addEventListener("click", toggleDarkLight);
  }

  document.addEventListener("keydown", function (ev) {
    if (ev.ctrlKey && ev.shiftKey && (ev.key === "D" || ev.key === "d")) {
      ev.preventDefault();
      toggleDarkLight();
    }
  });

  try {
    window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", function () {
      if (getMode() === "system") setMode("system");
    });
  } catch (e) {}
})();
