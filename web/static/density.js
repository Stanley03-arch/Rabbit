/* UI density presets: comfortable / compact / dense */
(function () {
  const KEY = "rabbit-density";
  const root = document.documentElement;

  function apply(mode) {
    root.classList.remove("density-comfortable", "density-compact", "density-dense");
    const m = mode === "compact" || mode === "dense" ? mode : "comfortable";
    root.classList.add("density-" + m);
    try {
      localStorage.setItem(KEY, m);
    } catch (e) {}
    const sel = document.getElementById("density-select");
    if (sel && sel.value !== m) sel.value = m;
  }

  function current() {
    try {
      return localStorage.getItem(KEY) || "comfortable";
    } catch (e) {
      return "comfortable";
    }
  }

  function ensureUI() {
    const panel = document.querySelector(".panel.controls");
    if (!panel || document.getElementById("density-select")) return;
    const label = document.createElement("label");
    label.innerHTML =
      'Density <select id="density-select">' +
      '<option value="comfortable">Comfortable</option>' +
      '<option value="compact">Compact</option>' +
      '<option value="dense">Dense</option>' +
      "</select>";
    const scale = document.getElementById("font-scale");
    if (scale && scale.closest("label")) {
      scale.closest("label").insertAdjacentElement("afterend", label);
    } else {
      panel.appendChild(label);
    }
    const sel = document.getElementById("density-select");
    sel.value = current();
    sel.addEventListener("change", function () {
      apply(sel.value);
      if (window.rabbitToast) rabbitToast("Density: " + sel.value, "ok");
    });
  }

  ensureUI();
  apply(current());

  window.rabbitDensity = { apply: apply, current: current };
})();
