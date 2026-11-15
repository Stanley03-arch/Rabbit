/* Collapsible sidebar panels (remembered) */
(function () {
  const KEY = "rabbit-sidebar-collapsed";
  const sidebar = document.querySelector(".sidebar");
  if (!sidebar) return;

  function load() {
    try {
      const raw = localStorage.getItem(KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (e) {
      return {};
    }
  }

  function save(map) {
    try {
      localStorage.setItem(KEY, JSON.stringify(map));
    } catch (e) {}
  }

  const state = load();

  sidebar.querySelectorAll("section.panel").forEach(function (panel, idx) {
    const h = panel.querySelector("h3");
    if (!h) return;
    const id = h.textContent.trim().toLowerCase().replace(/\s+/g, "-") || "p" + idx;
    panel.dataset.panelId = id;
    h.classList.add("panel-toggle");
    h.title = "Click to collapse/expand";
    h.setAttribute("role", "button");
    h.tabIndex = 0;

    if (state[id]) panel.classList.add("panel-collapsed");

    function toggle() {
      panel.classList.toggle("panel-collapsed");
      state[id] = panel.classList.contains("panel-collapsed");
      save(state);
    }
    h.addEventListener("click", toggle);
    h.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        toggle();
      }
    });
  });
})();
