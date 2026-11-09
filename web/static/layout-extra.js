/* Sidebar collapse + shortcuts help */
(function () {
  const KEY = "rabbit-sidebar-collapsed";
  const toggle = document.getElementById("sidebar-toggle");
  const expand = document.getElementById("sidebar-expand");
  const helpModal = document.getElementById("help-modal");
  const helpOpen = document.getElementById("help-open");
  const helpClose = document.getElementById("help-close");

  function setCollapsed(on) {
    document.body.classList.toggle("sidebar-collapsed", !!on);
    if (expand) expand.hidden = !on;
    try {
      localStorage.setItem(KEY, on ? "1" : "0");
    } catch (e) {}
  }

  function isCollapsed() {
    return document.body.classList.contains("sidebar-collapsed");
  }

  try {
    if (localStorage.getItem(KEY) === "1") setCollapsed(true);
  } catch (e) {}

  if (toggle) toggle.addEventListener("click", function () {
    setCollapsed(!isCollapsed());
  });
  if (expand) expand.addEventListener("click", function () {
    setCollapsed(false);
  });

  function openHelp() {
    if (helpModal) helpModal.hidden = false;
  }
  function closeHelp() {
    if (helpModal) helpModal.hidden = true;
  }
  if (helpOpen) helpOpen.addEventListener("click", openHelp);
  if (helpClose) helpClose.addEventListener("click", closeHelp);
  if (helpModal) {
    helpModal.addEventListener("click", function (ev) {
      if (ev.target === helpModal) closeHelp();
    });
  }

  document.addEventListener("keydown", function (ev) {
    if (ev.key === "?" && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
      const tag = (ev.target && ev.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      ev.preventDefault();
      if (helpModal && !helpModal.hidden) closeHelp();
      else openHelp();
    }
    if (ev.key === "Escape" && helpModal && !helpModal.hidden) {
      closeHelp();
    }
    if (ev.ctrlKey && !ev.altKey && !ev.metaKey && ev.key === "\\") {
      ev.preventDefault();
      setCollapsed(!isCollapsed());
    }
  });
})();
