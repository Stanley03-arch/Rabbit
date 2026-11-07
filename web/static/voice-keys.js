/* Voice keyboard shortcuts — Ctrl+M always, M/V when not typing */
(function () {
  function findToggle() {
    if (typeof toggleMic === "function") return toggleMic;
    const btn = document.getElementById("mic");
    if (btn) return function () { btn.click(); };
    return null;
  }
  document.addEventListener(
    "keydown",
    function (ev) {
      const toggle = findToggle();
      if (!toggle) return;
      if ((ev.key === "m" || ev.key === "M") && (ev.ctrlKey || ev.metaKey) && !ev.altKey) {
        ev.preventDefault();
        ev.stopPropagation();
        toggle();
        return;
      }
      const tag = (ev.target && ev.target.tagName) || "";
      const typing =
        tag === "INPUT" || tag === "TEXTAREA" || (ev.target && ev.target.isContentEditable);
      if (
        !typing &&
        !ev.ctrlKey &&
        !ev.metaKey &&
        !ev.altKey &&
        (ev.key === "m" || ev.key === "M" || ev.key === "v" || ev.key === "V")
      ) {
        ev.preventDefault();
        toggle();
      }
    },
    true
  );
})();
