/* Voice keyboard shortcuts + right-side toggles */
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

  function syncVoiceButtons() {
    const mic = document.getElementById("mic");
    const on = !!(mic && mic.classList.contains("listening"));
    ["voice-toggle-top", "voice-toggle-fab"].forEach(function (id) {
      const el = document.getElementById(id);
      if (!el) return;
      el.classList.toggle("listening", on);
      el.setAttribute("aria-pressed", on ? "true" : "false");
    });
    const lab = document.getElementById("voice-toggle-label");
    if (lab) lab.textContent = on ? "Listening" : "Voice";
  }

  function bindVoiceSideButtons() {
    function fire() {
      const t = findToggle();
      if (t) t();
      else {
        const mic = document.getElementById("mic");
        if (mic) mic.click();
      }
      setTimeout(syncVoiceButtons, 50);
    }
    ["voice-toggle-top", "voice-toggle-fab"].forEach(function (id) {
      const el = document.getElementById(id);
      if (el && !el.dataset.bound) {
        el.dataset.bound = "1";
        el.addEventListener("click", fire);
      }
    });
    const mic = document.getElementById("mic");
    if (mic && !mic.dataset.voiceObs) {
      mic.dataset.voiceObs = "1";
      const obs = new MutationObserver(syncVoiceButtons);
      obs.observe(mic, { attributes: true, attributeFilter: ["class"] });
    }
    syncVoiceButtons();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindVoiceSideButtons);
  } else {
    bindVoiceSideButtons();
  }
  window.addEventListener("load", function () {
    setTimeout(bindVoiceSideButtons, 400);
  });
})();
