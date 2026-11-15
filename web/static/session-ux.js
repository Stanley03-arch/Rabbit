/* Session rename (dblclick) · pin session · quiet hours for completion sound */
(function () {
  const PIN_KEY = "rabbit-pinned-sessions";
  const QUIET_KEY = "rabbit-quiet-hours";
  const tabsHost = document.getElementById("session-tabs");

  function loadPins() {
    try {
      const raw = localStorage.getItem(PIN_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function savePins(arr) {
    try {
      localStorage.setItem(PIN_KEY, JSON.stringify(arr));
    } catch (e) {}
  }

  function sessionIdFromTab(tab) {
    return tab.dataset.sessionId || tab.dataset.id || tab.getAttribute("data-id") || tab.textContent.trim();
  }

  function decoratePins() {
    if (!tabsHost) return;
    const pins = loadPins();
    tabsHost.querySelectorAll(".session-tab").forEach(function (tab) {
      const id = sessionIdFromTab(tab);
      const pinned = pins.indexOf(id) >= 0;
      tab.classList.toggle("session-pinned", pinned);
      if (pinned && tab.textContent.indexOf("📌") < 0) {
        // lightweight marker without breaking title handlers
        if (!tab.querySelector(".pin-mark")) {
          const s = document.createElement("span");
          s.className = "pin-mark";
          s.textContent = "📌";
          s.style.marginRight = "0.25rem";
          tab.insertBefore(s, tab.firstChild);
        }
      }
      if (!pinned) {
        const m = tab.querySelector(".pin-mark");
        if (m) m.remove();
      }
    });
  }

  function togglePinActive() {
    const active = tabsHost && tabsHost.querySelector(".session-tab.active");
    if (!active) {
      if (window.rabbitToast) rabbitToast("No active session", "ok");
      return;
    }
    const id = sessionIdFromTab(active);
    let pins = loadPins();
    const i = pins.indexOf(id);
    if (i >= 0) {
      pins.splice(i, 1);
      if (window.rabbitToast) rabbitToast("Session unpinned", "ok");
    } else {
      pins.unshift(id);
      if (window.rabbitToast) rabbitToast("Session pinned", "ok");
    }
    savePins(pins);
    decoratePins();
  }

  // Double-click tab title to rename
  if (tabsHost) {
    tabsHost.addEventListener("dblclick", function (ev) {
      const tab = ev.target.closest(".session-tab");
      if (!tab) return;
      ev.preventDefault();
      const current = (tab.dataset.title || tab.textContent || "").replace(/^📌\s*/, "").trim();
      const next = window.prompt("Rename session", current);
      if (next == null) return;
      const name = next.trim();
      if (!name) return;
      tab.dataset.title = name;
      // try common APIs
      if (window.rabbitSessions && typeof rabbitSessions.rename === "function") {
        rabbitSessions.rename(sessionIdFromTab(tab), name);
      } else if (window.rabbitSessionsExtra && typeof rabbitSessionsExtra.rename === "function") {
        rabbitSessionsExtra.rename(sessionIdFromTab(tab), name);
      } else {
        // DOM-only fallback: replace text nodes
        const pin = tab.querySelector(".pin-mark");
        tab.textContent = "";
        if (pin) tab.appendChild(pin);
        tab.appendChild(document.createTextNode(name));
      }
      decoratePins();
      if (window.rabbitToast) rabbitToast("Renamed", "ok");
      if (window.rabbitPersist) rabbitPersist.save();
    });

    new MutationObserver(function () {
      decoratePins();
    }).observe(tabsHost, { childList: true, subtree: true });
    decoratePins();
  }

  // Quiet hours: mute completion sound between local hours
  function ensureQuietUI() {
    const panel = document.querySelector(".panel.controls");
    if (!panel || document.getElementById("quiet-hours")) return;
    const label = document.createElement("label");
    label.className = "check";
    label.innerHTML =
      '<input type="checkbox" id="quiet-hours" /> Quiet hours (mute reply sound 22:00–07:00)';
    const sound = document.getElementById("completion-sound");
    if (sound && sound.closest("label")) {
      sound.closest("label").insertAdjacentElement("afterend", label);
    } else {
      panel.appendChild(label);
    }
    const el = document.getElementById("quiet-hours");
    try {
      el.checked = localStorage.getItem(QUIET_KEY) === "1";
    } catch (e) {}
    el.addEventListener("change", function () {
      try {
        localStorage.setItem(QUIET_KEY, el.checked ? "1" : "0");
      } catch (e) {}
    });
  }
  ensureQuietUI();

  function inQuietHours() {
    try {
      if (localStorage.getItem(QUIET_KEY) !== "1") return false;
    } catch (e) {
      return false;
    }
    const h = new Date().getHours();
    return h >= 22 || h < 7;
  }

  // Soft-block completion sound during quiet hours by temporarily unchecking
  const soundEl = document.getElementById("completion-sound");
  if (soundEl) {
    const orig = soundEl.checked;
    setInterval(function () {
      if (!soundEl.dataset.userWantSound) {
        soundEl.dataset.userWantSound = soundEl.checked ? "1" : "0";
      }
    }, 2000);
    soundEl.addEventListener("change", function () {
      soundEl.dataset.userWantSound = soundEl.checked ? "1" : "0";
    });
    // When app might play sound, force off if quiet
    const guard = setInterval(function () {
      if (inQuietHours() && soundEl.checked) {
        soundEl.checked = false;
      } else if (!inQuietHours() && soundEl.dataset.userWantSound === "1" && !soundEl.checked) {
        soundEl.checked = true;
      }
    }, 5000);
  }

  // Palette / keyboard: Ctrl+Shift+R already used by browser sometimes — use Ctrl+Alt+P for pin
  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.altKey && (ev.key === "p" || ev.key === "P")) {
      ev.preventDefault();
      togglePinActive();
    }
  });

  window.rabbitSessionUx = {
    togglePinActive: togglePinActive,
    decoratePins: decoratePins,
    inQuietHours: inQuietHours,
  };
})();
