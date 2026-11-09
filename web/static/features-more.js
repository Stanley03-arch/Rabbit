/* Pin messages, Ctrl+F search, reduced motion note */
(function () {
  const chatEl = document.getElementById("chat");
  const searchEl = document.getElementById("chat-search");
  const pinStrip = document.getElementById("pinned-strip");
  if (!chatEl) return;

  const PIN_KEY = "rabbit-pins-v1";

  function loadPins() {
    try {
      return JSON.parse(localStorage.getItem(PIN_KEY) || "[]");
    } catch (e) {
      return [];
    }
  }
  function savePins(list) {
    try {
      localStorage.setItem(PIN_KEY, JSON.stringify(list.slice(0, 12)));
    } catch (e) {}
  }

  function renderPins() {
    if (!pinStrip) return;
    const pins = loadPins();
    pinStrip.innerHTML = "";
    if (!pins.length) {
      pinStrip.hidden = true;
      return;
    }
    pinStrip.hidden = false;
    pins.forEach(function (p, idx) {
      const el = document.createElement("div");
      el.className = "pin-item";
      const text = document.createElement("span");
      text.className = "pin-text";
      text.textContent =
        (p.role === "user" ? "You: " : "Rabbit: ") + (p.content || "").slice(0, 120);
      const rm = document.createElement("button");
      rm.type = "button";
      rm.className = "pin-rm";
      rm.textContent = "\u00d7";
      rm.title = "Unpin";
      rm.addEventListener("click", function () {
        const list = loadPins();
        list.splice(idx, 1);
        savePins(list);
        renderPins();
      });
      el.appendChild(text);
      el.appendChild(rm);
      el.addEventListener("click", function (ev) {
        if (ev.target === rm) return;
        if (searchEl) {
          searchEl.value = (p.content || "").slice(0, 40);
          searchEl.dispatchEvent(new Event("input"));
        }
      });
      pinStrip.appendChild(el);
    });
  }

  function ensurePinButtons() {
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      if (b.querySelector(".pin-btn")) return;
      const actions = document.createElement("div");
      actions.className = "bubble-actions";
      const pin = document.createElement("button");
      pin.type = "button";
      pin.className = "bubble-act pin-btn";
      pin.textContent = "\uD83D\uDCCC";
      pin.title = "Pin message";
      pin.addEventListener("click", function (ev) {
        ev.stopPropagation();
        const body = b.querySelector(".body");
        const raw =
          body && body.dataset && body.dataset.raw != null
            ? body.dataset.raw
            : body
              ? body.textContent
              : "";
        const role = b.classList.contains("user") ? "user" : "bot";
        const list = loadPins();
        list.unshift({ role: role, content: (raw || "").trim(), ts: Date.now() });
        savePins(list);
        renderPins();
        if (window.rabbitToast) rabbitToast("Pinned", "ok");
      });
      actions.appendChild(pin);
      b.appendChild(actions);
    });
  }

  const obs = new MutationObserver(function () {
    ensurePinButtons();
  });
  obs.observe(chatEl, { childList: true });
  setTimeout(ensurePinButtons, 500);
  renderPins();

  document.addEventListener("keydown", function (ev) {
    if (ev.ctrlKey && !ev.altKey && !ev.metaKey && (ev.key === "f" || ev.key === "F")) {
      if (searchEl) {
        ev.preventDefault();
        searchEl.focus();
        searchEl.select();
      }
    }
    if (ev.key === "Escape" && searchEl && document.activeElement === searchEl) {
      searchEl.value = "";
      searchEl.dispatchEvent(new Event("input"));
      searchEl.blur();
    }
  });

  try {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      document.documentElement.classList.add("reduce-motion");
    }
  } catch (e) {}
})();
