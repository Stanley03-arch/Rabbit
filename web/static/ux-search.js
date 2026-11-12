/* Search highlight, density, i-focus — loaded after ux-extra.js */
(function () {
  const chatEl = document.getElementById("chat");
  if (!chatEl) return;
  const promptEl = document.getElementById("prompt");
  const searchEl = document.getElementById("chat-search");
  const densityEl = document.getElementById("compact-bubbles");
  const DENSITY_KEY = "rabbit-compact-bubbles";

  function loadPref(key, def) {
    try {
      const v = localStorage.getItem(key);
      if (v === null) return def;
      return v === "1";
    } catch (e) {
      return def;
    }
  }

  function applyDensity() {
    const on = loadPref(DENSITY_KEY, false);
    document.body.classList.toggle("compact-bubbles", on);
    if (densityEl) densityEl.checked = on;
  }
  applyDensity();
  if (densityEl) {
    densityEl.addEventListener("change", function () {
      const on = densityEl.checked;
      try {
        localStorage.setItem(DENSITY_KEY, on ? "1" : "0");
      } catch (e) {}
      applyDensity();
      if (window.rabbitToast) rabbitToast("Compact bubbles " + (on ? "on" : "off"), "ok");
    });
  }

  function clearSearchHits() {
    chatEl.querySelectorAll(".msg-hit").forEach(function (el) {
      const parent = el.parentNode;
      if (!parent) return;
      parent.replaceChild(document.createTextNode(el.textContent), el);
      parent.normalize();
    });
    chatEl.querySelectorAll(".bubble.search-match").forEach(function (b) {
      b.classList.remove("search-match");
    });
  }

  function applySearchHighlight(q) {
    clearSearchHits();
    const query = (q || "").trim();
    if (!query || query.length < 2) return;
    let first = null;
    chatEl.querySelectorAll(".bubble .body").forEach(function (body) {
      const raw = body.dataset.raw != null ? body.dataset.raw : body.textContent || "";
      if (!raw.toLowerCase().includes(query.toLowerCase())) return;
      const bubble = body.closest(".bubble");
      if (bubble) {
        bubble.classList.add("search-match");
        if (!first) first = bubble;
      }
      if (body.querySelector("a")) return;
      const qi = raw.toLowerCase().indexOf(query.toLowerCase());
      if (qi < 0) return;
      const before = raw.slice(0, qi);
      const match = raw.slice(qi, qi + query.length);
      const after = raw.slice(qi + query.length);
      body.innerHTML = "";
      if (before) body.appendChild(document.createTextNode(before));
      const mark = document.createElement("mark");
      mark.className = "msg-hit";
      mark.textContent = match;
      body.appendChild(mark);
      if (after) body.appendChild(document.createTextNode(after));
    });
    if (first) first.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  if (searchEl) {
    let st = null;
    searchEl.addEventListener("input", function () {
      clearTimeout(st);
      st = setTimeout(function () {
        applySearchHighlight(searchEl.value);
      }, 120);
    });
  }

  document.addEventListener("keydown", function (ev) {
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (ev.key === "i" || ev.key === "I") {
      if (promptEl) {
        ev.preventDefault();
        promptEl.focus();
      }
    }
  });
})();
