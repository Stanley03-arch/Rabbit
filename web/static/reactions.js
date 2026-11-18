/* Lightweight emoji reactions on bubbles (local only) */
(function () {
  const KEY = "rabbit-reactions-v1";
  const chatEl = document.getElementById("chat");
  if (!chatEl) return;

  const EMOJIS = ["👍", "👎", "🔥", "💡", "❓"];

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

  function bubbleKey(b) {
    const body = b.querySelector(".body");
    const raw =
      body && body.dataset && body.dataset.raw != null
        ? body.dataset.raw
        : body
          ? body.textContent
          : "";
    const role = b.classList.contains("user") ? "u" : "b";
    return role + ":" + String(raw || "").slice(0, 120);
  }

  function renderBar(b) {
    if (b.querySelector(".react-bar")) return;
    const bar = document.createElement("div");
    bar.className = "react-bar";
    const map = load();
    const key = bubbleKey(b);
    const cur = map[key] || "";
    EMOJIS.forEach(function (em) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "react-btn" + (cur === em ? " active" : "");
      btn.textContent = em;
      btn.title = "React " + em;
      btn.addEventListener("click", function (ev) {
        ev.stopPropagation();
        const m = load();
        if (m[key] === em) delete m[key];
        else m[key] = em;
        save(m);
        // refresh all matching
        chatEl.querySelectorAll(".bubble").forEach(function (x) {
          const bb = x.querySelector(".react-bar");
          if (bb) bb.remove();
          x.dataset.reacted = "";
        });
        scan();
      });
      bar.appendChild(btn);
    });
    if (cur) {
      const mark = document.createElement("span");
      mark.className = "react-current";
      mark.textContent = cur;
      bar.appendChild(mark);
    }
    b.appendChild(bar);
  }

  function scan() {
    chatEl.querySelectorAll(".bubble").forEach(function (b) {
      if (b.classList.contains("streaming")) return;
      renderBar(b);
    });
  }

  new MutationObserver(function () {
    scan();
  }).observe(chatEl, { childList: true, subtree: true });
  scan();
})();
