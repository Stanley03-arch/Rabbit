/* Focus mode: dim sidebar; reading progress; message deep-link copy */
(function () {
  const KEY = "rabbit-focus-mode";
  const chatEl = document.getElementById("chat");
  const main = document.querySelector(".main");

  // Focus mode toggle
  function applyFocus(on) {
    document.documentElement.classList.toggle("focus-mode", !!on);
    try {
      localStorage.setItem(KEY, on ? "1" : "0");
    } catch (e) {}
    const btn = document.getElementById("focus-mode-btn");
    if (btn) {
      btn.setAttribute("aria-pressed", on ? "true" : "false");
      btn.textContent = on ? "Exit focus" : "Focus";
    }
  }

  function ensureFocusBtn() {
    const top = document.querySelector(".top-right");
    if (!top || document.getElementById("focus-mode-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "focus-mode-btn";
    btn.className = "btn-focus";
    btn.title = "Focus mode (Ctrl+Shift+F is favorite — use Ctrl+. )";
    btn.textContent = "Focus";
    top.insertBefore(btn, top.firstChild);
    btn.addEventListener("click", function () {
      applyFocus(!document.documentElement.classList.contains("focus-mode"));
    });
  }
  ensureFocusBtn();
  try {
    applyFocus(localStorage.getItem(KEY) === "1");
  } catch (e) {}

  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.key === ".") {
      ev.preventDefault();
      applyFocus(!document.documentElement.classList.contains("focus-mode"));
    }
  });

  // Reading progress bar
  let bar = document.getElementById("read-progress");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "read-progress";
    bar.className = "read-progress";
    bar.innerHTML = '<div class="read-progress-fill" id="read-progress-fill"></div>';
    document.body.appendChild(bar);
  }
  const fill = document.getElementById("read-progress-fill");

  function updateProgress() {
    if (!chatEl) return;
    const max = chatEl.scrollHeight - chatEl.clientHeight;
    const p = max <= 0 ? 0 : Math.min(1, chatEl.scrollTop / max);
    fill.style.width = (p * 100).toFixed(1) + "%";
  }
  if (chatEl) {
    chatEl.addEventListener("scroll", updateProgress, { passive: true });
    new MutationObserver(updateProgress).observe(chatEl, { childList: true, subtree: true });
    updateProgress();
  }

  // Deep link: assign data-mid and copy link button on bubbles
  function hashId(s) {
    let h = 0;
    const str = String(s || "");
    for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) | 0;
    return "m" + Math.abs(h).toString(36);
  }

  function enhance(b) {
    if (!b || b.dataset.deepLink === "1") return;
    b.dataset.deepLink = "1";
    const body = b.querySelector(".body");
    const raw =
      body && body.dataset && body.dataset.raw != null
        ? body.dataset.raw
        : body
          ? body.textContent
          : "";
    const id = hashId((b.classList.contains("user") ? "u" : "b") + raw);
    b.id = b.id || id;
    b.dataset.mid = id;

    const actions = b.querySelector(".bubble-actions");
    const host = actions || b;
    if (host.querySelector(".deep-link-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "bubble-act deep-link-btn";
    btn.textContent = "Link";
    btn.title = "Copy link to this message";
    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      const url = location.origin + location.pathname + location.search + "#" + id;
      const done = function () {
        if (window.rabbitToast) rabbitToast("Message link copied", "ok");
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(url).then(done);
      } else {
        const ta = document.createElement("textarea");
        ta.value = url;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
        done();
      }
    });
    host.appendChild(btn);
  }

  function scan() {
    if (!chatEl) return;
    chatEl.querySelectorAll(".bubble").forEach(enhance);
  }

  if (chatEl) {
    new MutationObserver(scan).observe(chatEl, { childList: true, subtree: true });
    scan();
  }

  // Jump on hash
  function jumpHash() {
    const id = (location.hash || "").replace(/^#/, "");
    if (!id) return;
    const el = document.getElementById(id) || document.querySelector('[data-mid="' + id + '"]');
    if (el) {
      el.scrollIntoView({ block: "center", behavior: "smooth" });
      el.classList.add("msg-focus");
      setTimeout(function () {
        el.classList.remove("msg-focus");
      }, 2000);
    }
  }
  window.addEventListener("hashchange", jumpHash);
  setTimeout(jumpHash, 400);

  window.rabbitFocusMode = {
    toggle: function () {
      applyFocus(!document.documentElement.classList.contains("focus-mode"));
    },
  };
})();
