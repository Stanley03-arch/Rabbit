/* Bookmarks jump list + Ctrl+Enter send + first-visit tips */
(function () {
  const BM_KEY = "rabbit-bookmarks-v1";
  const TIPS_KEY = "rabbit-tips-seen-v2";
  const chatEl = document.getElementById("chat");
  const promptEl = document.getElementById("prompt");
  const formEl = document.getElementById("form");

  function loadBm() {
    try {
      const raw = localStorage.getItem(BM_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function saveBm(arr) {
    try {
      localStorage.setItem(BM_KEY, JSON.stringify(arr.slice(0, 30)));
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
    return role + ":" + String(raw || "").slice(0, 100);
  }

  function ensureBmBar() {
    let bar = document.getElementById("bookmarks-bar");
    if (bar) return bar;
    bar = document.createElement("div");
    bar.id = "bookmarks-bar";
    bar.className = "bookmarks-bar";
    bar.hidden = true;
    const strip = document.getElementById("pinned-strip");
    if (strip && strip.parentNode) strip.parentNode.insertBefore(bar, strip.nextSibling);
    else if (chatEl) chatEl.parentNode.insertBefore(bar, chatEl);
    else document.body.appendChild(bar);
    return bar;
  }

  function renderBm() {
    const bar = ensureBmBar();
    const list = loadBm();
    if (!list.length) {
      bar.hidden = true;
      bar.innerHTML = "";
      return;
    }
    bar.hidden = false;
    bar.innerHTML =
      '<span class="bm-label">★</span>' +
      list
        .map(function (item, i) {
          const label = (item.label || item.text || "").replace(/\s+/g, " ").slice(0, 36);
          return (
            '<button type="button" class="bm-chip" data-i="' +
            i +
            '" title="' +
            escapeAttr(item.text || "") +
            '">' +
            escapeHtml(label) +
            (label.length >= 36 ? "…" : "") +
            "</button>" +
            '<button type="button" class="bm-x" data-x="' +
            i +
            '" title="Remove">×</button>'
          );
        })
        .join("");
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
  function escapeAttr(s) {
    return escapeHtml(s).replace(/"/g, "&quot;");
  }

  document.body.addEventListener("click", function (ev) {
    const bar = document.getElementById("bookmarks-bar");
    if (!bar) return;
    const chip = ev.target.closest(".bm-chip");
    if (chip && bar.contains(chip)) {
      const item = loadBm()[+chip.dataset.i];
      if (!item) return;
      if (item.mid) {
        const el = document.getElementById(item.mid) || document.querySelector('[data-mid="' + item.mid + '"]');
        if (el) {
          el.scrollIntoView({ block: "center", behavior: "smooth" });
          el.classList.add("msg-focus");
          setTimeout(function () {
            el.classList.remove("msg-focus");
          }, 1500);
          return;
        }
      }
      if (promptEl && item.text) {
        promptEl.value = item.text;
        promptEl.focus();
        promptEl.dispatchEvent(new Event("input", { bubbles: true }));
      }
      return;
    }
    const x = ev.target.closest(".bm-x");
    if (x && bar.contains(x)) {
      const list = loadBm();
      list.splice(+x.dataset.x, 1);
      saveBm(list);
      renderBm();
    }
  });

  function enhance(b) {
    if (!b || b.dataset.bmReady === "1") return;
    b.dataset.bmReady = "1";
    const host = b.querySelector(".bubble-actions") || b;
    if (host.querySelector(".bm-msg-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "bubble-act bm-msg-btn";
    btn.textContent = "★";
    btn.title = "Bookmark this message";
    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      const body = b.querySelector(".body");
      const text =
        body && body.dataset && body.dataset.raw != null
          ? body.dataset.raw
          : body
            ? body.textContent
            : "";
      const mid = b.id || b.dataset.mid || "";
      const key = bubbleKey(b);
      let list = loadBm().filter(function (x) {
        return x.key !== key;
      });
      list.unshift({
        key: key,
        text: (text || "").trim(),
        label: (text || "").trim().slice(0, 40),
        mid: mid,
        t: Date.now(),
      });
      saveBm(list);
      renderBm();
      if (window.rabbitToast) rabbitToast("Bookmarked", "ok");
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
  renderBm();

  // Ctrl+Enter always sends (even if Enter-to-send is off)
  if (promptEl && formEl) {
    promptEl.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter") return;
      if (!(ev.ctrlKey || ev.metaKey)) return;
      ev.preventDefault();
      if (typeof formEl.requestSubmit === "function") formEl.requestSubmit();
      else formEl.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });
  }

  // First-visit tips strip
  try {
    if (localStorage.getItem(TIPS_KEY) !== "1") {
      const tips = document.createElement("div");
      tips.id = "first-tips";
      tips.className = "first-tips";
      tips.innerHTML =
        "<div><strong>Quick tips:</strong> <kbd>Ctrl+K</kbd> palette · <kbd>/</kbd> search · <kbd>Ctrl+Enter</kbd> send · ★ bookmark · <kbd>Ctrl+Shift+A</kbd> archive</div>" +
        '<button type="button" id="tips-dismiss">Got it</button>';
      const main = document.querySelector(".main");
      if (main) main.insertBefore(tips, main.firstChild);
      else document.body.appendChild(tips);
      document.getElementById("tips-dismiss").addEventListener("click", function () {
        tips.remove();
        try {
          localStorage.setItem(TIPS_KEY, "1");
        } catch (e) {}
      });
    }
  } catch (e) {}

  window.rabbitBookmarks = { list: loadBm, render: renderBm };
})();
