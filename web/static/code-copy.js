/* Copy buttons on fenced code in bubbles + Continue chip + font scale keys */
(function () {
  const chatEl = document.getElementById("chat");
  const promptEl = document.getElementById("prompt");
  const formEl = document.getElementById("form");
  if (!chatEl) return;

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
    return Promise.resolve();
  }

  function enhanceBubble(b) {
    if (!b || b.dataset.codeCopy === "1") return;
    b.dataset.codeCopy = "1";
    const body = b.querySelector(".body");
    if (!body) return;

    // Wrap ``` fenced regions as pre blocks if still plain text
    if (!body.querySelector("pre") && body.dataset.raw) {
      const raw = body.dataset.raw;
      if (raw.indexOf("```") >= 0) {
        // leave raw text; still add a top-level copy for long messages
      }
    }

    body.querySelectorAll("pre").forEach(function (pre) {
      if (pre.parentElement && pre.parentElement.classList.contains("code-wrap")) return;
      const wrap = document.createElement("div");
      wrap.className = "code-wrap";
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "code-copy-btn";
      btn.textContent = "Copy";
      btn.addEventListener("click", function (ev) {
        ev.stopPropagation();
        copyText(pre.textContent || "").then(function () {
          btn.textContent = "Copied";
          if (window.rabbitToast) rabbitToast("Code copied", "ok");
          setTimeout(function () {
            btn.textContent = "Copy";
          }, 1200);
        });
      });
      wrap.appendChild(btn);
    });
  }

  function enhanceAll() {
    chatEl.querySelectorAll(".bubble").forEach(enhanceBubble);
  }

  const obs = new MutationObserver(function () {
    enhanceAll();
    maybeContinueChip();
  });
  obs.observe(chatEl, { childList: true, subtree: true });
  enhanceAll();

  // Continue chip after last bot message
  let contBar = document.getElementById("continue-bar");
  if (!contBar) {
    contBar = document.createElement("div");
    contBar.id = "continue-bar";
    contBar.className = "continue-bar";
    contBar.hidden = true;
    contBar.innerHTML =
      '<button type="button" class="chip continue-chip" id="continue-gen">Continue</button>' +
      '<button type="button" class="chip continue-chip" id="continue-shorter">Shorter</button>' +
      '<button type="button" class="chip continue-chip" id="continue-detail">More detail</button>';
    const tools = document.querySelector(".composer-tools");
    if (tools && tools.parentNode) tools.parentNode.insertBefore(contBar, tools);
    else chatEl.parentNode.appendChild(contBar);
  }

  function maybeContinueChip() {
    const bots = chatEl.querySelectorAll(".bubble.bot:not(.streaming)");
    const last = bots[bots.length - 1];
    const busy = document.querySelector(".bubble.bot.streaming") || (document.getElementById("stop") && !document.getElementById("stop").hidden);
    contBar.hidden = !last || !!busy;
  }

  function sendPrompt(text) {
    if (!promptEl || !formEl) return;
    promptEl.value = text;
    promptEl.dispatchEvent(new Event("input", { bubbles: true }));
    if (typeof formEl.requestSubmit === "function") formEl.requestSubmit();
    else formEl.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
  }

  contBar.addEventListener("click", function (ev) {
    const id = ev.target && ev.target.id;
    if (id === "continue-gen") sendPrompt("Continue.");
    if (id === "continue-shorter") sendPrompt("Make that shorter.");
    if (id === "continue-detail") sendPrompt("Go into more detail.");
  });

  // Font scale with Ctrl+= / Ctrl+-
  const scaleEl = document.getElementById("font-scale");
  function bumpScale(dir) {
    if (!scaleEl) return;
    const opts = Array.from(scaleEl.options).map(function (o) {
      return o.value;
    });
    let i = opts.indexOf(scaleEl.value);
    if (i < 0) i = opts.indexOf("1");
    i = Math.max(0, Math.min(opts.length - 1, i + dir));
    scaleEl.value = opts[i];
    scaleEl.dispatchEvent(new Event("change", { bubbles: true }));
    document.documentElement.style.fontSize = 16 * parseFloat(opts[i]) + "px";
    if (window.rabbitToast) rabbitToast("UI scale " + opts[i], "ok");
  }

  document.addEventListener("keydown", function (ev) {
    if (!(ev.ctrlKey || ev.metaKey)) return;
    if (ev.key === "=" || ev.key === "+") {
      ev.preventDefault();
      bumpScale(1);
    }
    if (ev.key === "-" || ev.key === "_") {
      ev.preventDefault();
      bumpScale(-1);
    }
    if (ev.key === "0") {
      ev.preventDefault();
      if (scaleEl) {
        scaleEl.value = "1";
        scaleEl.dispatchEvent(new Event("change", { bubbles: true }));
        document.documentElement.style.fontSize = "";
      }
    }
  });

  // Apply stored scale on load if select exists
  if (scaleEl) {
    scaleEl.addEventListener("change", function () {
      const v = parseFloat(scaleEl.value) || 1;
      document.documentElement.style.fontSize = v === 1 ? "" : 16 * v + "px";
    });
  }
})();
