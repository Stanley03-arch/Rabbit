/* Composer history (↑/↓), panic mute, unread session badges */
(function () {
  const HIST_KEY = "rabbit-prompt-history-v1";
  const promptEl = document.getElementById("prompt");
  const formEl = document.getElementById("form");
  if (!promptEl) return;

  function loadHist() {
    try {
      const raw = localStorage.getItem(HIST_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function saveHist(arr) {
    try {
      localStorage.setItem(HIST_KEY, JSON.stringify(arr.slice(0, 50)));
    } catch (e) {}
  }

  let hist = loadHist();
  let histIdx = -1;
  let draft = "";

  if (formEl) {
    formEl.addEventListener("submit", function () {
      const t = (promptEl.value || "").trim();
      if (!t) return;
      hist = hist.filter(function (x) {
        return x !== t;
      });
      hist.unshift(t);
      saveHist(hist);
      histIdx = -1;
      draft = "";
    });
  }

  promptEl.addEventListener("keydown", function (ev) {
    if (ev.key !== "ArrowUp" && ev.key !== "ArrowDown") return;
    // only when caret at start (up) or navigating history
    const val = promptEl.value || "";
    const start = promptEl.selectionStart || 0;
    if (ev.key === "ArrowUp") {
      if (histIdx === -1 && start > 0 && val.indexOf("\n") >= 0) return; // multi-line edit
      if (histIdx === -1 && start > 0) return;
      if (!hist.length) return;
      ev.preventDefault();
      if (histIdx === -1) draft = val;
      histIdx = Math.min(hist.length - 1, histIdx + 1);
      promptEl.value = hist[histIdx] || "";
      promptEl.setSelectionRange(promptEl.value.length, promptEl.value.length);
      promptEl.dispatchEvent(new Event("input", { bubbles: true }));
    } else if (ev.key === "ArrowDown") {
      if (histIdx < 0) return;
      ev.preventDefault();
      histIdx -= 1;
      if (histIdx < 0) {
        promptEl.value = draft;
      } else {
        promptEl.value = hist[histIdx] || "";
      }
      promptEl.setSelectionRange(promptEl.value.length, promptEl.value.length);
      promptEl.dispatchEvent(new Event("input", { bubbles: true }));
    }
  });

  // Panic: Ctrl+Shift+Esc or Ctrl+Shift+X — stop TTS, mic, generation
  function panicMute() {
    try {
      if (window.speechSynthesis) speechSynthesis.cancel();
    } catch (e) {}
    const ttsStop = document.getElementById("tts-stop");
    if (ttsStop && !ttsStop.hidden) ttsStop.click();
    const stop = document.getElementById("stop");
    if (stop && !stop.hidden) stop.click();
    const mic = document.getElementById("mic");
    if (mic && mic.classList.contains("active")) mic.click();
    const voiceTop = document.getElementById("voice-toggle-top");
    if (voiceTop && voiceTop.getAttribute("aria-pressed") === "true") voiceTop.click();
    if (window.rabbitToast) rabbitToast("Stopped audio & generation", "ok");
  }

  document.addEventListener("keydown", function (ev) {
    if (!(ev.ctrlKey || ev.metaKey) || !ev.shiftKey) return;
    if (ev.key === "X" || ev.key === "x" || ev.key === "Escape") {
      ev.preventDefault();
      panicMute();
    }
  });

  // Unread badges: when a bot bubble is added while another session is active
  const tabsHost = document.getElementById("session-tabs");
  const chatEl = document.getElementById("chat");
  const unread = {};

  function activeSessionId() {
    const t = tabsHost && tabsHost.querySelector(".session-tab.active");
    if (!t) return "";
    return t.dataset.sessionId || t.dataset.id || t.getAttribute("data-id") || t.textContent.trim();
  }

  function paintUnread() {
    if (!tabsHost) return;
    tabsHost.querySelectorAll(".session-tab").forEach(function (tab) {
      const id = tab.dataset.sessionId || tab.dataset.id || tab.getAttribute("data-id") || tab.textContent.trim();
      let badge = tab.querySelector(".unread-badge");
      const n = unread[id] || 0;
      if (n > 0 && !tab.classList.contains("active")) {
        if (!badge) {
          badge = document.createElement("span");
          badge.className = "unread-badge";
          tab.appendChild(badge);
        }
        badge.textContent = n > 9 ? "9+" : String(n);
      } else if (badge) {
        badge.remove();
      }
    });
  }

  if (tabsHost) {
    tabsHost.addEventListener("click", function (ev) {
      const tab = ev.target.closest(".session-tab");
      if (!tab) return;
      const id = tab.dataset.sessionId || tab.dataset.id || tab.getAttribute("data-id") || tab.textContent.trim();
      unread[id] = 0;
      paintUnread();
    });
  }

  if (chatEl) {
    new MutationObserver(function (muts) {
      let addedBot = false;
      muts.forEach(function (m) {
        m.addedNodes &&
          m.addedNodes.forEach(function (n) {
            if (n.nodeType === 1 && n.classList && n.classList.contains("bubble") && n.classList.contains("bot")) {
              addedBot = true;
            }
          });
      });
      if (!addedBot) return;
      // mark other sessions? we only know current session got a bot reply — clear current
      const cur = activeSessionId();
      if (cur) unread[cur] = 0;
      paintUnread();
    }).observe(chatEl, { childList: true, subtree: true });
  }

  // When queue delivers while user might have switched — expose API
  window.rabbitPromptHistory = {
    list: function () {
      return hist.slice();
    },
    panicMute: panicMute,
    markUnread: function (sessionId) {
      if (!sessionId) return;
      unread[sessionId] = (unread[sessionId] || 0) + 1;
      paintUnread();
    },
  };
})();
