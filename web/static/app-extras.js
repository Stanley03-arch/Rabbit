/* Rabbit dashboard extras — settings, history, connection, regenerate, prompts, composer */
(function () {
  const SETTINGS_KEY = "rabbit-dashboard-v1";
  const HISTORY_KEY = "rabbit-dashboard-history-v1";
  const PROMPTS_KEY = "rabbit-dashboard-prompts-v1";

  const tpsEl = document.getElementById("tps");
  const voiceContinuous = document.getElementById("voice-continuous");
  const streamEl = document.getElementById("stream");
  const systemEl = document.getElementById("system");
  const voiceAutoSend = document.getElementById("voice-auto-send");
  const voiceSpeak = document.getElementById("voice-speak");
  const voiceLang = document.getElementById("voice-lang");
  const voiceRate = document.getElementById("voice-rate");
  const latencyEl = document.getElementById("latency");
  const persistChatEl = document.getElementById("persist-chat");
  const connEl = document.getElementById("conn-status");
  const regenBtn = document.getElementById("regen-last");
  const chatEl = document.getElementById("chat");
  const form = document.getElementById("form");
  const promptEl = document.getElementById("prompt");
  const emptyState = document.getElementById("empty-state");
  const autoScrollEl = document.getElementById("auto-scroll");
  const composerCount = document.getElementById("composer-count");
  const savePromptBtn = document.getElementById("save-prompt");
  const chipsEl = document.getElementById("chips");

  window.rabbitContinuousArmed = false;
  let restoring = false;
  let observed = false;

  function loadSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (!raw) return;
      const s = JSON.parse(raw);
      if (s.tokens != null) document.getElementById("tokens").value = s.tokens;
      if (s.temp != null) document.getElementById("temp").value = s.temp;
      if (s.top_p != null) document.getElementById("top-p").value = s.top_p;
      if (s.seed != null) document.getElementById("seed").value = s.seed;
      if (typeof s.stream === "boolean" && streamEl) streamEl.checked = s.stream;
      if (s.system != null && systemEl) systemEl.value = s.system;
      if (typeof s.voice_auto_send === "boolean" && voiceAutoSend)
        voiceAutoSend.checked = s.voice_auto_send;
      if (typeof s.voice_speak === "boolean" && voiceSpeak)
        voiceSpeak.checked = s.voice_speak;
      if (typeof s.voice_continuous === "boolean" && voiceContinuous)
        voiceContinuous.checked = s.voice_continuous;
      if (s.voice_lang && voiceLang) voiceLang.value = s.voice_lang;
      if (s.voice_rate != null && voiceRate) voiceRate.value = s.voice_rate;
      if (typeof s.persist_chat === "boolean" && persistChatEl)
        persistChatEl.checked = s.persist_chat;
      if (typeof s.auto_scroll === "boolean" && autoScrollEl)
        autoScrollEl.checked = s.auto_scroll;
    } catch (e) {
      console.warn(e);
    }
  }

  function saveSettings() {
    try {
      localStorage.setItem(
        SETTINGS_KEY,
        JSON.stringify({
          tokens: Number(document.getElementById("tokens").value),
          temp: Number(document.getElementById("temp").value),
          top_p: Number(document.getElementById("top-p").value),
          seed: document.getElementById("seed").value,
          stream: !!(streamEl && streamEl.checked),
          system: systemEl ? systemEl.value : "",
          voice_auto_send: !!(voiceAutoSend && voiceAutoSend.checked),
          voice_speak: !!(voiceSpeak && voiceSpeak.checked),
          voice_continuous: !!(voiceContinuous && voiceContinuous.checked),
          voice_lang: voiceLang ? voiceLang.value : "en-US",
          voice_rate: voiceRate ? Number(voiceRate.value) : 1,
          persist_chat: !!(persistChatEl && persistChatEl.checked),
          auto_scroll: !!(autoScrollEl && autoScrollEl.checked),
        })
      );
    } catch (e) {
      console.warn(e);
    }
  }

  window.rabbitSaveSettings = saveSettings;
  window.rabbitLoadSettings = loadSettings;

  window.rabbitEstimateTokens = function (text) {
    const t = String(text || "").trim();
    if (!t) return 0;
    return Math.max(1, Math.round(t.split(/\s+/).filter(Boolean).length * 1.3));
  };

  window.rabbitRecordSpeed = function (ms, text) {
    if (!ms || ms <= 0) {
      if (tpsEl) tpsEl.textContent = "\u2014";
      const met = document.getElementById("met-tps");
      if (met) met.textContent = "\u2014";
      return;
    }
    const toks = window.rabbitEstimateTokens(text);
    const tps = ((toks * 1000) / ms).toFixed(1);
    if (tpsEl) tpsEl.textContent = tps + " tok/s";
    const met = document.getElementById("met-tps");
    if (met) met.textContent = tps + " tok/s";
    if (latencyEl) latencyEl.textContent = ms + " ms";
  };

  window.rabbitShouldContinueListening = function () {
    return !!(
      window.rabbitContinuousArmed ||
      (voiceContinuous && voiceContinuous.checked)
    );
  };

  function shouldPersistChat() {
    return !!(persistChatEl && persistChatEl.checked);
  }

  function shouldAutoScroll() {
    return !autoScrollEl || autoScrollEl.checked;
  }

  function readBubbles() {
    if (!chatEl) return [];
    return Array.from(chatEl.querySelectorAll(".bubble")).map(function (div) {
      const role = div.classList.contains("user") ? "user" : "bot";
      const body = div.querySelector(".body");
      return { role: role, content: body ? body.textContent : "", ts: Date.now() };
    });
  }

  function saveHistoryFromDom() {
    if (restoring || !shouldPersistChat()) return;
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(readBubbles().slice(-80)));
    } catch (e) {
      console.warn(e);
    }
  }

  function injectBubble(role, text) {
    if (!chatEl) return;
    if (emptyState) emptyState.style.display = "none";
    const div = document.createElement("div");
    div.className = "bubble " + role;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = (role === "user" ? "you" : "rabbit") + " \u00b7 restored";
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = text || "";
    div.appendChild(meta);
    div.appendChild(body);
    chatEl.appendChild(div);
  }

  function restoreHistory() {
    if (!shouldPersistChat() || !chatEl) return;
    if (chatEl.querySelector(".bubble")) return;
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      if (!raw) return;
      const msgs = JSON.parse(raw);
      if (!Array.isArray(msgs) || !msgs.length) return;
      restoring = true;
      msgs.forEach(function (m) {
        injectBubble(m.role === "user" ? "user" : "bot", m.content || "");
      });
      restoring = false;
      if (shouldAutoScroll()) chatEl.scrollTop = chatEl.scrollHeight;
    } catch (e) {
      restoring = false;
      console.warn(e);
    }
  }

  function regenerateLast() {
    if (!chatEl || !form || !promptEl) return;
    const bubbles = Array.from(chatEl.querySelectorAll(".bubble"));
    if (!bubbles.length) return;
    let lastUserIdx = -1;
    for (let i = bubbles.length - 1; i >= 0; i--) {
      if (bubbles[i].classList.contains("user")) {
        lastUserIdx = i;
        break;
      }
    }
    if (lastUserIdx < 0) return;
    const body = bubbles[lastUserIdx].querySelector(".body");
    const msg = body ? body.textContent : "";
    if (!msg.trim()) return;
    for (let i = bubbles.length - 1; i >= lastUserIdx; i--) bubbles[i].remove();
    if (!chatEl.querySelector(".bubble") && emptyState) emptyState.style.display = "";
    promptEl.value = msg.trim();
    updateComposerCount();
    form.requestSubmit();
  }

  function ensureObserver() {
    if (observed || !chatEl) return;
    observed = true;
    const obs = new MutationObserver(function () {
      saveHistoryFromDom();
      if (shouldAutoScroll()) chatEl.scrollTop = chatEl.scrollHeight;
    });
    obs.observe(chatEl, { childList: true, subtree: true, characterData: true });
  }

  window.rabbitOnAppReady = function () {
    restoreHistory();
    ensureObserver();
  };

  function updateComposerCount() {
    if (!composerCount || !promptEl) return;
    const t = promptEl.value || "";
    const chars = t.length;
    const toks = t.trim() ? window.rabbitEstimateTokens(t) : 0;
    composerCount.textContent = chars + " chars \u00b7 ~" + toks + " tok";
  }

  if (promptEl) {
    promptEl.addEventListener("input", updateComposerCount);
    promptEl.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" && !ev.shiftKey) {
        ev.preventDefault();
        if (form) form.requestSubmit();
      }
    });
  }

  function loadSavedPrompts() {
    try {
      const raw = localStorage.getItem(PROMPTS_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  }

  function storeSavedPrompts(list) {
    try {
      localStorage.setItem(PROMPTS_KEY, JSON.stringify(list.slice(0, 24)));
    } catch (e) {
      console.warn(e);
    }
  }

  function renderSavedPrompts() {
    if (!chipsEl) return;
    chipsEl.querySelectorAll(".chip.custom").forEach(function (c) {
      c.remove();
    });
    loadSavedPrompts().forEach(function (p, idx) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip custom";
      btn.setAttribute("data-prompt", p);
      const label = p.length > 28 ? p.slice(0, 28) + "\u2026" : p;
      btn.appendChild(document.createTextNode(label));
      const x = document.createElement("span");
      x.className = "chip-x";
      x.textContent = "\u00d7";
      x.title = "Remove saved prompt";
      x.addEventListener("click", function (ev) {
        ev.stopPropagation();
        const list = loadSavedPrompts();
        list.splice(idx, 1);
        storeSavedPrompts(list);
        renderSavedPrompts();
      });
      btn.appendChild(x);
      btn.addEventListener("click", function (ev) {
        if (ev.target === x) return;
        if (promptEl) {
          promptEl.value = p;
          updateComposerCount();
          promptEl.focus();
        }
      });
      chipsEl.appendChild(btn);
    });
  }

  if (savePromptBtn) {
    savePromptBtn.addEventListener("click", function () {
      const t = ((promptEl && promptEl.value) || "").trim();
      if (!t) return;
      const list = loadSavedPrompts();
      if (list.indexOf(t) >= 0) return;
      list.unshift(t);
      storeSavedPrompts(list);
      renderSavedPrompts();
    });
  }

  async function pingConnection() {
    if (!connEl) return;
    try {
      const r = await fetch("/api/info", { cache: "no-store" });
      if (r.ok) {
        connEl.textContent = "online";
        connEl.className = "conn online";
      } else {
        connEl.textContent = "error " + r.status;
        connEl.className = "conn offline";
      }
    } catch (e) {
      connEl.textContent = "offline";
      connEl.className = "conn offline";
    }
  }

  document.addEventListener("keydown", function (ev) {
    if (ev.ctrlKey && !ev.altKey && !ev.metaKey) {
      if (ev.key === "l" || ev.key === "L") {
        const clearBtn = document.getElementById("clear-chat");
        if (clearBtn) {
          ev.preventDefault();
          clearBtn.click();
        }
      }
      if (ev.key === "r" || ev.key === "R") {
        ev.preventDefault();
        regenerateLast();
      }
      if ((ev.key === "s" || ev.key === "S") && promptEl && document.activeElement === promptEl) {
        ev.preventDefault();
        if (savePromptBtn) savePromptBtn.click();
      }
    }
  });

  [
    "tokens",
    "temp",
    "top-p",
    "seed",
    "stream",
    "system",
    "voice-auto-send",
    "voice-speak",
    "voice-continuous",
    "voice-lang",
    "voice-rate",
    "persist-chat",
    "auto-scroll",
  ].forEach(function (id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", saveSettings);
    el.addEventListener("input", saveSettings);
  });

  if (persistChatEl) {
    persistChatEl.addEventListener("change", function () {
      saveSettings();
      if (!persistChatEl.checked) {
        try {
          localStorage.removeItem(HISTORY_KEY);
        } catch (e) {}
      } else saveHistoryFromDom();
    });
  }

  if (regenBtn) regenBtn.addEventListener("click", regenerateLast);

  loadSettings();
  renderSavedPrompts();
  updateComposerCount();
  pingConnection();
  setInterval(pingConnection, 10000);

  window.addEventListener("load", function () {
    setTimeout(function () {
      restoreHistory();
      ensureObserver();
      updateComposerCount();
    }, 400);
  });
})();
