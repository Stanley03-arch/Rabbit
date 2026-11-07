/* Rabbit dashboard extras — settings, continuous listen helpers, tok/s */
(function () {
  const SETTINGS_KEY = "rabbit-dashboard-v1";
  const tpsEl = document.getElementById("tps");
  const voiceContinuous = document.getElementById("voice-continuous");
  const streamEl = document.getElementById("stream");
  const systemEl = document.getElementById("system");
  const voiceAutoSend = document.getElementById("voice-auto-send");
  const voiceSpeak = document.getElementById("voice-speak");
  const voiceLang = document.getElementById("voice-lang");
  const voiceRate = document.getElementById("voice-rate");
  const latencyEl = document.getElementById("latency");

  window.rabbitContinuousArmed = false;

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
  ].forEach(function (id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", saveSettings);
    el.addEventListener("input", saveSettings);
  });

  loadSettings();
})();
