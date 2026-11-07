const chatEl = document.getElementById("chat");
const form = document.getElementById("form");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const stopBtn = document.getElementById("stop");
const statusPill = document.getElementById("status-pill");
const clearBtn = document.getElementById("clear-chat");
const copyBtn = document.getElementById("copy-last");
const refreshBtn = document.getElementById("refresh-stats");
const exportBtn = document.getElementById("export-chat");
const importBtn = document.getElementById("import-chat");
const importFile = document.getElementById("import-file");
const emptyState = document.getElementById("empty-state");
const latencyEl = document.getElementById("latency");
const tpsEl = document.getElementById("tps");
const streamEl = document.getElementById("stream");
const systemEl = document.getElementById("system");
const micBtn = document.getElementById("mic");
const ttsStopBtn = document.getElementById("tts-stop");
const voiceAutoSend = document.getElementById("voice-auto-send");
const voiceSpeak = document.getElementById("voice-speak");
const voiceContinuous = document.getElementById("voice-continuous");
const voiceLang = document.getElementById("voice-lang");
const voiceRate = document.getElementById("voice-rate");
const voiceMicStatus = document.getElementById("voice-mic-status");
const voiceHint = document.getElementById("voice-hint");
const voiceCmdLog = document.getElementById("voice-cmd-log");

let lastReply = "";
let abortCtrl = null;
let history = [];
let recognition = null;
let listening = false;
let interimTranscript = "";
let skipAutoSendOnce = false;
let continuousArmed = false;

function hideEmpty() {
  if (emptyState) emptyState.style.display = "none";
}

function showEmpty() {
  if (emptyState) emptyState.style.display = "";
}

function addBubble(role, text, ms) {
  hideEmpty();
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  const meta = document.createElement("div");
  meta.className = "meta";
  const ts = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  let label = role === "user" ? "you" : "rabbit";
  if (ms != null && role === "bot") label += ` · ${ms} ms`;
  meta.textContent = `${label} · ${ts}`;
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  div.appendChild(meta);
  div.appendChild(body);
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  if (role === "bot") lastReply = text;
  history.push({ role, content: text, ts: Date.now(), ms: ms ?? null });
  return { div, body, meta };
}

function setStatus(mode, label) {
  statusPill.className = "status-pill " + (mode || "");
  statusPill.textContent = label;
}

function renderExperts(n, warmIds, nShared) {
  const grid = document.getElementById("expert-grid");
  grid.innerHTML = "";
  const routed = Math.min(Math.max(n || 8, 1), 32);
  const shared = Math.min(Math.max(nShared || 0, 0), 8);
  const warm = new Set(warmIds || []);
  for (let i = 0; i < routed; i++) {
    const cell = document.createElement("div");
    cell.className = "expert-cell";
    if (warm.has(i)) cell.classList.add(i % 3 === 0 ? "hot" : "warm");
    cell.textContent = String(i);
    cell.title = `routed expert ${i}`;
    grid.appendChild(cell);
  }
  for (let s = 0; s < shared; s++) {
    const cell = document.createElement("div");
    cell.className = "expert-cell shared";
    cell.textContent = `S${s}`;
    cell.title = `shared expert ${s} (always-on)`;
    grid.appendChild(cell);
  }
}

function quantLabel(q) {
  if (q === 0 || q === "0") return "f32";
  if (q == null) return "—";
  return `${q}-bit`;
}

function logVoiceCmd(name) {
  if (voiceCmdLog) voiceCmdLog.textContent = name;
  setStatus("ok", "cmd: " + name);
}

function normalizeVoiceText(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/[.,!?;:]+/g, " ")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const VOICE_COMMANDS = [
  { id: "clear", phrases: ["clear", "clear chat", "reset chat", "new chat", "start over"] },
  { id: "stop", phrases: ["stop", "stop generating", "cancel", "abort"] },
  { id: "mute", phrases: ["mute", "be quiet", "stop talking", "shut up", "silence"] },
  { id: "speak_on", phrases: ["speak on", "enable speech", "turn on speech", "voice on"] },
  { id: "speak_off", phrases: ["speak off", "disable speech", "turn off speech", "voice off"] },
  { id: "stream_on", phrases: ["stream on", "enable stream", "streaming on"] },
  { id: "stream_off", phrases: ["stream off", "disable stream", "streaming off"] },
  { id: "listen_on", phrases: ["listen on", "continuous on", "keep listening"] },
  { id: "listen_off", phrases: ["listen off", "continuous off", "stop listening"] },
  { id: "copy", phrases: ["copy", "copy reply", "copy last", "copy that"] },
  { id: "export", phrases: ["export", "export chat", "save chat", "download chat"] },
  { id: "refresh", phrases: ["refresh", "refresh stats", "update stats", "reload stats"] },
  { id: "repeat", phrases: ["repeat", "say again", "read again", "speak again"] },
  { id: "help", phrases: ["help", "voice help", "commands", "what can i say"] },
  { id: "send", phrases: ["send", "submit", "go"] },
];

function matchVoiceCommand(text) {
  const n = normalizeVoiceText(text);
  if (!n) return null;
  for (const cmd of VOICE_COMMANDS) {
    for (const p of cmd.phrases) {
      if (n === p) return cmd.id;
    }
  }
  const stripped = n
    .replace(/^(hey |ok |okay )?rabbit\s+/, "")
    .replace(/^computer\s+/, "")
    .trim();
  if (stripped && stripped !== n) {
    for (const cmd of VOICE_COMMANDS) {
      for (const p of cmd.phrases) {
        if (stripped === p) return cmd.id;
      }
    }
  }
  return null;
}

function runVoiceCommand(id) {
  switch (id) {
    case "clear":
      clearChat();
      logVoiceCmd("clear");
      return true;
    case "stop":
      if (abortCtrl) abortCtrl.abort();
      stopListening();
      stopSpeaking();
      logVoiceCmd("stop");
      return true;
    case "mute":
      stopSpeaking();
      logVoiceCmd("mute");
      return true;
    case "speak_on":
      if (voiceSpeak) voiceSpeak.checked = true;
      logVoiceCmd("speak on");
      return true;
    case "speak_off":
      if (voiceSpeak) voiceSpeak.checked = false;
      stopSpeaking();
      logVoiceCmd("speak off");
      return true;
    case "stream_on":
      if (streamEl) streamEl.checked = true;
      logVoiceCmd("stream on");
      return true;
    case "stream_off":
      if (streamEl) streamEl.checked = false;
      if (typeof rabbitSaveSettings === "function") rabbitSaveSettings();
      logVoiceCmd("stream off");
      return true;
    case "listen_on":
      if (voiceContinuous) voiceContinuous.checked = true;
      continuousArmed = true;
      if (typeof rabbitSaveSettings === "function") rabbitSaveSettings();
      startListening();
      logVoiceCmd("listen on");
      return true;
    case "listen_off":
      continuousArmed = false;
      if (voiceContinuous) voiceContinuous.checked = false;
      stopListening();
      if (typeof rabbitSaveSettings === "function") rabbitSaveSettings();
      logVoiceCmd("listen off");
      return true;
    case "copy":
      copyLast();
      logVoiceCmd("copy");
      return true;
    case "export":
      exportChat();
      logVoiceCmd("export");
      return true;
    case "refresh":
      refreshInfo();
      logVoiceCmd("refresh");
      return true;
    case "repeat":
      if (lastReply) speakText(lastReply);
      else setStatus("", "nothing to repeat");
      logVoiceCmd("repeat");
      return true;
    case "help": {
      const list =
        "Voice commands: clear, stop, mute, speak on, speak off, stream on, stream off, listen on, listen off, copy, export, refresh, repeat, send, help.";
      addBubble("bot", list);
      if (voiceSpeak && voiceSpeak.checked) speakText(list);
      logVoiceCmd("help");
      return true;
    }
    case "send": {
      const t = (promptEl.value || "").trim();
      if (t) form.requestSubmit();
      else setStatus("", "nothing to send");
      logVoiceCmd("send");
      return true;
    }
    default:
      return false;
  }
}

function tryHandleAsVoiceCommand(text) {
  const id = matchVoiceCommand(text);
  if (!id) return false;
  promptEl.value = "";
  delete promptEl.dataset.committed;
  skipAutoSendOnce = true;
  runVoiceCommand(id);
  return true;
}

function stopSpeaking() {
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
  if (ttsStopBtn) ttsStopBtn.hidden = true;
}

function speakText(text) {
  if (!voiceSpeak || !voiceSpeak.checked) return;
  if (!window.speechSynthesis) return;
  const clean = String(text || "")
    .replace(/\[error[^\]]*\]/gi, "")
    .replace(/\[stopped\]/gi, "")
    .replace(/---[\s\S]*$/m, "")
    .replace(/\*+/g, "")
    .trim();
  if (!clean) return;
  stopSpeaking();
  const u = new SpeechSynthesisUtterance(clean.slice(0, 1500));
  u.lang = (voiceLang && voiceLang.value) || "en-US";
  const rate = Number(voiceRate && voiceRate.value);
  u.rate = Number.isFinite(rate) ? Math.min(2, Math.max(0.5, rate)) : 1;
  u.pitch = 1;
  u.onstart = () => {
    if (ttsStopBtn) ttsStopBtn.hidden = false;
  };
  u.onend = () => {
    if (ttsStopBtn) ttsStopBtn.hidden = true;
  };
  u.onerror = () => {
    if (ttsStopBtn) ttsStopBtn.hidden = true;
  };
  window.speechSynthesis.speak(u);
}

if (ttsStopBtn) ttsStopBtn.addEventListener("click", stopSpeaking);

function setMicUI(on) {
  listening = on;
  if (micBtn) micBtn.classList.toggle("listening", on);
  if (voiceMicStatus) voiceMicStatus.textContent = on ? "listening" : "idle";
  if (on) setStatus("busy", "listening");
}

function initSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    if (voiceHint)
      voiceHint.textContent =
        "Speech recognition not supported in this browser. Use Chrome or Edge.";
    if (micBtn) {
      micBtn.disabled = true;
      micBtn.title = "Speech recognition unavailable";
    }
    if (voiceMicStatus) voiceMicStatus.textContent = "unsupported";
    return null;
  }
  const rec = new SR();
  rec.continuous = false;
  rec.interimResults = true;
  rec.maxAlternatives = 1;

  rec.onstart = () => {
    setMicUI(true);
    interimTranscript = "";
    skipAutoSendOnce = false;
  };

  rec.onresult = (event) => {
    let interim = "";
    let final = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) final += t;
      else interim += t;
    }
    if (interim) {
      interimTranscript = interim;
      promptEl.value = (promptEl.dataset.committed || "") + interim;
      setStatus("busy", "listening…");
    }
    if (final) {
      if (matchVoiceCommand(final.trim())) {
        tryHandleAsVoiceCommand(final.trim());
        return;
      }
      const base = promptEl.dataset.committed || "";
      const next = (base + final).trim();
      promptEl.dataset.committed = next + " ";
      promptEl.value = next;
      interimTranscript = "";
    }
  };

  rec.onerror = (ev) => {
    setMicUI(false);
    const err = ev.error || "error";
    if (err === "not-allowed") {
      setStatus("", "mic denied");
      if (voiceHint)
        voiceHint.textContent = "Microphone permission denied. Allow mic access and retry.";
    } else if (err !== "aborted" && err !== "no-speech") {
      setStatus("", "mic: " + err);
    } else {
      setStatus("ok", "ready");
    }
  };

  rec.onend = () => {
    setMicUI(false);
    delete promptEl.dataset.committed;
    const text = (promptEl.value || "").trim();

    if (skipAutoSendOnce) {
      skipAutoSendOnce = false;
      promptEl.value = "";
      setStatus("ok", "ready");
      if (typeof rabbitShouldContinueListening === "function" && rabbitShouldContinueListening())
        setTimeout(() => startListening(), 250);
      return;
    }

    if (text && tryHandleAsVoiceCommand(text)) {
      return;
    }

    if (text && voiceAutoSend && voiceAutoSend.checked) {
      setStatus("ok", "ready");
      form.requestSubmit();
    } else if (text) {
      setStatus("ok", "ready");
      promptEl.focus();
      if (typeof rabbitShouldContinueListening === "function" && rabbitShouldContinueListening())
        setTimeout(() => startListening(), 400);
    } else {
      setStatus("ok", "ready");
      if (typeof rabbitShouldContinueListening === "function" && rabbitShouldContinueListening())
        setTimeout(() => startListening(), 300);
    }
  };

  return rec;
}

function startListening() {
  stopSpeaking();
  if (!recognition) recognition = initSpeechRecognition();
  if (!recognition) return;
  try {
    recognition.lang = (voiceLang && voiceLang.value) || "en-US";
    promptEl.dataset.committed = "";
    recognition.start();
  } catch (e) {
    console.warn(e);
  }
}

function stopListening() {
  if (recognition && listening) {
    try {
      recognition.stop();
    } catch {}
  }
  setMicUI(false);
}

function toggleMic() {
  if (listening) stopListening();
  else startListening();
}

if (micBtn) micBtn.addEventListener("click", toggleMic);

async function refreshInfo() {
  try {
    const r = await fetch("/api/info");
    const j = await r.json();
    document.getElementById("version").textContent = j.version || "engine";
    document.getElementById("model-name").textContent = j.model_name || "—";
    document.getElementById("s-layers").textContent = j.layers ?? "—";
    document.getElementById("s-hidden").textContent = j.hidden_dim ?? "—";
    document.getElementById("s-experts").textContent =
      j.experts != null ? `${j.experts} (top-${j.top_k ?? "?"})` : "—";
    document.getElementById("s-shared").textContent =
      j.n_shared_experts != null && j.n_shared_experts > 0
        ? String(j.n_shared_experts)
        : "0";
    const nh = j.n_heads;
    const nkv = j.n_kv_heads;
    const hd = j.head_dim;
    document.getElementById("s-gqa").textContent =
      nh || nkv || hd ? `${nh ?? "?"}/${nkv ?? "?"} × ${hd ?? "?"}` : "—";
    document.getElementById("s-quant").textContent = quantLabel(j.expert_quant);
    document.getElementById("s-dense").textContent = j.dense || "—";
    document.getElementById("s-map").textContent = j.mapped ? "mapped" : "—";
    const tok = j.tokenizer;
    document.getElementById("s-tok").textContent = tok
      ? String(tok).split(/[/\\]/).pop()
      : "byte";

    document.getElementById("m-budget").textContent =
      j.mem_budget_mb != null ? `${j.mem_budget_mb} MB` : "—";
    document.getElementById("m-est").textContent =
      j.mem_total_est_mb != null ? `${j.mem_total_est_mb} MB` : "—";
    document.getElementById("m-slots").textContent = j.cache_slots ?? "—";
    const statusEl = document.getElementById("m-status");
    statusEl.textContent = j.over_budget
      ? "over budget"
      : j.mem_budget_mb != null
      ? "ok"
      : "—";
    statusEl.className = j.over_budget ? "warn-text" : "";

    renderExperts(j.experts || 8, j.warm_experts || [], j.n_shared_experts || 0);

    const hits = j.cache_hits || 0;
    const misses = j.cache_misses || 0;
    const total = hits + misses;
    const rate = total ? (100 * hits) / total : 0;
    document.getElementById("c-hits").textContent = hits;
    document.getElementById("c-misses").textContent = misses;
    document.getElementById("c-inserts").textContent = j.cache_inserts || 0;
    document.getElementById("c-rate").textContent = total ? rate.toFixed(1) + "%" : "—";
    document.getElementById("c-bar").style.width = Math.min(100, rate) + "%";

    const met = j.metrics || {};
    document.getElementById("met-req").textContent = met.requests ?? 0;
    document.getElementById("met-err").textContent = met.errors ?? 0;
    document.getElementById("met-avg").textContent =
      met.avg_ms != null ? `${met.avg_ms} ms` : "—";
    document.getElementById("met-tok").textContent = met.tokens_est ?? 0;
  } catch (e) {
    console.warn(e);
  }
}

function clearChat() {
  if (abortCtrl) {
    abortCtrl.abort();
    abortCtrl = null;
  }
  stopListening();
  stopSpeaking();
  chatEl.querySelectorAll(".bubble").forEach((b) => b.remove());
  showEmpty();
  lastReply = "";
  history = [];
  latencyEl.textContent = "—";
  setStatus("ok", "ready");
  sendBtn.disabled = false;
  stopBtn.hidden = true;
  promptEl.focus();
}

async function copyLast() {
  if (!lastReply) {
    setStatus("", "nothing to copy");
    return;
  }
  try {
    await navigator.clipboard.writeText(lastReply);
    setStatus("ok", "copied");
    setTimeout(() => setStatus("ok", "ready"), 1200);
  } catch {
    setStatus("", "copy failed");
  }
}

function exportChat() {
  const payload = {
    exported_at: new Date().toISOString(),
    system: systemEl.value || "",
    messages: history,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `rabbit-chat-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
  setStatus("ok", "exported");
}

function importChatFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result);
      clearChat();
      if (data.system) systemEl.value = data.system;
      const msgs = data.messages || data.history || [];
      msgs.forEach((m) => {
        const role =
          m.role === "user" || m.role === "assistant" || m.role === "bot"
            ? m.role === "assistant"
              ? "bot"
              : m.role
            : "bot";
        addBubble(role === "user" ? "user" : "bot", m.content || m.text || "");
      });
      setStatus("ok", "imported");
    } catch (e) {
      setStatus("", "import failed");
      console.warn(e);
    }
  };
  reader.readAsText(file);
}

if (clearBtn) clearBtn.addEventListener("click", clearChat);
if (copyBtn) copyBtn.addEventListener("click", copyLast);
if (refreshBtn) refreshBtn.addEventListener("click", () => refreshInfo());
if (exportBtn) exportBtn.addEventListener("click", exportChat);
if (importBtn)
  importBtn.addEventListener("click", () => importFile && importFile.click());
if (importFile)
  importFile.addEventListener("change", () => {
    if (importFile.files && importFile.files[0]) importChatFile(importFile.files[0]);
    importFile.value = "";
  });

document.getElementById("chips").addEventListener("click", (ev) => {
  const t = ev.target.closest(".chip");
  if (!t) return;
  promptEl.value = t.getAttribute("data-prompt") || "";
  promptEl.focus();
});

if (stopBtn) {
  stopBtn.addEventListener("click", () => {
    if (abortCtrl) abortCtrl.abort();
    stopListening();
    stopSpeaking();
  });
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") {
    stopListening();
    promptEl.focus();
  }
  if (
    (ev.key === "m" || ev.key === "M") &&
    !ev.ctrlKey &&
    !ev.metaKey &&
    !ev.altKey &&
    document.activeElement !== promptEl &&
    document.activeElement !== systemEl
  ) {
    ev.preventDefault();
    toggleMic();
  }
});

function genParams() {
  const tokens = Number(document.getElementById("tokens").value) || 48;
  const temp = Number(document.getElementById("temp").value);
  const topP = Number(document.getElementById("top-p").value);
  const seedRaw = document.getElementById("seed").value;
  const seed = seedRaw === "" ? null : Number(seedRaw);
  return {
    tokens,
    temperature: Number.isFinite(temp) ? temp : 0.8,
    top_p: Number.isFinite(topP) ? topP : 1.0,
    seed,
    system: (systemEl.value || "").trim(),
  };
}

async function sendNonStream(message, params, signal) {
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, ...params }),
    signal,
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "request failed");
  return j.reply || "[empty]";
}

async function sendStream(message, params, signal, onToken) {
  const r = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, ...params }),
    signal,
  });
  if (!r.ok) {
    let err = "stream failed";
    try {
      const j = await r.json();
      err = j.error || err;
    } catch {}
    throw new Error(err);
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let full = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";
    for (const block of parts) {
      const line = block.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        const obj = JSON.parse(line.slice(6));
        if (obj.type === "token" && obj.text) {
          full += obj.text;
          onToken(full);
        } else if (obj.type === "error") {
          throw new Error(obj.error || "stream error");
        }
      } catch (e) {
        if (e.message && !e.message.includes("JSON")) throw e;
      }
    }
  }
  return full || "[empty]";
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  stopListening();
  stopSpeaking();
  const message = promptEl.value.trim();
  if (!message) return;

  if (tryHandleAsVoiceCommand(message)) return;

  addBubble("user", message);
  promptEl.value = "";
  sendBtn.disabled = true;
  stopBtn.hidden = false;
  setStatus("busy", "generating");

  const params = genParams();
  const useStream = streamEl && streamEl.checked;
  abortCtrl = new AbortController();
  const t0 = performance.now();

  let botBody = null;
  let botMeta = null;
  if (useStream) {
    const bubble = addBubble("bot", "…");
    botBody = bubble.body;
    botMeta = bubble.meta;
    history.pop();
  }

  try {
    let reply;
    if (useStream) {
      reply = await sendStream(message, params, abortCtrl.signal, (full) => {
        botBody.textContent = full;
        lastReply = full;
        chatEl.scrollTop = chatEl.scrollHeight;
      });
      botBody.textContent = reply;
      const ms = Math.round(performance.now() - t0);
      latencyEl.textContent = `${ms} ms`;
      if (typeof rabbitRecordSpeed === "function") rabbitRecordSpeed(ms, reply);
      const ts = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
      botMeta.textContent = `rabbit · ${ms} ms · ${ts}`;
      history.push({ role: "bot", content: reply, ts: Date.now(), ms });
      lastReply = reply;
    } else {
      reply = await sendNonStream(message, params, abortCtrl.signal);
      const ms = Math.round(performance.now() - t0);
      latencyEl.textContent = `${ms} ms`;
      if (typeof rabbitRecordSpeed === "function") rabbitRecordSpeed(ms, reply);
      addBubble("bot", reply, ms);
    }
    setStatus("ok", "ready");
    speakText(reply);
    await refreshInfo();
  } catch (e) {
    if (e.name === "AbortError") {
      if (botBody) botBody.textContent = (botBody.textContent || "") + " [stopped]";
      else addBubble("bot", "[stopped]");
      setStatus("", "stopped");
    } else {
      if (botBody) botBody.textContent = "[error] " + e.message;
      else addBubble("bot", "[error] " + e.message);
      setStatus("", "error");
    }
  } finally {
    abortCtrl = null;
    sendBtn.disabled = false;
    stopBtn.hidden = true;
    promptEl.focus();
    if (typeof rabbitShouldContinueListening === "function" && rabbitShouldContinueListening())
      setTimeout(() => startListening(), 500);
  }
});

recognition = initSpeechRecognition();
refreshInfo();
setInterval(refreshInfo, 8000);
setStatus("ok", "ready");
