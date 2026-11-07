const chatEl = document.getElementById("chat");
const form = document.getElementById("form");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const stopBtn = document.getElementById("stop");
const statusPill = document.getElementById("status-pill");
const clearBtn = document.getElementById("clear-chat");
const copyBtn = document.getElementById("copy-last");
const refreshBtn = document.getElementById("refresh-stats");
const emptyState = document.getElementById("empty-state");
const latencyEl = document.getElementById("latency");

let lastReply = "";
let abortCtrl = null;

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
  const ts = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  let label = role === "user" ? "you" : "rabbit";
  if (ms != null && role === "bot") label += ` · ${ms} ms`;
  meta.textContent = `${label} · ${ts}`;
  const body = document.createElement("div");
  body.textContent = text;
  div.appendChild(meta);
  div.appendChild(body);
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  if (role === "bot") lastReply = text;
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
  } catch (e) {
    console.warn(e);
  }
}

function clearChat() {
  if (abortCtrl) {
    abortCtrl.abort();
    abortCtrl = null;
  }
  const bubbles = chatEl.querySelectorAll(".bubble");
  bubbles.forEach((b) => b.remove());
  showEmpty();
  lastReply = "";
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

if (clearBtn) clearBtn.addEventListener("click", clearChat);
if (copyBtn) copyBtn.addEventListener("click", copyLast);
if (refreshBtn) refreshBtn.addEventListener("click", () => refreshInfo());

document.getElementById("chips").addEventListener("click", (ev) => {
  const t = ev.target.closest(".chip");
  if (!t) return;
  promptEl.value = t.getAttribute("data-prompt") || "";
  promptEl.focus();
});

if (stopBtn) {
  stopBtn.addEventListener("click", () => {
    if (abortCtrl) abortCtrl.abort();
  });
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") promptEl.focus();
});

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const message = promptEl.value.trim();
  if (!message) return;

  addBubble("user", message);
  promptEl.value = "";
  sendBtn.disabled = true;
  stopBtn.hidden = false;
  setStatus("busy", "generating");

  const tokens = Number(document.getElementById("tokens").value) || 48;
  const temp = Number(document.getElementById("temp").value);
  const topP = Number(document.getElementById("top-p").value);
  const seedRaw = document.getElementById("seed").value;
  const seed = seedRaw === "" ? null : Number(seedRaw);

  abortCtrl = new AbortController();
  const t0 = performance.now();

  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        tokens,
        temperature: Number.isFinite(temp) ? temp : 0.8,
        top_p: Number.isFinite(topP) ? topP : 1.0,
        seed,
      }),
      signal: abortCtrl.signal,
    });
    const j = await r.json();
    const ms = Math.round(performance.now() - t0);
    latencyEl.textContent = `${ms} ms`;
    if (!r.ok) throw new Error(j.error || "request failed");
    addBubble("bot", j.reply || "[empty]", ms);
    setStatus("ok", "ready");
    await refreshInfo();
  } catch (e) {
    if (e.name === "AbortError") {
      addBubble("bot", "[stopped]");
      setStatus("", "stopped");
    } else {
      addBubble("bot", "[error] " + e.message);
      setStatus("", "error");
    }
  } finally {
    abortCtrl = null;
    sendBtn.disabled = false;
    stopBtn.hidden = true;
    promptEl.focus();
  }
});

refreshInfo();
setInterval(refreshInfo, 8000);
setStatus("ok", "ready");
