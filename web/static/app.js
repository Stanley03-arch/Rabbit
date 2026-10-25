const chatEl = document.getElementById("chat");
const form = document.getElementById("form");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const statusPill = document.getElementById("status-pill");

function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = role === "user" ? "you" : "rabbit";
  const body = document.createElement("div");
  body.textContent = text;
  div.appendChild(meta);
  div.appendChild(body);
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setStatus(mode, label) {
  statusPill.className = "status-pill " + (mode || "");
  statusPill.textContent = label;
}

function renderExperts(n, warmIds) {
  const grid = document.getElementById("expert-grid");
  grid.innerHTML = "";
  const count = Math.min(Math.max(n || 8, 1), 32);
  const warm = new Set(warmIds || []);
  for (let i = 0; i < count; i++) {
    const cell = document.createElement("div");
    cell.className = "expert-cell";
    if (warm.has(i)) cell.classList.add(i % 3 === 0 ? "hot" : "warm");
    cell.textContent = String(i);
    grid.appendChild(cell);
  }
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
    document.getElementById("s-quant").textContent =
      j.expert_quant ? `${j.expert_quant}-bit` : "—";
    document.getElementById("s-dense").textContent = j.dense || "—";
    document.getElementById("s-map").textContent = j.mapped ? "mapped" : "—";
    renderExperts(j.experts || 8, j.warm_experts || []);

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

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const message = promptEl.value.trim();
  if (!message) return;

  addBubble("user", message);
  promptEl.value = "";
  sendBtn.disabled = true;
  setStatus("busy", "generating");

  const tokens = Number(document.getElementById("tokens").value) || 48;
  const temp = Number(document.getElementById("temp").value) || 0.8;

  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, tokens, temperature: temp }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "request failed");
    addBubble("bot", j.reply || "[empty]");
    setStatus("ok", "ready");
    await refreshInfo();
  } catch (e) {
    addBubble("bot", "[error] " + e.message);
    setStatus("", "error");
  } finally {
    sendBtn.disabled = false;
    promptEl.focus();
  }
});

refreshInfo();
setInterval(refreshInfo, 8000);
setStatus("ok", "ready");
