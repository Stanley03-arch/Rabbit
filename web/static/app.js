const chatEl = document.getElementById("chat");
const form = document.getElementById("form");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const statusPill = document.getElementById("status-pill");

let lastInfo = {};

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

function renderTiers(j) {
  const row = document.getElementById("tier-row");
  if (!row) return;
  row.querySelector('[data-t="dense"]').classList.toggle("on", true);
  row.querySelector('[data-t="lru"]').classList.toggle("on", (j.cache_slots || 0) > 0);
  row.querySelector('[data-t="mmap"]').classList.toggle("on", !!j.mapped);
  row.querySelector('[data-t="lru"]').classList.toggle(
    "warn",
    !!(j.over_budget)
  );
}

function renderAtlas(j) {
  const layers = Math.max(1, Number(j.layers) || 1);
  const epl = Math.max(1, Number(j.experts) || 8);
  const warm = new Set(j.warm_experts || []);
  const maxShowE = Math.min(epl, 24);
  const maxShowL = Math.min(layers, 16);

  const table = document.getElementById("atlas-table");
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  const hr = document.createElement("tr");
  const corner = document.createElement("th");
  corner.textContent = "L\\E";
  hr.appendChild(corner);
  for (let e = 0; e < maxShowE; e++) {
    const th = document.createElement("th");
    th.textContent = String(e);
    hr.appendChild(th);
  }
  if (epl > maxShowE) {
    const th = document.createElement("th");
    th.textContent = "…";
    hr.appendChild(th);
  }
  thead.appendChild(hr);

  for (let L = 0; L < maxShowL; L++) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = String(L);
    tr.appendChild(th);
    for (let e = 0; e < maxShowE; e++) {
      const td = document.createElement("td");
      const localWarm = warm.has(e);
      td.className = "atlas-cell" + (localWarm ? (e % 3 === 0 ? " hot" : " warm") : "");
      td.title = `layer ${L} expert ${e}`;
      tr.appendChild(td);
    }
    if (epl > maxShowE) {
      const td = document.createElement("td");
      td.className = "atlas-cell dim";
      td.textContent = "·";
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  if (layers > maxShowL) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = "…";
    tr.appendChild(th);
    tbody.appendChild(tr);
  }

  document.getElementById("atlas-meta").textContent =
    `${layers} layers × ${epl} experts/layer · showing ${maxShowL}×${maxShowE}` +
    (j.tiers ? ` · tiers: ${(j.tiers || []).join(" → ")}` : "");
}

async function refreshInfo() {
  try {
    const r = await fetch("/api/info");
    const j = await r.json();
    lastInfo = j;
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

    document.getElementById("m-budget").textContent =
      j.mem_budget_mb != null ? `${j.mem_budget_mb} MB` : "—";
    document.getElementById("m-est").textContent =
      j.mem_total_est_mb != null ? `${j.mem_total_est_mb} MB` : "—";
    document.getElementById("m-slots").textContent = j.cache_slots ?? "—";
    document.getElementById("m-status").textContent = j.over_budget
      ? "over budget"
      : j.mem_budget_mb != null
      ? "ok"
      : "—";

    renderTiers(j);
    renderExperts(j.experts || 8, j.warm_experts || []);
    renderAtlas(j);

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

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const view = btn.dataset.view;
    document.getElementById("view-chat").classList.toggle("hidden", view !== "chat");
    document.getElementById("view-atlas").classList.toggle("hidden", view !== "atlas");
    document.getElementById("view-title").textContent =
      view === "atlas" ? "Expert Atlas" : "Chat";
    document.getElementById("view-sub").textContent =
      view === "atlas"
        ? "Layer × expert map · warm = recent routing / cache"
        : "Dense in RAM · experts streamed from disk";
  });
});

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
