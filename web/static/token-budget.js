/* Composer token budget warning vs Tokens setting */
(function () {
  const promptEl = document.getElementById("prompt");
  const tokensEl = document.getElementById("tokens");
  const countEl = document.getElementById("composer-count");
  if (!promptEl || !countEl) return;

  function estimateTokens(text) {
    // rough: ~4 chars / token
    return Math.max(0, Math.ceil(String(text || "").length / 4));
  }

  function update() {
    const est = estimateTokens(promptEl.value);
    const budget = tokensEl ? parseInt(tokensEl.value, 10) || 48 : 48;
    // keep existing count text pattern if present
    const chars = (promptEl.value || "").length;
    countEl.textContent = chars + " chars · ~" + est + " tok";
    countEl.classList.remove("budget-warn", "budget-over", "warn", "danger");
    // warn when prompt alone approaches generation budget (context pressure signal)
    if (est >= budget) {
      countEl.classList.add("budget-over");
      countEl.title = "Prompt ~tokens ≥ generation Tokens setting";
    } else if (est >= budget * 0.7) {
      countEl.classList.add("budget-warn");
      countEl.title = "Prompt is large vs Tokens setting";
    } else {
      countEl.title = "";
    }
  }

  promptEl.addEventListener("input", update);
  if (tokensEl) tokensEl.addEventListener("input", update);
  tokensEl && tokensEl.addEventListener("change", update);
  update();
})();
