/* Prompt templates: insert system + starter user message */
(function () {
  const KEY = "rabbit-custom-templates";
  const promptEl = document.getElementById("prompt");
  const systemEl = document.getElementById("system");
  const chips = document.getElementById("chips");
  if (!promptEl) return;

  const BUILTIN = [
    {
      id: "debug",
      label: "Debug MoE",
      system: "You are Rabbit, a local MoE engine assistant. Prefer precise, technical answers about routing, experts, and memory.",
      prompt: "Summarize current expert-routing behavior in one paragraph.",
    },
    {
      id: "teach",
      label: "Teach me",
      system: "Explain concepts simply, as if teaching a beginner. Use short paragraphs.",
      prompt: "Explain Mixture of Experts like I'm new to ML.",
    },
    {
      id: "brief",
      label: "Brief mode",
      system: "Answer in one short paragraph. Be direct. No preamble.",
      prompt: "What is the main benefit of local MoE inference?",
    },
    {
      id: "code",
      label: "Code helper",
      system: "You are a careful coding assistant. Prefer correct, minimal code with brief comments.",
      prompt: "Write a tiny Python helper that times a function call.",
    },
  ];

  function loadCustom() {
    try {
      const raw = localStorage.getItem(KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function saveCustom(arr) {
    try {
      localStorage.setItem(KEY, JSON.stringify(arr.slice(0, 20)));
    } catch (e) {}
  }

  function allTemplates() {
    return BUILTIN.concat(loadCustom());
  }

  function applyTemplate(t) {
    if (!t) return;
    if (systemEl && t.system != null) {
      systemEl.value = t.system;
      systemEl.dispatchEvent(new Event("input", { bubbles: true }));
      const preset = document.getElementById("system-preset");
      if (preset) preset.value = "";
    }
    if (t.prompt != null) {
      promptEl.value = t.prompt;
      promptEl.focus();
      promptEl.dispatchEvent(new Event("input", { bubbles: true }));
    }
    if (window.rabbitToast) rabbitToast("Template: " + (t.label || t.id), "ok");
  }

  function renderChips() {
    if (!chips) return;
    // remove old template chips
    chips.querySelectorAll(".chip-template").forEach(function (c) {
      c.remove();
    });
    allTemplates().forEach(function (t) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip chip-template";
      btn.textContent = t.label || t.id;
      btn.title = (t.system || "").slice(0, 80);
      btn.addEventListener("click", function () {
        applyTemplate(t);
      });
      chips.appendChild(btn);
    });
  }

  function saveCurrentAsTemplate() {
    const label = window.prompt("Template name", "My template");
    if (!label) return;
    const t = {
      id: "c-" + Date.now(),
      label: label.trim().slice(0, 40),
      system: systemEl ? systemEl.value : "",
      prompt: promptEl.value || "",
    };
    const list = loadCustom();
    list.unshift(t);
    saveCustom(list);
    renderChips();
    if (window.rabbitToast) rabbitToast("Saved template", "ok");
  }

  // Save template button near save-prompt
  const tools = document.querySelector(".composer-tools");
  if (tools && !document.getElementById("save-template")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "save-template";
    btn.className = "linkish";
    btn.title = "Save system + prompt as template";
    btn.textContent = "Save template";
    btn.addEventListener("click", saveCurrentAsTemplate);
    tools.appendChild(btn);
  }

  renderChips();

  window.rabbitTemplates = {
    apply: applyTemplate,
    list: allTemplates,
    saveCurrent: saveCurrentAsTemplate,
  };
})();
