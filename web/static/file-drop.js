/* Drag-and-drop text/markdown/json into the composer */
(function () {
  const promptEl = document.getElementById("prompt");
  const chatEl = document.getElementById("chat");
  const formEl = document.getElementById("form");
  if (!promptEl) return;

  const targets = [chatEl, formEl, promptEl, document.body].filter(Boolean);

  function isFileDrag(ev) {
    const types = ev.dataTransfer && ev.dataTransfer.types;
    if (!types) return false;
    return Array.prototype.indexOf.call(types, "Files") >= 0;
  }

  function setDrag(on) {
    document.body.classList.toggle("file-drag", on);
  }

  let dragDepth = 0;
  document.addEventListener("dragenter", function (ev) {
    if (!isFileDrag(ev)) return;
    dragDepth++;
    setDrag(true);
  });
  document.addEventListener("dragleave", function (ev) {
    if (!isFileDrag(ev)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) setDrag(false);
  });
  document.addEventListener("dragover", function (ev) {
    if (!isFileDrag(ev)) return;
    ev.preventDefault();
    ev.dataTransfer.dropEffect = "copy";
  });

  function readFile(file) {
    return new Promise(function (resolve, reject) {
      const reader = new FileReader();
      reader.onload = function () {
        resolve(String(reader.result || ""));
      };
      reader.onerror = reject;
      reader.readAsText(file);
    });
  }

  document.addEventListener("drop", function (ev) {
    if (!isFileDrag(ev)) return;
    ev.preventDefault();
    dragDepth = 0;
    setDrag(false);
    const files = ev.dataTransfer && ev.dataTransfer.files;
    if (!files || !files.length) return;

    const allowed = [];
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      const name = (f.name || "").toLowerCase();
      const ok =
        /\.(txt|md|markdown|json|csv|log|py|js|ts|c|h|cpp|rs|go|toml|yaml|yml)$/.test(name) ||
        (f.type && f.type.indexOf("text") === 0) ||
        f.type === "application/json";
      if (ok && f.size < 2 * 1024 * 1024) allowed.push(f);
    }
    if (!allowed.length) {
      if (window.rabbitToast) rabbitToast("Drop text-like files under 2MB", "err");
      return;
    }

    Promise.all(allowed.map(readFile)).then(function (texts) {
      const blocks = allowed.map(function (f, i) {
        return "```" + (f.name || "file") + "\n" + texts[i] + "\n```";
      });
      const chunk = blocks.join("\n\n");
      const cur = promptEl.value.trim();
      promptEl.value = cur ? cur + "\n\n" + chunk : chunk;
      promptEl.focus();
      promptEl.dispatchEvent(new Event("input", { bubbles: true }));
      if (window.rabbitToast) {
        rabbitToast("Added " + allowed.length + " file(s) to composer", "ok");
      }
    });
  });
})();
