/* Right-click session tab menu: pin, rename, jump pinned, export */
(function () {
  const tabsHost = document.getElementById("session-tabs");
  if (!tabsHost) return;

  let menu = document.getElementById("session-ctx-menu");
  if (!menu) {
    menu = document.createElement("div");
    menu.id = "session-ctx-menu";
    menu.className = "session-ctx-menu";
    menu.hidden = true;
    menu.innerHTML =
      '<button type="button" data-act="rename">Rename</button>' +
      '<button type="button" data-act="pin">Pin / unpin</button>' +
      '<button type="button" data-act="jump-pin">Jump to first pinned</button>' +
      '<button type="button" data-act="export">Export this chat</button>' +
      '<button type="button" data-act="dup">Duplicate</button>';
    document.body.appendChild(menu);
  }

  let targetTab = null;

  function hide() {
    menu.hidden = true;
    targetTab = null;
  }

  function show(x, y, tab) {
    targetTab = tab;
    menu.hidden = false;
    const pad = 8;
    const mw = menu.offsetWidth || 180;
    const mh = menu.offsetHeight || 160;
    let left = x;
    let top = y;
    if (left + mw > window.innerWidth - pad) left = window.innerWidth - mw - pad;
    if (top + mh > window.innerHeight - pad) top = window.innerHeight - mh - pad;
    menu.style.left = left + "px";
    menu.style.top = top + "px";
  }

  tabsHost.addEventListener("contextmenu", function (ev) {
    const tab = ev.target.closest(".session-tab");
    if (!tab) return;
    ev.preventDefault();
    show(ev.clientX, ev.clientY, tab);
  });

  document.addEventListener("click", function (ev) {
    if (!menu.hidden && !menu.contains(ev.target)) hide();
  });
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape") hide();
  });

  function jumpFirstPinned() {
    const pinned = tabsHost.querySelector(".session-tab.session-pinned");
    if (pinned) {
      pinned.click();
      if (window.rabbitToast) rabbitToast("Jumped to pinned session", "ok");
    } else if (window.rabbitToast) {
      rabbitToast("No pinned sessions", "ok");
    }
  }

  menu.addEventListener("click", function (ev) {
    const btn = ev.target.closest("button[data-act]");
    if (!btn) return;
    const act = btn.dataset.act;
    const tab = targetTab;
    hide();
    if (!tab && act !== "jump-pin") return;

    if (act === "rename" && tab) {
      tab.dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    }
    if (act === "pin") {
      if (tab) tab.click();
      setTimeout(function () {
        if (window.rabbitSessionUx) rabbitSessionUx.togglePinActive();
      }, 30);
    }
    if (act === "jump-pin") jumpFirstPinned();
    if (act === "export") {
      if (tab) tab.click();
      setTimeout(function () {
        const b = document.getElementById("export-chat");
        if (b) b.click();
      }, 40);
    }
    if (act === "dup") {
      if (tab) tab.click();
      setTimeout(function () {
        if (window.rabbitSessionsExtra) rabbitSessionsExtra.duplicate();
      }, 40);
    }
  });

  // Ctrl+Shift+J jump to pinned
  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.shiftKey && (ev.key === "j" || ev.key === "J")) {
      const tag = (ev.target && ev.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      ev.preventDefault();
      jumpFirstPinned();
    }
  });

  window.rabbitSessionMenu = { jumpFirstPinned: jumpFirstPinned };
})();
