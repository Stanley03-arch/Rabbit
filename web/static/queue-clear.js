/* Clear message queue button + sticky mobile composer helper class */
(function () {
  function ensureClearBtn() {
    const bar = document.getElementById("msg-queue-bar");
    if (!bar) return;
    if (bar.querySelector(".msg-queue-clear")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "msg-queue-clear";
    btn.textContent = "Clear queue";
    btn.title = "Drop all queued messages";
    btn.addEventListener("click", function () {
      if (window.rabbitMsgQueue && typeof rabbitMsgQueue.clear === "function") {
        rabbitMsgQueue.clear();
        if (window.rabbitToast) rabbitToast("Queue cleared", "ok");
      }
    });
    bar.appendChild(btn);
  }

  // Observe queue bar visibility / creation
  const obs = new MutationObserver(ensureClearBtn);
  obs.observe(document.body, { childList: true, subtree: true });
  ensureClearBtn();
  setInterval(ensureClearBtn, 1500);

  // Sticky composer polish on narrow screens
  document.documentElement.classList.add("sticky-composer");
})();
