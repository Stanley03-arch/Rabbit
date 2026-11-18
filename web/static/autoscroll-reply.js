/* Stronger auto-scroll when a bot reply finishes */
(function () {
  const chatEl = document.getElementById("chat");
  const auto = document.getElementById("auto-scroll");
  if (!chatEl) return;

  let lastBotCount = 0;

  function botCount() {
    return chatEl.querySelectorAll(".bubble.bot:not(.streaming)").length;
  }

  function maybeScroll() {
    if (auto && !auto.checked) return;
    const n = botCount();
    if (n > lastBotCount) {
      const last = chatEl.querySelector(".bubble.bot:not(.streaming):last-of-type") ||
        chatEl.querySelector(".bubble:last-of-type");
      if (last && last.scrollIntoView) {
        last.scrollIntoView({ block: "end", behavior: "smooth" });
      }
    }
    lastBotCount = n;
  }

  new MutationObserver(function () {
    setTimeout(maybeScroll, 50);
  }).observe(chatEl, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });
  lastBotCount = botCount();
})();
