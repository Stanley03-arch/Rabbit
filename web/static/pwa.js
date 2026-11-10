/* Register service worker + theme-color + install + offline */
(function () {
  function setThemeColor() {
    const dark = document.documentElement.getAttribute("data-theme") !== "light";
    const color = dark ? "#0c0f14" : "#f4f6fa";
    let meta = document.querySelector('meta[name="theme-color"]');
    if (!meta) {
      meta = document.createElement("meta");
      meta.name = "theme-color";
      document.head.appendChild(meta);
    }
    meta.content = color;
  }
  setThemeColor();
  const obs = new MutationObserver(setThemeColor);
  obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", function () {
      navigator.serviceWorker.register("/sw.js").catch(function (e) {
        console.warn("SW register failed", e);
      });
    });
  }

  const banner = document.getElementById("offline-banner");
  function syncOnline() {
    if (!banner) return;
    banner.hidden = navigator.onLine;
  }
  window.addEventListener("online", function () {
    syncOnline();
    if (window.rabbitToast) rabbitToast("Back online", "ok");
  });
  window.addEventListener("offline", function () {
    syncOnline();
    if (window.rabbitToast) rabbitToast("You are offline", "warn");
  });
  syncOnline();

  let deferredPrompt = null;
  const installBtn = document.getElementById("pwa-install");
  window.addEventListener("beforeinstallprompt", function (e) {
    e.preventDefault();
    deferredPrompt = e;
    if (installBtn) installBtn.hidden = false;
  });
  window.addEventListener("appinstalled", function () {
    deferredPrompt = null;
    if (installBtn) installBtn.hidden = true;
    if (window.rabbitToast) rabbitToast("Rabbit installed", "ok");
  });
  if (installBtn) {
    installBtn.addEventListener("click", async function () {
      if (!deferredPrompt) {
        if (window.rabbitToast) {
          rabbitToast("Use browser menu \u2192 Add to Home Screen", "ok");
        }
        return;
      }
      deferredPrompt.prompt();
      try {
        await deferredPrompt.userChoice;
      } catch (e) {}
      deferredPrompt = null;
      installBtn.hidden = true;
    });
  }

  try {
    const standalone =
      window.matchMedia("(display-mode: standalone)").matches ||
      window.navigator.standalone === true;
    if (standalone) {
      document.documentElement.classList.add("pwa-standalone");
      if (installBtn) installBtn.hidden = true;
    }
  } catch (e) {}
})();
