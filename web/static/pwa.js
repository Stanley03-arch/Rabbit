/* Register service worker + sync theme-color for mobile chrome */
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
})();
