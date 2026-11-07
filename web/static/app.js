/* Rabbit dashboard — load main bundle */
(function () {
  var s = document.createElement("script");
  s.src = "/static/app-p0.js?v=6";
  s.onload = function () {
    if (typeof rabbitOnAppReady === "function") {
      try {
        rabbitOnAppReady();
      } catch (e) {
        console.warn(e);
      }
    }
  };
  s.onerror = function () {
    var p = document.getElementById("status-pill");
    if (p) p.textContent = "script load error";
  };
  document.body.appendChild(s);
})();
