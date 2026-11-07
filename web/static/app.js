/* Rabbit dashboard — load main bundle */
(function () {
  var s = document.createElement("script");
  s.src = "/static/app-p0.js?v=3";
  s.onerror = function () {
    var p = document.getElementById("status-pill");
    if (p) p.textContent = "script load error";
  };
  document.body.appendChild(s);
})();
