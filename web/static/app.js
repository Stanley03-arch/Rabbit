/* Rabbit dashboard — load multi-part bundle */
(async function () {
  const parts = ["app-p0.js", "app-p1.js", "app-p2.js", "app-p3.js"];
  let code = "";
  for (const name of parts) {
    const r = await fetch("/static/" + name + "?v=2");
    if (!r.ok) {
      console.error("failed to load", name, r.status);
      document.getElementById("status-pill") && (document.getElementById("status-pill").textContent = "load error");
      return;
    }
    code += await r.text();
  }
  const s = document.createElement("script");
  s.textContent = code;
  document.body.appendChild(s);
})();
