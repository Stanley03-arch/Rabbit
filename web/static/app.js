/* Rabbit dashboard — load main bundle (p0a + p0b) */
(async function () {
  try {
    const names = ["app-p0a.js", "app-p0b.js"];
    let code = "";
    for (const name of names) {
      const r = await fetch("/static/" + name + "?v=4");
      if (!r.ok) throw new Error(name + " " + r.status);
      code += await r.text();
    }
    const s = document.createElement("script");
    s.textContent = code;
    document.body.appendChild(s);
  } catch (e) {
    console.error(e);
    var p = document.getElementById("status-pill");
    if (p) p.textContent = "script load error";
  }
})();
