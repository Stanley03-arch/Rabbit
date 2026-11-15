/* Connection pulse on #conn-status + sidebar settings filter */
(function () {
  const conn = document.getElementById("conn-status");
  if (conn) {
    function paint() {
      const t = (conn.textContent || "").toLowerCase();
      conn.classList.remove("conn-ok", "conn-bad", "conn-wait");
      if (t.indexOf("online") >= 0 || t.indexOf("ok") >= 0 || t.indexOf("connected") >= 0) {
        conn.classList.add("conn-ok");
      } else if (t.indexOf("off") >= 0 || t.indexOf("error") >= 0 || t.indexOf("fail") >= 0) {
        conn.classList.add("conn-bad");
      } else {
        conn.classList.add("conn-wait");
      }
    }
    new MutationObserver(paint).observe(conn, {
      childList: true,
      characterData: true,
      subtree: true,
    });
    paint();
  }

  // Sidebar settings filter
  const sidebar = document.querySelector(".sidebar");
  const controls = document.querySelector(".panel.controls");
  if (sidebar && controls && !document.getElementById("settings-filter")) {
    const wrap = document.createElement("div");
    wrap.className = "settings-filter-wrap";
    wrap.innerHTML =
      '<input type="search" id="settings-filter" class="settings-filter" placeholder="Filter settings…" autocomplete="off" />';
    controls.insertBefore(wrap, controls.children[1] || null);

    const input = document.getElementById("settings-filter");
    input.addEventListener("input", function () {
      const q = (input.value || "").trim().toLowerCase();
      controls.querySelectorAll("label, .btn-row, .kv, .check").forEach(function (el) {
        if (el.closest(".settings-filter-wrap")) return;
        if (!q) {
          el.style.display = "";
          return;
        }
        const text = (el.textContent || "").toLowerCase();
        el.style.display = text.indexOf(q) >= 0 ? "" : "none";
      });
    });
  }

  // Search: Enter jumps to next .search-match / .msg-hit
  const search = document.getElementById("chat-search");
  const chatEl = document.getElementById("chat");
  if (search && chatEl) {
    let hitIdx = -1;
    search.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter") return;
      ev.preventDefault();
      const hits = chatEl.querySelectorAll(".bubble.search-match, .bubble:has(mark.msg-hit)");
      if (!hits.length) {
        // fallback: any mark.msg-hit parent bubble
        const marks = chatEl.querySelectorAll("mark.msg-hit");
        const bubbles = [];
        marks.forEach(function (m) {
          const b = m.closest(".bubble");
          if (b && bubbles.indexOf(b) < 0) bubbles.push(b);
        });
        if (!bubbles.length) return;
        hitIdx = (hitIdx + 1) % bubbles.length;
        bubbles[hitIdx].scrollIntoView({ block: "center", behavior: "smooth" });
        bubbles[hitIdx].classList.add("msg-focus");
        return;
      }
      hitIdx = (hitIdx + 1) % hits.length;
      hits[hitIdx].scrollIntoView({ block: "center", behavior: "smooth" });
      chatEl.querySelectorAll(".bubble.msg-focus").forEach(function (b) {
        b.classList.remove("msg-focus");
      });
      hits[hitIdx].classList.add("msg-focus");
    });
    search.addEventListener("input", function () {
      hitIdx = -1;
    });
  }
})();
