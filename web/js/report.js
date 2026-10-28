// End-of-session report: the agent's report, then the delivery metrics (tiles and per-answer charts).
const Report = {
  render(m) {
    $("#report-status").textContent = "";
    if (m.submitted) {  // a session link: the report goes to whoever created the session
      $("#report-title").textContent = "All done";
      $("#report").innerHTML = "<p>Thank you! The session has been submitted. You can close this tab.</p>";
      return;
    }
    $("#report-title").textContent = "Interview report";
    $("#report").innerHTML = markdown(m.markdown || "");
    const d = m.delivery || {};
    const turns = d.per_turn || [];
    for (const id of ["#delivery-h", "#delivery-note"]) $(id).classList.toggle("hidden", !d.answers);
    $("#delivery-tiles").innerHTML = d.answers ? this.tiles(d) : "";
    $("#delivery-charts").innerHTML = "";
    if (turns.length) {
      const lat = turns.filter((t) => t.latency !== null && t.latency !== undefined);
      if (lat.length) this.columns("#delivery-charts", "Response time per answer (s)", lat.map((t) => ({ x: t.turn, y: t.latency })), (v) => v.toFixed(1) + " s");
      this.columns("#delivery-charts", "Words per answer", turns.map((t) => ({ x: t.turn, y: t.words })), (v) => String(Math.round(v)));
    }
    $("#report-folder").textContent = m.folder ? `Saved to ${m.folder}` : "";
  },

  tiles(d) {
    const tiles = [
      [d.words_per_minute, "words per minute of speech"],
      [d.response_time_s !== null && d.response_time_s !== undefined ? `${d.response_time_s.toFixed(1)} s` : "-", "typical response time"],
      [d.words_per_answer, "words per answer"],
      [d.long_pauses_per_min, "long pauses per minute"],
      [d.fillers_per_100_words, "fillers per 100 words"],
      [`${Math.round(d.talk_time_s / 6) / 10} min`, "talk time"],
    ];
    return tiles.map(([v, k]) => `<div class="stat"><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join("");
  },

  // single-series column chart: thin columns, 4px rounded tops, recessive grid, hover tooltip, table view
  columns(container, title, data, fmtv) {
    const id = "viz" + Math.random().toString(36).slice(2, 8);
    $(container).insertAdjacentHTML("beforeend", `<div class="viz" id="${id}"></div>`);
    const root = document.getElementById(id);
    const W = Math.max(240, root.clientWidth - 28), H = 150, pad = { l: 34, r: 8, t: 8, b: 22 };  // real pixels: no stretched text
    const max = Math.max(...data.map((d) => d.y), 0.0001);
    const nice = (() => { const p = Math.pow(10, Math.floor(Math.log10(max))); return Math.ceil(max / p) * p; })();
    const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
    const band = iw / data.length, bw = Math.min(24, band - 2);
    const y = (v) => pad.t + ih - (v / nice) * ih;
    const grid = [0, nice / 2, nice].map((g) =>
      `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(g)}" y2="${y(g)}" stroke="var(--grid)" stroke-width="1"/>` +
      `<text class="axis" x="${pad.l - 6}" y="${y(g) + 3}" text-anchor="end">${fmtv(g).replace(/\.0(?=\D|$)/, "")}</text>`).join("");
    const step = Math.ceil(data.length / 12);
    const bars = data.map((d, i) => {
      const x = pad.l + i * band + (band - bw) / 2, top = y(d.y), h = Math.max(0, pad.t + ih - top), r = Math.min(4, h, bw / 2);
      const path = h > 0 ? `M${x},${pad.t + ih} V${top + r} Q${x},${top} ${x + r},${top} H${x + bw - r} Q${x + bw},${top} ${x + bw},${top + r} V${pad.t + ih} Z` : "";
      const label = i % step === 0 ? `<text class="axis" x="${x + bw / 2}" y="${H - 6}" text-anchor="middle">${d.x}</text>` : "";
      return `<g class="bar" data-i="${i}"><rect x="${pad.l + i * band}" y="${pad.t}" width="${band}" height="${ih}" fill="transparent"/>` +
        `<path d="${path}" fill="var(--series-1)"/>${label}</g>`;
    }).join("");
    root.innerHTML = `<h4>${esc(title)}</h4>
        <svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(title)}">${grid}${bars}</svg>
        <div class="tip"></div>
        <details><summary>Table</summary><table><tr><th>Turn</th><th>Value</th></tr>
          ${data.map((d) => `<tr><td>${d.x}</td><td>${esc(fmtv(d.y))}</td></tr>`).join("")}</table></details>`;
    const tip = root.querySelector(".tip");
    root.querySelectorAll(".bar").forEach((g) => {
      g.addEventListener("mousemove", (e) => {
        const d = data[Number(g.dataset.i)];
        tip.textContent = `Turn ${d.x}: ${fmtv(d.y)}`;
        const box = root.getBoundingClientRect();
        tip.style.left = `${e.clientX - box.left + 10}px`; tip.style.top = `${e.clientY - box.top - 28}px`;
        tip.style.display = "block";
      });
      g.addEventListener("mouseleave", () => (tip.style.display = "none"));
    });
  },
};
