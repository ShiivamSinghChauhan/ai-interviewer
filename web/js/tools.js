// History, the repo report library and the YAML settings editor.
const Tools = {
  async history() {
    show("history");
    const rows = await fetch("/api/sessions").then((r) => r.json());
    const name = { interview: "🎯 Interview", conversation: "💬 Conversation", phone: "📞 Phone call" };
    $("#history-table").innerHTML = "<tr><th>When</th><th>Mode</th><th>With / about</th><th>Minutes</th><th>Stammered syllables</th><th>Nervousness</th><th></th></tr>" +
      (rows.length ? rows.map((r) => `<tr><td>${esc(r.id.replace("_", " ").replace(/-(\d\d)-(\d\d)$/, ":$1"))}</td>
        <td>${name[r.mode] || esc(r.mode)}</td><td>${esc(r.title)}</td><td>${r.minutes ?? "-"}</td>
        <td>${r.stammer ?? "-"}${r.stammer !== null && r.stammer !== undefined ? "%" : ""}</td><td>${r.nervous ?? "-"}</td>
        <td>${r.has_report ? `<button class="ghost small-btn" data-open="${esc(r.mode)}/${esc(r.id)}">Open report</button>` : ""}</td></tr>`).join("")
        : `<tr><td colspan="7" class="muted">No sessions yet.</td></tr>`);
    $$("#history-table [data-open]").forEach((b) => (b.onclick = () => this.openReport(...b.dataset.open.split("/"))));
  },
  async openReport(mode, id) {
    const d = await fetch(`/api/sessions/${mode}/${id}`).then((r) => r.json());
    S.mode = mode;
    show("report");
    Report.render(d);
  },

  async repos() {
    show("repos");
    if (S.meta.public) $("#repo-src").placeholder = "https://github.com/you/project";
    await this.listRepos();
    $("#repo-analyze").onclick = () => this.analyze(false);
    $("#repo-force").onclick = () => this.analyze(true);
  },
  async listRepos() {
    const rows = await fetch("/api/interview/repos").then((r) => r.json());
    $("#repo-table").innerHTML = "<tr><th>Repository</th><th>Commit</th><th>Analysed</th><th>Summary</th><th></th></tr>" +
      (rows.length ? rows.map((r) => `<tr><td>${esc(r.source)}</td><td><code>${esc(r.sha.slice(0, 10))}</code></td>
        <td>${new Date(r.created * 1000).toLocaleDateString()}</td><td>${esc(r.summary)}</td>
        <td><button class="ghost small-btn" data-view="${esc(r.source)}">View</button>
            ${S.meta.public ? "" : `<button class="ghost small-btn" data-del="${esc(r.source)}">Delete</button>`}</td></tr>`).join("")
        : `<tr><td colspan="5" class="muted">No saved reports yet.</td></tr>`);
    $$("#repo-table [data-view]").forEach((b) => (b.onclick = async () => {
      const rep = await fetch("/api/interview/repos/report?source=" + encodeURIComponent(b.dataset.view)).then((r) => r.json());
      $("#repo-view").textContent = JSON.stringify(rep, null, 2); $("#repo-view").classList.remove("hidden");
    }));
    $$("#repo-table [data-del]").forEach((b) => (b.onclick = async () => {
      if (!confirm("Delete this saved report?")) return;
      await fetch("/api/interview/repos?source=" + encodeURIComponent(b.dataset.del), { method: "DELETE" });
      this.listRepos();
    }));
  },
  async analyze(force) {
    const source = $("#repo-src").value.trim();
    if (!source) return;
    const r = await fetch("/api/interview/repos/analyze", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, force }) });
    if (!r.ok) { $("#repo-job").textContent = (await r.json()).detail || "Couldn't start the analysis."; return; }
    const { job } = await r.json();
    const poll = async () => {
      const j = await fetch("/api/interview/repos/job/" + job).then((r) => r.json());
      $("#repo-job").textContent = j.log.join("\n") + (j.error ? "\nFailed: " + j.error : "");
      if (j.status === "running") setTimeout(poll, 1500); else this.listRepos();
    };
    poll();
  },

  async settings() {
    show("settings");
    if (S.meta.public) {
      $("#cfg-note").textContent = "This is the real configuration behind the demo: every prompt, preset and scenario. "
        + "It's read-only here; run the app locally to edit it.";
      $("#cfg-text").readOnly = true;
      $("#cfg-save").classList.add("hidden");
    }
    const load = async () => {
      $("#cfg-text").value = await fetch("/api/config/" + $("#cfg-file").value).then((r) => r.text());
      $("#cfg-msg").textContent = "";
    };
    $("#cfg-file").onchange = load;
    $("#cfg-save").onclick = async () => {
      const r = await fetch("/api/config/" + $("#cfg-file").value, { method: "PUT", headers: { "Content-Type": "text/plain" },
        body: $("#cfg-text").value });
      $("#cfg-msg").textContent = r.ok ? "Saved. Applies to the next session." : "Not saved: " + (await r.json()).detail;
    };
    await load();
  },
};
