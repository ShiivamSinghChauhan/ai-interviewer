// Interviewer agent setup: role, JD, the candidate's resume and repositories, preparation options.
const Interview = {
  opts: null, voice: null,
  async init() {
    if (this.opts) return;
    const o = (this.opts = await fetch("/api/interview/options").then((r) => r.json()));
    const f = $("#iv-form"), d = { ...o.defaults, ...store.get("iv_last", {}) };
    fillSelect("#iv-level", o.levels); fillSelect("#iv-company", o.companies); fillSelect("#iv-style", o.styles);
    for (const k of ["candidate_name", "role", "level", "company", "duration_min", "style", "interviewer_name", "jd", "resume"])
      if (f.elements[k] && d[k] !== undefined) f.elements[k].value = d[k];
    const focus = d.focus || o.defaults.focus;
    $("#iv-focus").innerHTML = Object.entries(o.focus_areas)
      .map(([k, t]) => `<span class="chip ${focus.includes(k) ? "on" : ""}" data-k="${k}">${esc(t)}</span>`).join("");
    $("#iv-focus").onclick = (e) => e.target.classList.contains("chip") && e.target.classList.toggle("on");

    const p = o.prep;
    $("#iv-prep").checked = p.enabled; $("#iv-step-repos").checked = p.steps.repos; $("#iv-step-critique").checked = p.steps.critique;
    $("#iv-history").checked = p.history; $("#iv-show-plan").checked = p.show_plan; $("#iv-reuse").checked = o.repos.reuse_cached;
    $("#iv-temp").value = p.question_temperature; $("#iv-per-section").value = p.per_section;
    const tv = () => ($("#iv-temp-val").textContent = Number($("#iv-temp").value).toFixed(1));
    $("#iv-temp").oninput = tv; tv();

    (d.repos || []).forEach((r) => this.addRepo(r.project, r.source));
    $("#iv-add-repo").onclick = () => this.addRepo("", "");
    $("#iv-find-projects").onclick = () => this.findProjects();
    this.voice = voiceBlock($("#iv-voiceblock"), "iv", d);
    f.onsubmit = (e) => { e.preventDefault(); this.start(); };
  },

  addRepo(project, source) {
    const where = S.meta.public ? "https://github.com/you/repo" : "https://github.com/you/repo  or  C:\\path\\to\\folder";
    $("#iv-repos").insertAdjacentHTML("beforeend", `
      <div class="repo-row"><input class="proj-in" placeholder="Project name" value="${esc(project)}">
        <input class="src-in" placeholder="${where}" value="${esc(source)}">
        <label class="check small"><input type="checkbox" class="force-in"> re-analyse</label>
        <button type="button" class="ghost remove">✕</button></div>`);
    const row = $("#iv-repos").lastElementChild;
    row.querySelector(".remove").onclick = () => row.remove();
  },

  async findProjects() {
    const resume = $("#iv-resume").value;
    if (!resume.trim()) return alert("Paste the resume first.");
    const btn = $("#iv-find-projects");
    btn.disabled = true; btn.textContent = "Reading the resume…";
    try {
      const ps = await fetch("/api/interview/projects", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ resume }) }).then((r) => r.json());
      const have = new Set($$("#iv-repos .proj-in").map((i) => i.value.toLowerCase()));
      for (const p of ps) if (!have.has((p.name || "").toLowerCase()))
        this.addRepo(p.name, (p.links || []).find((l) => /github|gitlab|bitbucket/.test(l)) || "");
      if (!ps.length) alert("No projects found - add repositories manually.");
    } finally { btn.disabled = false; btn.textContent = "Find projects in the resume"; }
  },

  async start() {
    const f = $("#iv-form"), v = this.voice();
    const setup = Object.fromEntries(new FormData(f));
    setup.duration_min = Number(setup.duration_min);
    setup.focus = $$("#iv-focus .chip.on").map((c) => c.dataset.k);
    Object.assign(setup, v);
    const repos = $$("#iv-repos .repo-row").map((r) => ({
      project: r.querySelector(".proj-in").value.trim(), source: r.querySelector(".src-in").value.trim(),
      force: r.querySelector(".force-in").checked })).filter((r) => r.source);
    const overrides = {
      prep: { enabled: $("#iv-prep").checked, show_plan_to_candidate: $("#iv-show-plan").checked,
        history: { enabled: $("#iv-history").checked },
        steps: { repos: { enabled: $("#iv-step-repos").checked }, critique: { enabled: $("#iv-step-critique").checked },
          questions: { temperature: Number($("#iv-temp").value), per_section: Number($("#iv-per-section").value) } } },
      repos: { reuse_cached: $("#iv-reuse").checked },
    };
    store.set("iv_last", { ...setup, repos: repos.map(({ project, source }) => ({ project, source })) });
    await this.launch(setup, { repos, overrides, prep: $("#iv-prep").checked });
  },

  // also the entry point for a session link, where the server already holds the repos and overrides
  async launch(setup, { repos = [], overrides = {}, prep = true } = {}) {
    $("#avatar-initials").textContent = (setup.interviewer_name || "A")[0].toUpperCase();
    $("#partner-label").textContent = `${setup.interviewer_name || "Interviewer"} · Interviewer`;
    $("#room-sub").textContent = "Introduction";
    $("#live-manual").checked = !!setup.manual_turns;
    S.handlers = {};
    if (await beginSetup({ setup, repos, overrides }, { headphones: setup.headphones, prep }))
      $("#live-patience").value = setup.patience || "patient";
  },
};
