// Conversation partner setup: persona builder with presets, surprise and saved people.
const Conversation = {
  opts: null, voice: null, interests: [],
  async init() {
    if (this.opts) return;
    const o = (this.opts = await fetch("/api/conversation/options").then((r) => r.json()));
    fillSelect("#cv-accent", o.accents); fillSelect("#cv-mode", o.modes); fillSelect("#cv-scene", o.scenes);
    fillSelect("#cv-relationship", o.relationships); fillSelect("#cv-level", o.levels);
    fillSelect("#cv-length", o.lengths); fillSelect("#cv-corrections", o.corrections);
    $("#cv-preset").innerHTML += Object.keys(o.presets).map((k) => `<option>${esc(k)}</option>`).join("");
    $("#cv-traits").innerHTML = Object.keys(o.traits).map((k) => `
      <div class="trait"><span>${k[0].toUpperCase() + k.slice(1)}</span>
        <input type="range" min="1" max="5" step="1" data-trait="${k}"><span class="desc" id="cv-desc-${k}"></span></div>`).join("");
    $("#cv-traits").oninput = (e) => e.target.dataset.trait && this.traitLabel(e.target.dataset.trait);
    $("#cv-interests").onclick = (e) => {
      if (!e.target.classList.contains("chip")) return;
      e.target.classList.toggle("on");
      this.interests = $$("#cv-interests .chip.on").map((c) => c.dataset.k);
    };
    $("#cv-interest-add").onkeydown = (e) => {
      if (e.key !== "Enter") return;
      e.preventDefault();
      const v = e.target.value.trim().toLowerCase();
      if (v && !this.interests.includes(v)) this.interests.push(v);
      e.target.value = ""; this.renderInterests();
    };
    const f = $("#cv-form");
    f.age.oninput = () => ($("#cv-age-val").textContent = f.age.value);
    f.mode.onchange = () => $("#cv-scenario-wrap").classList.toggle("hidden", f.mode.value !== "roleplay");
    $("#cv-preset").onchange = (e) => e.target.value && this.apply({ ...o.defaults, ...o.presets[e.target.value] });
    $("#cv-surprise").onclick = async () => this.apply({ ...o.defaults, ...(await fetch("/api/conversation/surprise").then((r) => r.json())) });
    $("#cv-save").onclick = () => this.save();
    $("#cv-saved").onchange = (e) => { const p = store.get("tp_saved", {})[e.target.value]; if (p) this.apply(p); };
    this.renderSaved();
    const last = store.get("tp_last", null);
    this.voice = voiceBlock($("#cv-voiceblock"), "cv", last || o.defaults);
    this.apply(last || o.defaults);
    f.onsubmit = (e) => { e.preventDefault(); this.start(); };
  },
  traitLabel(k) {
    const v = Number(document.querySelector(`#cv-traits [data-trait="${k}"]`).value);
    $(`#cv-desc-${k}`).textContent = this.opts.traits[k][v - 1];
  },
  renderInterests() {
    const all = [...new Set([...this.opts.interests, ...this.interests])];
    $("#cv-interests").innerHTML = all.map((k) => `<span class="chip ${this.interests.includes(k) ? "on" : ""}" data-k="${esc(k)}">${esc(k)}</span>`).join("");
  },
  apply(p) {
    const f = $("#cv-form");
    for (const [k, v] of Object.entries(p)) {
      const el = f.elements[k];
      if (!el || el instanceof RadioNodeList) continue;
      if (el.type === "checkbox") el.checked = !!v; else el.value = typeof v === "boolean" ? String(v) : v;
    }
    for (const k of Object.keys(this.opts.traits)) {
      document.querySelector(`#cv-traits [data-trait="${k}"]`).value = (p.traits || {})[k] || 3;
      this.traitLabel(k);
    }
    this.interests = [...(p.interests || [])];
    this.renderInterests();
    $("#cv-age-val").textContent = f.age.value;
    $("#cv-scenario-wrap").classList.toggle("hidden", f.mode.value !== "roleplay");
  },
  collect() {
    const f = $("#cv-form");
    const p = Object.fromEntries(new FormData(f));
    p.age = Number(p.age); p.duration_min = Number(p.duration_min);
    p.keep_me_talking = f.keep_me_talking.checked; p.stay_in_character = f.stay_in_character.checked;
    p.interests = this.interests;
    p.traits = Object.fromEntries($$("#cv-traits [data-trait]").map((el) => [el.dataset.trait, Number(el.value)]));
    return { ...p, ...this.voice() };
  },
  renderSaved() {
    $("#cv-saved").innerHTML = `<option value="">My people…</option>` + Object.keys(store.get("tp_saved", {})).map((k) => `<option>${esc(k)}</option>`).join("");
  },
  save() {
    const p = this.collect(), all = store.get("tp_saved", {}), key = `${p.name} (${p.occupation})`;
    all[key] = p; store.set("tp_saved", all); this.renderSaved(); $("#cv-saved").value = key;
  },
  async start() {
    const p = this.collect();
    store.set("tp_last", p);
    $("#avatar-initials").textContent = (p.name || "?")[0].toUpperCase();
    $("#partner-label").textContent = `${p.name}, ${p.age} · ${p.occupation}`;
    $("#room-sub").textContent = this.opts.scenes[p.scene] || "";
    $("#live-manual").checked = p.manual_turns;
    S.handlers = {};
    if (await beginSetup({ persona: p }, { headphones: p.headphones })) $("#live-patience").value = p.patience;
  },
};
