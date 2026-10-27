// Phone-call practice: waits for (or places) a call, rings, phone-line audio, call screen.
const Phone = {
  opts: null, voice: null, call: null, stopTone: null, stopLine: null, declines: 0, waitTimer: null,
  async init() {
    if (this.opts) return;
    const o = (this.opts = await fetch("/api/phone/options").then((r) => r.json()));
    const f = $("#ph-form"), d = { ...o.defaults, ...store.get("ph_last", {}) };
    fillSelect("#ph-category", { any: "Any kind", ...o.categories });
    fillSelect("#ph-difficulty", o.difficulties);
    $("#ph-line").innerHTML += Object.entries(o.lines).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
    for (const k of ["direction", "category", "difficulty", "region", "duration_min"]) if (d[k] !== undefined) f.elements[k].value = d[k];
    const fillScen = () => {
      const dir = f.direction.value, cat = f.category.value;
      const list = o.scenarios.filter((s) => (dir === "any" || s.direction === dir) && (cat === "any" || s.category === cat));
      $("#ph-scenario").innerHTML = `<option value="random">🎲 Random (${list.length} possible)</option>` +
        list.map((s) => `<option value="${s.id}">${esc(s.title)}</option>`).join("");
    };
    f.direction.onchange = fillScen; f.category.onchange = fillScen; fillScen();
    this.voice = voiceBlock($("#ph-voiceblock"), "ph", { ...d, tts_engine: d.tts_engine || "edge" });
    f.onsubmit = (e) => { e.preventDefault(); this.start(); };
  },

  async start() {
    const f = $("#ph-form"), v = this.voice();
    const setup = Object.fromEntries(new FormData(f));
    setup.duration_min = Number(setup.duration_min);
    const win = Number($("#ph-when").value);
    setup.surprise_window_min = win;
    Object.assign(setup, v);
    if (!setup.line) delete setup.line;
    store.set("ph_last", setup);
    this.call = null; this.declines = 0;
    S.handlers = {
      ready: (m) => { this.call = m.call; },
      afterMic: () => this.waitForCall(),
      state: (s) => this.onState(s),
      tick: (t) => { if (S.started) $("#ph-sub").textContent = t; },
      partial: (m) => { if (S.started) $("#ph-caption").textContent = (S.aiCaption ? S.aiCaption + "\n\n" : "") + "You: " + m.text; },
      ended: () => { this.cleanup(); },
    };
    $("#mic-sentence").textContent = "Hello, this is ___ speaking. Yes, I can hear you. Could you please repeat the order number? Thank you, bye.";
    await beginSetup({ setup }, { headphones: v.headphones });
  },

  async waitForCall() {
    send({ type: "skip_enroll" });  // the mic check already ran; stop enrolment
    show("call");
    $("#ph-buttons").innerHTML = ""; $("#ph-task").classList.add("hidden"); $("#ph-caption").textContent = "";
    $("#ph-caller").textContent = "-"; $("#ph-sub").textContent = "";
    $("#ph-status").textContent = "Getting the call ready…";
    while (!this.call) await new Promise((r) => setTimeout(r, 300));
    const c = this.call;
    if (c.direction === "outgoing") return this.outgoing();
    const [lo, hi] = c.ring_delay_s || [2, 8];
    const delay = c.surprise_window_min > 0 ? Math.random() * c.surprise_window_min * 60 : lo + Math.random() * (hi - lo);
    $("#ph-status").textContent = c.surprise_window_min > 0
      ? `Your phone will ring sometime in the next ${c.surprise_window_min} minutes. Keep this tab open.` : "Waiting for a call…";
    $("#ph-avatar").textContent = "📱";
    this.waitTimer = setTimeout(() => this.ring(), delay * 1000);
  },

  ring() {
    const c = this.call;
    $("#ph-status").textContent = "Incoming call";
    $("#ph-caller").textContent = c.caller_id;
    $("#ph-sub").textContent = "mobile";
    $("#ph-avatar").textContent = "👤"; $("#ph-avatar").classList.add("ringing");
    this.stopTone = tone([[0.4, 0.2], [0.4, 2.0]], [440, 480], 0.15);
    $("#ph-buttons").innerHTML = `<button class="decline" title="Decline">✕</button><button class="accept" title="Answer">✆</button>`;
    $("#ph-buttons .accept").onclick = () => this.connectCall();
    $("#ph-buttons .decline").onclick = () => this.decline();
    this.ringTimeout = setTimeout(() => this.decline(true), 30000);  // unanswered for 30 s = missed
  },

  decline(missed = false) {
    clearTimeout(this.ringTimeout);
    if (this.stopTone) this.stopTone();
    $("#ph-avatar").classList.remove("ringing");
    send({ type: "declined" });
    this.declines++;
    $("#ph-buttons").innerHTML = `<button class="wide" id="ph-home">Back to setup</button>`;
    $("#ph-home").onclick = () => location.reload();
    if (this.declines >= 2) { $("#ph-status").textContent = missed ? "Missed call" : "Call declined"; return; }
    const after = this.call.callback_after_decline_s || 20;
    $("#ph-status").textContent = `${missed ? "Missed call" : "Declined"}. They'll probably try again…`;
    this.waitTimer = setTimeout(() => this.ring(), after * 1000);
  },

  outgoing() {
    const c = this.call;
    $("#ph-status").textContent = "Your task";
    $("#ph-caller").textContent = c.callee || "Calling…";
    $("#ph-task").textContent = c.user_task; $("#ph-task").classList.remove("hidden");
    $("#ph-avatar").textContent = "📞";
    $("#ph-buttons").innerHTML = `<button class="accept" title="Call">✆</button>`;
    $("#ph-buttons .accept").onclick = () => {
      $("#ph-status").textContent = "Calling…";
      $("#ph-buttons").innerHTML = "";
      this.stopTone = tone([[0.4, 0.2], [0.4, 2.0]], [400, 450], 0.08);
      setTimeout(() => this.connectCall(), (2 + Math.random() * 4) * 1000);
    };
  },

  connectCall() {
    clearTimeout(this.ringTimeout);
    if (this.stopTone) this.stopTone();
    $("#ph-avatar").classList.remove("ringing");
    this.stopLine = phoneLine(this.call.line || { bandpass: true, noise: 0.003 });
    $("#ph-status").textContent = "Connected";
    $("#ph-sub").textContent = "00:00";
    $("#ph-buttons").innerHTML = `<button id="ph-mute" title="Mute (M)">Mute</button>
      <button id="ph-done" title="Space">Done ␣</button><button class="hangup" id="ph-hang" title="Hang up">✆</button>`;
    $("#ph-mute").onclick = () => { toggleMute(); $("#ph-mute").textContent = S.muted ? "Unmute" : "Mute"; };
    $("#ph-done").onclick = endMyTurn;
    $("#ph-hang").onclick = () => { send({ type: "end" }); };
    send({ type: "begin" });
  },

  onState(s) {
    if (!S.started) return;
    $("#ph-status").textContent = { speaking: "On call · they're talking", user: "On call · you're talking",
      listening: "On call", thinking: "On call" }[s] || "On call";
  },

  cleanup() {
    clearTimeout(this.waitTimer); clearTimeout(this.ringTimeout);
    if (this.stopTone) this.stopTone();
    if (this.stopLine) this.stopLine();
  },
};
