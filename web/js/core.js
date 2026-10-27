// Practice Studio in the browser: shared state, the socket, the setup -> mic check -> session flow, the room.
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const S = {
  meta: null, mode: null, ws: null, headphones: true, handlers: {},
  micCtx: null, micAnalyser: null, muted: false,
  playCtx: null, outAnalyser: null, outInput: null, gen: 0, chain: Promise.resolve(), nextTime: 0,
  active: new Set(), pending: 0, aiCaption: "",
  started: false, startAt: 0, duration: 15, ended: false, timer: null,
};
const SCREENS = ["home", "interview", "conversation", "phone", "mic", "room", "call", "report", "history", "repos", "settings"];
function show(name) { for (const s of SCREENS) $(`#screen-${s}`).classList.toggle("hidden", s !== name); window.scrollTo(0, 0); }
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const fillSelect = (sel, obj, labelFn) => {
  const entries = Array.isArray(obj) ? obj.map((v) => [v, v]) : Object.entries(obj);
  $(sel).innerHTML = entries.map(([k, v]) => `<option value="${esc(k)}">${esc(labelFn ? labelFn(k, v) : v)}</option>`).join("");
};
const fmt = (v, d = 2) => (v === null || v === undefined || v < 0 ? "-" : Number(v).toFixed(d));
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k) || "null") ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

// ---------- boot ----------
async function boot() {
  S.meta = await fetch("/api/meta").then((r) => r.json());
  const p = S.meta.providers, e = S.meta.engines;
  const pill = (label, ok) => `<span class="pill ${ok ? "ok" : "off"}">${label}</span>`;
  $("#pills").innerHTML = [
    pill(`LLM ${p.groq + p.openrouter + p.gemini ? "ok" : "no key"}`, p.groq + p.openrouter + p.gemini),
    pill(`Jev ${p.jev ? "ok" : "off"}`, p.jev),
    pill(`Fish ${e.fish ? "×" + p.fish : "off"}`, e.fish),
    pill(`ElevenLabs ${e.elevenlabs ? "×" + p.elevenlabs : "off"}`, e.elevenlabs),
  ].join("");
  if (S.meta.public) publicDemo();
  $$(".mode-card").forEach((b) => (b.onclick = () => openMode(b.dataset.mode)));
  $("#home-link").onclick = (ev) => { ev.preventDefault(); if (!S.started) show("home"); };
  $("#nav-repos").onclick = () => !S.started && Tools.repos();
  $("#nav-settings").onclick = () => !S.started && Tools.settings();
  $("#nav-history").onclick = () => !S.started && Tools.history();
  $("#btn-again").onclick = () => location.reload();
  const h = location.hash.slice(1);  // deep links: #interview #conversation #phone #repos #settings #history
  if (["interview", "conversation", "phone"].includes(h)) openMode(h);
  else if (h === "repos") Tools.repos();
  else if (h === "settings") Tools.settings();
  else if (h === "history" && !S.meta.public) Tools.history();
  else if (h.startsWith("report/")) Tools.openReport(...h.split("/").slice(1));
  else show("home");
}

// The hosted demo keeps nothing and can't be reconfigured from the browser; say so up front.
function publicDemo() {
  const cap = S.meta.max_session_min ? ` Sessions are capped at ${S.meta.max_session_min} minutes.` : "";
  const busy = S.meta.busy ? " <b>Someone is practising right now</b>, so you may have to wait a few minutes." : "";
  $("#demo-note").innerHTML = `Public demo: one person at a time, nothing is saved, and it works best in Chrome or Edge
    with headphones.${cap}${busy}`;
  $("#demo-note").classList.remove("hidden");
  $("#nav-history").classList.add("hidden");
}

async function openMode(mode) {
  S.mode = mode;
  show(mode);
  await ({ interview: Interview, conversation: Conversation, phone: Phone }[mode]).init();
  const cap = S.meta.max_session_min;
  if (cap) for (const input of $$("input[name=duration_min]")) { input.max = cap; input.value = Math.min(input.value, cap); }
}

// voice + support controls shared by every setup form
function voiceBlock(container, prefix, d) {
  const e = S.meta.engines;
  container.innerHTML = `
    <label>Voice engine<select id="${prefix}-engine">
      <option value="edge">Neural (free)</option>
      <option value="fish" ${e.fish ? "" : "disabled"}>Fish Audio (expressive, free tier)</option>
      <option value="elevenlabs" ${e.elevenlabs ? "" : "disabled"}>ElevenLabs (realistic)</option>
      <option value="kokoro" ${e.kokoro ? "" : "disabled"}>Offline (Kokoro, slow)</option></select></label>
    <label class="span2">Voice<select id="${prefix}-voice"></select></label>
    <label>Speaking speed <span class="val" id="${prefix}-speed-val"></span><input id="${prefix}-speed" type="range" min="0.75" max="1.25" step="0.05"></label>
    <label>Patience with my pauses<select id="${prefix}-patience"></select></label>
    <label>Ending my turn<select id="${prefix}-manual"><option value="false">Automatic + Space key</option><option value="true">Only when I press Space</option></select></label>
    <label class="check span3"><input type="checkbox" id="${prefix}-headphones" checked> I'm wearing headphones (uncheck on laptop speakers: extra echo cancellation)</label>`;
  fillSelect(`#${prefix}-patience`, S.meta.patience);
  const engine = $(`#${prefix}-engine`);
  engine.value = d.tts_engine && S.meta.engines[d.tts_engine] ? d.tts_engine : "edge";
  $(`#${prefix}-speed`).value = d.speed || 1;
  $(`#${prefix}-patience`).value = d.patience || "patient";
  $(`#${prefix}-manual`).value = String(!!d.manual_turns);
  const sp = () => ($(`#${prefix}-speed-val`).textContent = Number($(`#${prefix}-speed`).value).toFixed(2) + "×");
  $(`#${prefix}-speed`).oninput = sp; sp();
  const fill = async (keep) => {
    const eng = engine.value;
    let list = [["", eng === "edge" ? "Auto (matches the person / Indian English)" : "Default"]];
    if (eng === "fish" || eng === "elevenlabs") {
      const vs = await fetch(`/api/voices/${eng}`).then((r) => r.json()).catch(() => []);
      list = list.concat(vs.map((v) => [v.id, v.name]));
    } else if (eng === "edge") {
      list = list.concat(["en-IN-PrabhatNeural", "en-IN-NeerjaNeural", "en-IN-NeerjaExpressiveNeural"].map((v) => [v, v]));
    } else list = list.concat(["hm_omega", "hm_psi", "hf_alpha", "hf_beta"].map((v) => [v, v]));
    $(`#${prefix}-voice`).innerHTML = list.map(([id, n]) => `<option value="${esc(id)}">${esc(n)}</option>`).join("");
    if (keep) $(`#${prefix}-voice`).value = keep;
  };
  engine.onchange = () => fill();
  fill(d.voice);
  return () => ({
    tts_engine: engine.value, voice: $(`#${prefix}-voice`).value, speed: Number($(`#${prefix}-speed`).value),
    patience: $(`#${prefix}-patience`).value, manual_turns: $(`#${prefix}-manual`).value === "true",
    headphones: $(`#${prefix}-headphones`).checked,
  });
}

// ---------- socket ----------
// Text frames are JSON events; binary frames are voice clips: 4-byte header length, JSON header, then the audio.
function send(obj) { if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify(obj)); }

function connect(mode, setupMsg) {
  return new Promise((resolve) => {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${scheme}://${location.host}/ws/${mode}`);
    ws.binaryType = "arraybuffer";
    S.ws = ws;
    ws.onopen = () => { send({ type: "setup", ...setupMsg }); resolve(); };
    ws.onclose = () => { if (!S.ended) log("connection closed"); };
    ws.onmessage = (e) => {
      if (typeof e.data !== "string") {
        const n = new DataView(e.data).getUint32(0, true);
        const header = JSON.parse(new TextDecoder().decode(new Uint8Array(e.data, 4, n)));
        enqueueClip(header, e.data.slice(4 + n));
        return;
      }
      const m = JSON.parse(e.data);
      if (S.handlers[m.type] && S.handlers[m.type](m) === false) return;  // a mode's handler can swallow it
      onMessage(m);
    };
  });
}

function onMessage(m) {
  switch (m.type) {
    case "partial":
      if (!S.started) $("#mic-transcript").textContent = m.text;
      else { const el = $("#cap-user"); el.textContent = m.text; el.classList.toggle("tentative", !!m.tentative); }
      break;
    case "user_speaking": if (m.interrupt) stopAudio(); if (S.started) setState("user"); break;
    case "user_final":
      if (S.started) { $("#cap-user").textContent = m.text; $("#cap-user").classList.remove("tentative"); }
      log(`turn end: ${m.info.reason}${m.info.pauses && m.info.pauses.length ? `, ${m.info.pauses.length} pause(s)` : ""}`);
      break;
    case "user_idle": case "your_turn": if (S.started && !S.active.size) setState("listening"); break;
    case "ai_thinking": S.aiCaption = ""; setState("thinking"); break;
    case "stop_audio": stopAudio(); break;
    case "metrics": signal("Reply time", m.response_ms + " ms"); break;
    case "debug":
      if (m.smart_turn !== undefined) signal("Sounded finished", fmt(m.smart_turn));
      if (m.semantic_done !== undefined) signal("Thought complete", fmt(m.semantic_done));
      if (m.speaker_sim !== undefined) signal("Voice match", fmt(m.speaker_sim));
      break;
    case "assessment": $("#assessment").textContent = m.text; break;
    case "enrolled":
      $("#mic-transcript").textContent = m.text;
      $("#mic-status").textContent = "Got your voice and your calm baseline. Read again to recalibrate, or start.";
      $("#btn-begin").disabled = false;
      break;
    case "enroll_failed": $("#mic-status").textContent = "Please read the whole paragraph - take all the time you need."; break;
    case "prep": prepStep(m); break;
    case "prep_done": prepDone(m); break;
    case "started": onStarted(m); break;
    case "section": log("section → " + m.title); $("#room-sub").textContent = m.title; break;
    case "notice": log("⚠ " + m.text); break;
    case "ended": S.ended = true; if (S.timer) clearInterval(S.timer); show("report"); break;
    case "status": if (S.ended) $("#report-status").textContent = m.text; else log(m.text); break;
    case "report": Report.render(m); break;
    case "error": alert(m.text); if (!S.started) location.reload(); break;
  }
}

function onStarted(m) {
  S.started = true; S.startAt = performance.now(); S.duration = m.duration_min;
  if (S.mode !== "phone") { show("room"); setState("thinking"); }
  S.timer = setInterval(tick, 500);
}

// ---------- setup -> mic check -> begin ----------
async function beginSetup(setupMsg, opts = {}) {
  S.headphones = opts.headphones !== false;
  $("#live-patience").innerHTML = Object.entries(S.meta.patience).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
  try { await startAudio(); } catch (err) { alert("Microphone access failed: " + err.message); return false; }
  await connect(S.mode, setupMsg);
  $("#prep-card").classList.toggle("hidden", !opts.prep);
  $("#prep-steps").innerHTML = ""; $("#prep-plan").innerHTML = "";
  show("mic");
  return true;
}
$("#btn-skip").onclick = () => { send({ type: "skip_enroll" }); afterMic(); };
$("#btn-begin").onclick = () => afterMic();
function afterMic() {
  if (S.handlers.afterMic) return S.handlers.afterMic();
  send({ type: "begin" });
}

// ---------- interview prep progress ----------
const PREP_NAMES = { jd: "Job description", resume: "Resume vs JD", plan: "Planning unique questions", critique: "Bar-raiser review" };
function prepStep(m) {
  const id = "prep-" + m.step.replace(/[^a-z0-9]/gi, "_");
  let li = document.getElementById(id);
  if (!li) {
    const name = PREP_NAMES[m.step] || (m.step.startsWith("repo:") ? "Repo: " + m.step.slice(5) : m.step);
    $("#prep-steps").insertAdjacentHTML("beforeend", `<li id="${id}"><span class="mark"></span><span>${esc(name)}</span><span class="det"></span></li>`);
    li = document.getElementById(id);
  }
  li.querySelector(".mark").className = "mark st-" + m.status;
  li.querySelector(".det").textContent = m.detail;
}
function prepDone(m) {
  const html = m.error ? `<p class="muted">Preparation hit a problem (${esc(m.error)}); the interview will use the standard plan.</p>`
    : `<p class="muted">Ready in ${m.seconds}s: ${m.sections.map((s) => `${esc(s.title)} (${s.minutes}m)`).join(" → ")}</p>`;
  $("#prep-plan").innerHTML = html + (m.plan ? m.plan.map((s) => `<h4>${esc(s.title)}</h4><ul>${(s.questions || []).map((q) => `<li>${esc(q.q)}</li>`).join("")}</ul>`).join("") : "");
}

// ---------- room ----------
const LABELS = { listening: "Your turn - no rush", user: "Listening…", thinking: "…", speaking: "Speaking" };
function setState(s) {
  const el = $("#state");
  el.textContent = LABELS[s] || s;
  el.className = "state " + (s === "listening" ? "listening" : s === "user" ? "user" : "");
  if (S.handlers.state) S.handlers.state(s);
}
function tick() {
  if (S.ended) return;
  const sec = Math.floor((performance.now() - S.startAt) / 1000);
  const t = `${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(sec % 60).padStart(2, "0")}`;
  $("#timer").textContent = `${t} / ${S.duration}:00`;
  if (S.handlers.tick) S.handlers.tick(t);
}
function signal(name, value) {
  const dl = $("#signals");
  let dd = dl.querySelector(`[data-k="${name}"]`);
  if (!dd) { dl.insertAdjacentHTML("beforeend", `<dt>${name}</dt><dd data-k="${name}"></dd>`); dd = dl.querySelector(`[data-k="${name}"]`); }
  dd.textContent = value;
}
function log(text) { const el = $("#log"); el.textContent = `${new Date().toLocaleTimeString()} ${text}\n` + el.textContent.slice(0, 6000); }
function endMyTurn() { if (S.started && !S.ended) { send({ type: "end_turn" }); log("you ended your turn"); } }
function toggleMute() {
  S.muted = !S.muted;
  $("#btn-mute").classList.toggle("muted-on", S.muted);
  $("#btn-mute").textContent = S.muted ? "Unmute" : "Mute";
}
$("#btn-done").onclick = endMyTurn;
$("#btn-mute").onclick = toggleMute;
$("#btn-captions").onclick = () => { $("#captions").classList.toggle("off"); $("#btn-captions").classList.toggle("on"); };
$("#btn-coach").onclick = () => { $("#coach").classList.toggle("hidden"); $("#btn-coach").classList.toggle("on"); };
$("#btn-end").onclick = () => { if (confirm("End the session and get your feedback?")) send({ type: "end" }); };
const livePatience = () => send({ type: "patience", value: $("#live-patience").value, manual: $("#live-manual").checked });
$("#live-patience").onchange = livePatience;
$("#live-manual").onchange = livePatience;
document.addEventListener("keydown", (e) => {
  if (!S.started || ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
  if (e.code === "Space") { e.preventDefault(); endMyTurn(); }
  if (e.key === "m") toggleMute();
  if (e.key === "c") $("#btn-captions").click();
});

window.addEventListener("load", boot);
