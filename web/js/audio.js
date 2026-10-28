// Audio in and out: mic capture and clip playback.

function openMic(deviceId) {
  const audio = { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 };
  if (deviceId) audio.deviceId = { exact: deviceId };
  return navigator.mediaDevices.getUserMedia({ audio });
}

async function startAudio() {
  if (S.micCtx) return;
  const saved = store.get("mic", "");  // each site starts on the system default mic, which may be the wrong one
  const stream = await (saved ? openMic(saved).catch(() => openMic()) : openMic());
  let ctx, src;
  try { ctx = new AudioContext({ sampleRate: 16000 }); src = ctx.createMediaStreamSource(stream); }
  catch { if (ctx) ctx.close(); ctx = new AudioContext(); src = ctx.createMediaStreamSource(stream); }
  await ctx.audioWorklet.addModule("/static/js/capture-worklet.js");
  const node = new AudioWorkletNode(ctx, "capture");
  node.port.onmessage = (e) => { if (S.ws && S.ws.readyState === 1 && !S.muted && !S.ended) S.ws.send(e.data); };
  const sink = ctx.createGain(); sink.gain.value = 0;
  src.connect(node).connect(sink).connect(ctx.destination);
  S.micAnalyser = ctx.createAnalyser(); src.connect(S.micAnalyser);
  Object.assign(S, { micCtx: ctx, micNode: node, micSrc: src, micStream: stream, heardAt: performance.now() });
  listMics();

  S.playCtx = new AudioContext();
  S.outAnalyser = S.playCtx.createAnalyser();
  let out = S.playCtx.destination;
  if (!S.headphones) {
    try { out = await loopbackOutput(S.playCtx); } catch (e) { log("loopback AEC unavailable: " + e.message); }
  }
  S.outAnalyser.connect(out);
  S.outInput = S.outAnalyser;
  requestAnimationFrame(meters);
}

// The voice check's microphone picker. Names are only readable once the mic is allowed.
async function listMics() {
  const mics = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
  const current = S.micStream.getAudioTracks()[0].getSettings().deviceId;
  $("#mic-device").innerHTML = mics.map((d, i) =>
    `<option value="${esc(d.deviceId)}" ${d.deviceId === current ? "selected" : ""}>${esc(d.label || `Microphone ${i + 1}`)}</option>`).join("");
}

async function switchMic(deviceId) {
  const stream = await openMic(deviceId);
  S.micStream.getTracks().forEach((t) => t.stop());
  S.micSrc.disconnect();
  S.micSrc = S.micCtx.createMediaStreamSource(stream);
  S.micSrc.connect(S.micNode); S.micSrc.connect(S.micAnalyser);
  S.micStream = stream; S.heardAt = performance.now();
  store.set("mic", deviceId);
  sendMicInfo();
}
$("#mic-device").onchange = (e) => switchMic(e.target.value).catch((err) => alert("Couldn't open that microphone: " + err.message));

// lands in the server log, so a silent mic can be told apart from a recognizer that doesn't recognize
function sendMicInfo() {
  const track = S.micStream.getAudioTracks()[0], set = track.getSettings();
  send({ type: "mic", label: track.label, muted: track.muted, track_rate: set.sampleRate || null,
         rate: S.micCtx.sampleRate, state: S.micCtx.state });
}

// Chrome's echo canceller only sees WebRTC audio, so playback goes through a local loopback connection.
async function loopbackOutput(ctx) {
  const dest = ctx.createMediaStreamDestination();
  const a = new RTCPeerConnection(), b = new RTCPeerConnection();
  a.onicecandidate = (e) => e.candidate && b.addIceCandidate(e.candidate);
  b.onicecandidate = (e) => e.candidate && a.addIceCandidate(e.candidate);
  b.ontrack = (e) => { const el = $("#loopback-audio"); el.srcObject = e.streams[0]; el.play().catch(() => {}); };
  dest.stream.getTracks().forEach((t) => a.addTrack(t, dest.stream));
  const offer = await a.createOffer(); await a.setLocalDescription(offer); await b.setRemoteDescription(offer);
  const answer = await b.createAnswer(); await b.setLocalDescription(answer); await a.setRemoteDescription(answer);
  return dest;
}

function rms(analyser) {
  const buf = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(buf);
  let s = 0; for (const v of buf) s += v * v;
  return Math.sqrt(s / buf.length);
}

const MUTED_HINT = "Your microphone is muted on this computer. Press the laptop's mic-mute key, or unmute it in the "
  + "system's sound settings (Windows: Settings > System > Sound > Input).";
const SILENT_HINT = "No sound from this microphone. Pick another one above, or check that it isn't muted.";

function meters() {
  const mic = S.muted ? 0 : Math.min(100, rms(S.micAnalyser) * 600);
  if (mic > 3) S.heardAt = performance.now();
  const muted = S.micStream.getAudioTracks()[0].muted, hint = $("#mic-silent");  // muted: the system sends no audio
  const text = muted ? MUTED_HINT : SILENT_HINT;
  if (hint.textContent !== text) hint.textContent = text;
  hint.classList.toggle("hidden", !muted && performance.now() - S.heardAt < 6000);
  $("#mic-level").style.width = mic + "%";
  $("#self-level").style.width = mic + "%";
  const out = Math.min(1, rms(S.outAnalyser) * 8);
  const glow = `0 0 0 ${Math.round(out * 22)}px rgba(91,140,255,${0.15 + out * 0.35})`;
  $("#avatar").style.boxShadow = glow;
  requestAnimationFrame(meters);
}

// Clips play back to back. Every stop bumps S.gen, which silently drops clips still being decoded.
function enqueueClip(header, audio) {
  const gen = S.gen;
  S.pending++;
  S.chain = S.chain.then(async () => {
    if (gen !== S.gen) return;
    let buffer;
    try { buffer = await S.playCtx.decodeAudioData(audio); }
    catch (e) { log("audio decode failed: " + e.message); S.pending--; checkIdle(); return; }
    if (gen !== S.gen) return;
    const ctx = S.playCtx, src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(S.outInput);
    const at = Math.max(ctx.currentTime + 0.03, S.nextTime);
    src.start(at);
    S.nextTime = at + buffer.duration;
    S.active.add(src);
    setTimeout(() => {
      if (gen !== S.gen) return;
      send({ type: "clip_start", id: header.id });
      S.aiCaption = (S.aiCaption + " " + header.text).trim();
      $("#cap-ai").textContent = S.aiCaption;
      setState("speaking");
    }, Math.max(0, (at - ctx.currentTime) * 1000));
    src.onended = () => { if (gen !== S.gen) return; S.active.delete(src); S.pending--; checkIdle(); };
  });
}

function checkIdle() {
  if (S.pending <= 0 && S.active.size === 0) {
    S.pending = 0;
    send({ type: "playback_idle" });
    if (!S.ended) setState("listening");
  }
}

function stopAudio() {
  S.gen++;
  for (const s of S.active) { try { s.stop(); } catch {} }
  S.active.clear(); S.pending = 0; S.nextTime = 0; S.chain = Promise.resolve();
}
