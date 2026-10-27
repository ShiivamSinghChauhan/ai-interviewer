// Audio in and out: mic capture, clip playback, the phone-line effect and ring tones.

async function startAudio() {
  if (S.micCtx) return;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  let ctx, src;
  try { ctx = new AudioContext({ sampleRate: 16000 }); src = ctx.createMediaStreamSource(stream); }
  catch { if (ctx) ctx.close(); ctx = new AudioContext(); src = ctx.createMediaStreamSource(stream); }
  await ctx.audioWorklet.addModule("/static/js/capture-worklet.js");
  const node = new AudioWorkletNode(ctx, "capture");
  node.port.onmessage = (e) => { if (S.ws && S.ws.readyState === 1 && !S.muted && !S.ended) S.ws.send(e.data); };
  const sink = ctx.createGain(); sink.gain.value = 0;
  src.connect(node).connect(sink).connect(ctx.destination);
  S.micAnalyser = ctx.createAnalyser(); src.connect(S.micAnalyser);
  S.micCtx = ctx;

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

// Telephone sound for the caller's voice: 300-3400 Hz band, light compression, line hiss, dropouts.
function phoneLine(line) {
  const ctx = S.playCtx;
  const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = line.bandpass ? 300 : 40;
  const lp = ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = line.bandpass ? 3400 : 16000;
  const comp = ctx.createDynamicsCompressor(); comp.threshold.value = -24; comp.ratio.value = 4;
  const drop = ctx.createGain();
  hp.connect(lp).connect(comp).connect(drop).connect(S.outAnalyser);
  S.outInput = hp;
  let noiseSrc = null, dropTimer = null;
  if (line.noise > 0) {
    const buf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    noiseSrc = ctx.createBufferSource(); noiseSrc.buffer = buf; noiseSrc.loop = true;
    const nf = ctx.createBiquadFilter(); nf.type = "bandpass"; nf.frequency.value = 1500; nf.Q.value = 0.6;
    const ng = ctx.createGain(); ng.gain.value = line.noise;
    noiseSrc.connect(nf).connect(ng).connect(S.outAnalyser);
    noiseSrc.start();
  }
  if (line.dropouts) {
    dropTimer = setInterval(() => {
      if (Math.random() < 0.25) {
        const t = ctx.currentTime;
        drop.gain.setValueAtTime(1, t); drop.gain.linearRampToValueAtTime(0.08, t + 0.03);
        drop.gain.linearRampToValueAtTime(1, t + 0.15 + Math.random() * 0.25);
      }
    }, 3000);
  }
  return () => { if (noiseSrc) noiseSrc.stop(); if (dropTimer) clearInterval(dropTimer); S.outInput = S.outAnalyser; };
}

// pattern: [[on_s, off_s], ...], repeated until the returned function is called
function tone(pattern, freqs, gain = 0.12) {
  const ctx = S.playCtx;
  let stopped = false, timer = null;
  const play = () => {
    if (stopped) return;
    let t = ctx.currentTime, total = 0;
    for (const [on, off] of pattern) {
      const g = ctx.createGain(); g.gain.value = 0; g.connect(S.outAnalyser);
      for (const f of freqs) { const o = ctx.createOscillator(); o.frequency.value = f; o.connect(g); o.start(t + total); o.stop(t + total + on); }
      g.gain.setValueAtTime(gain, t + total); g.gain.setValueAtTime(0, t + total + on);
      total += on + off;
    }
    timer = setTimeout(play, total * 1000);
  };
  play();
  return () => { stopped = true; clearTimeout(timer); };
}

function rms(analyser) {
  const buf = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(buf);
  let s = 0; for (const v of buf) s += v * v;
  return Math.sqrt(s / buf.length);
}

function meters() {
  const mic = S.muted ? 0 : Math.min(100, rms(S.micAnalyser) * 600);
  $("#mic-level").style.width = mic + "%";
  $("#self-level").style.width = mic + "%";
  const out = Math.min(1, rms(S.outAnalyser) * 8);
  const glow = `0 0 0 ${Math.round(out * 22)}px rgba(91,140,255,${0.15 + out * 0.35})`;
  $("#avatar").style.boxShadow = glow;
  $("#ph-avatar").style.boxShadow = glow;
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
      $("#ph-caption").textContent = S.aiCaption;
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
