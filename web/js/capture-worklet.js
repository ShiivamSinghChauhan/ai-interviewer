// Mic capture: resample to 16 kHz mono, post 512-sample Int16 frames (32 ms) to the main thread.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;                       // fractional read position into this.buf
    this.buf = new Float32Array(0);
    this.out = new Int16Array(512);
    this.n = 0;
  }

  emit(v) {
    const s = Math.max(-1, Math.min(1, v));
    this.out[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    if (this.n === this.out.length) {
      this.port.postMessage(this.out.buffer, [this.out.buffer]);
      this.out = new Int16Array(512);
      this.n = 0;
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    if (this.ratio === 1) {
      for (let i = 0; i < ch.length; i++) this.emit(ch[i]);
      return true;
    }
    const merged = new Float32Array(this.buf.length + ch.length);
    merged.set(this.buf);
    merged.set(ch, this.buf.length);
    const k = Math.round(this.ratio);
    const integer = Math.abs(this.ratio - k) < 1e-6;
    while (this.pos + this.ratio < merged.length) {
      const i = Math.floor(this.pos);
      if (integer) {
        let acc = 0;                    // box filter = cheap anti-aliasing for 48k -> 16k
        for (let j = 0; j < k; j++) acc += merged[i + j];
        this.emit(acc / k);
      } else {
        const f = this.pos - i;
        this.emit(merged[i] * (1 - f) + merged[i + 1] * f);
      }
      this.pos += this.ratio;
    }
    const used = Math.floor(this.pos);
    this.buf = merged.slice(used);
    this.pos -= used;
    return true;
  }
}
registerProcessor("capture", Capture);
