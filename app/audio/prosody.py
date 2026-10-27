"""Acoustic features for the speech analysis: pitch (F0), loudness, and possible prolongations.

Pure numpy, CPU-cheap. These are estimates for practice feedback, not clinical measurements.
"""
import numpy as np

SR = 16000


def _frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        return np.zeros((0, win), dtype=np.float32)
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def loudness_db(x: np.ndarray, win: int = 400, hop: int = 160) -> np.ndarray:
    f = _frames(x.astype(np.float32), win, hop)
    return 10 * np.log10(np.mean(f * f, axis=1) + 1e-10)


def pitch(x: np.ndarray, fmin: float = 70, fmax: float = 400, win: int = 640, hop: int = 160,
          clarity_min: float = 0.5) -> np.ndarray:
    """Autocorrelation F0 per 10 ms frame (Hz); 0 where unvoiced/unclear."""
    f = _frames(x.astype(np.float32), win, hop)
    if not len(f):
        return np.zeros(0)
    f = f - f.mean(axis=1, keepdims=True)
    f *= np.hanning(win)[None, :]
    spec = np.fft.rfft(f, n=2 * win, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :win]
    ac /= ac[:, :1] + 1e-9
    lo, hi = int(SR / fmax), int(SR / fmin)
    seg = ac[:, lo:hi]
    lag = np.argmax(seg, axis=1) + lo
    clarity = seg[np.arange(len(seg)), lag - lo]
    energy = loudness_db(x, win, hop)[: len(f)]
    voiced = (clarity >= clarity_min) & (energy > energy.max() - 35)
    return np.where(voiced, SR / lag, 0.0)


def _mel_matrix(n_fft: int = 400, n_mels: int = 40) -> np.ndarray:
    def hz2mel(h):
        return 2595 * np.log10(1 + h / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    pts = mel2hz(np.linspace(hz2mel(60), hz2mel(7600), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        a, b, c = bins[i - 1], bins[i], bins[i + 1]
        if b > a:
            fb[i - 1, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[i - 1, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


_MEL = _mel_matrix()


def prolongations(x: np.ndarray, min_ms: float = 320, dist_max: float = 0.6, hop: int = 160,
                  lag_frames: int = 10) -> list[float]:
    """Durations (s) of stretches where a loud sound keeps the same spectrum for a long time
    (e.g. "sssso", "mmmmy"). Spectra are smoothed over 50 ms and compared 100 ms apart, because in
    fluent speech the sound changes within ~100-200 ms."""
    f = _frames(x.astype(np.float32), 400, hop)
    if len(f) < lag_frames + 5:
        return []
    spec = np.abs(np.fft.rfft(f * np.hanning(400)[None, :], axis=1)) ** 2
    logmel = np.log(spec @ _MEL.T + 1e-8)
    k = np.ones(5) / 5
    smooth = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, logmel)
    dist = np.full(len(f), np.inf)
    dist[lag_frames:] = np.mean(np.abs(smooth[lag_frames:] - smooth[:-lag_frames]), axis=1)
    energy = loudness_db(x, 400, hop)[: len(f)]
    steady = (dist < dist_max) & (energy > energy.max() - 25)
    out, run = [], 0
    for s in np.r_[steady, False]:
        if s:
            run += 1
        else:
            # a steady run of R frames at lag L means the sound held for about (R + L) frames
            dur = (run + lag_frames) * hop / SR if run else 0
            if dur * 1000 >= min_ms:
                out.append(round(dur, 2))
            run = 0
    return out


def features(x: np.ndarray, cfg: dict | None = None) -> dict:
    """Per-utterance acoustic summary."""
    cfg = cfg or {}
    if len(x) < SR * 0.3:
        return {}
    f0 = pitch(x)
    v = f0[f0 > 0]
    db = loudness_db(x)
    speech_db = db[db > db.max() - 35]
    out = {
        "f0_median": round(float(np.median(v)), 1) if len(v) > 10 else None,
        "f0_spread_st": round(float(np.std(12 * np.log2(v / np.median(v)))), 2) if len(v) > 10 else None,
        "loudness_db": round(float(np.median(speech_db)), 1) if len(speech_db) else None,
        "prolongations": prolongations(x, cfg.get("prolongation_min_ms", 320), cfg.get("prolongation_dist_max", 0.6)),
    }
    return out
