import io
import wave

import numpy as np


def wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Mono 16-bit WAV from float samples in [-1, 1]."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.clip(samples * 32767, -32768, 32767).astype(np.int16).tobytes())
    return buf.getvalue()
