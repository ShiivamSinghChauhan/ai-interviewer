"""The local models that listen: voice activity, speech-to-text, "did they finish?" and "is that you?".

All CPU. Nemotron 0.6B runs at about half real time on a Ryzen 5 5500U; the rest take a few ms per call.
"""
import logging
import os
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sherpa_onnx

from app.audio.whisper_features import N_SAMPLES, log_mel_8s

log = logging.getLogger("stt")
SR = 16000
FRAME = 512  # 32 ms, what Silero wants


def _onnx(path: str, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.inter_op_num_threads, so.intra_op_num_threads = 1, threads
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


class SileroVAD:
    def __init__(self, path: str):
        self.session = _onnx(path, 1)
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame.reshape(1, -1).astype(np.float32)], axis=1)
        inputs = {"input": x, "state": self.state, "sr": np.array(SR, dtype=np.int64)}
        out, self.state = self.session.run(None, inputs)
        self.context = x[:, -64:]
        return float(out[0][0])


class SmartTurn:
    """Smart Turn v3.2: judges from the audio alone whether the speaker sounded finished."""

    def __init__(self, path: str, threads: int = min(4, os.cpu_count() or 1)):
        self.session = _onnx(path, threads)

    def probability(self, audio: np.ndarray) -> float:
        audio = audio[-N_SAMPLES:]
        audio = np.pad(audio, (N_SAMPLES - len(audio), 0))  # the model expects the speech at the end
        out = self.session.run(None, {"input_features": log_mel_8s(audio)[None]})[0]
        return float(np.asarray(out).reshape(-1)[0])


class SpeakerGate:
    """TitaNet voice embeddings: keeps the TV, your roommate and the partner's own echo out of your turn."""

    def __init__(self, path: str, threads: int = 2):
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=path, num_threads=threads))

    def embed(self, audio: np.ndarray) -> np.ndarray:
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=SR, waveform=audio.astype(np.float32))
        stream.input_finished()
        emb = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)


class StreamingSTT:
    """NVIDIA Nemotron Speech Streaming 0.6B (INT8) through sherpa-onnx.

    Greedy decoding only: asking sherpa-onnx for beam search on a NeMo model kills the whole process (not an
    exception, the process), so hotwords are off the table. Jargon gets fixed after the fact instead.
    """

    def __init__(self, model_dir: str, threads: int = 3):
        d = Path(model_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"STT model not found at '{model_dir}'. Run scripts/download_models.py")

        def pick(stem: str) -> str:
            return str(sorted(d.glob(f"{stem}*.onnx"), key=lambda p: ("int8" not in p.name, p.name))[0])

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"),
            num_threads=threads, sample_rate=SR, enable_endpoint_detection=False, decoding_method="greedy_search")
        self.lock = threading.Lock()
        log.info("STT ready: %s", d.name)

    def new_stream(self):
        return self.recognizer.create_stream()

    def feed(self, stream, samples: np.ndarray) -> str:
        with self.lock:
            stream.accept_waveform(SR, samples)
            return self._decode(stream)

    def finish(self, stream, silence_fed: float = 0.0) -> str:
        """Flush the last chunk, padding only what the trailing silence hasn't covered yet."""
        with self.lock:
            if silence_fed < 0.6:
                stream.accept_waveform(SR, np.zeros(int((0.6 - silence_fed) * SR), dtype=np.float32))
            stream.input_finished()
            return self._decode(stream)

    def _decode(self, stream) -> str:
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        r = self.recognizer.get_result(stream)
        return (r if isinstance(r, str) else r.text).strip()
