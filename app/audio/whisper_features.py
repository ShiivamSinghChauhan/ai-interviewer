#
# Numpy-only Whisper log-mel features for Smart Turn v3.
#
# Adapted from pipecat (src/pipecat/audio/turn/smart_turn/_whisper_features.py),
# Copyright (c) 2024-2026, Daily, BSD 2-Clause License, which in turn derives from
# Hugging Face Transformers (Apache License 2.0), Copyright 2022-2023 The HuggingFace Inc. team.
#
# Replicates transformers.WhisperFeatureExtractor(chunk_length=8)(..., do_normalize=True)
# so we don't need to install `transformers`.
#
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

N_FFT = 400
HOP = 160
N_MELS = 80
SR = 16000
N_SAMPLES = SR * 8


def _hz_to_mel(freq):
    freq = np.atleast_1d(np.asarray(freq, dtype=np.float64))
    mels = 3.0 * freq / 200.0
    log_region = freq >= 1000.0
    mels[log_region] = 15.0 + np.log(freq[log_region] / 1000.0) * (27.0 / np.log(6.4))
    return mels


def _mel_to_hz(mels):
    mels = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    freq = 200.0 * mels / 3.0
    log_region = mels >= 15.0
    freq[log_region] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (mels[log_region] - 15.0))
    return freq


def _mel_filters() -> np.ndarray:
    mel_freqs = np.linspace(_hz_to_mel(0.0)[0], _hz_to_mel(SR / 2)[0], N_MELS + 2)
    filter_freqs = _mel_to_hz(mel_freqs)
    fft_freqs = np.linspace(0, SR // 2, N_FFT // 2 + 1)
    diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    filters *= (2.0 / (filter_freqs[2:N_MELS + 2] - filter_freqs[:N_MELS]))[None, :]
    return filters  # (201, 80)


_WINDOW = np.hanning(N_FFT + 1)[:-1]
_FILTERS = _mel_filters()


def log_mel_8s(audio: np.ndarray) -> np.ndarray:
    """audio: float 16 kHz, exactly 8 s (128000 samples) -> (80, 800) float32."""
    x = np.asarray(audio, dtype=np.float32)
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    padded = np.pad(x.astype(np.float64), (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = sliding_window_view(padded, N_FFT)[::HOP]
    power = (np.abs(np.fft.rfft(frames * _WINDOW, axis=-1)) ** 2).T
    log_spec = np.log10(np.maximum(1e-10, _FILTERS.T @ power))[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)
