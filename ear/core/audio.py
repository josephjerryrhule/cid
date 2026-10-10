"""Shared audio helpers. Analysis runs at 24 kHz so curves land on an exact 100 Hz grid."""
from __future__ import annotations

import numpy as np

ANALYSIS_SR = 24000
CURVE_HOP = 240
CURVE_HZ = ANALYSIS_SR // CURVE_HOP  # 100 samples per second


def load_mono(path, sr: int = ANALYSIS_SR) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def norm01(x, pct: float = 99.0, floor: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo = float(x.min()) if floor is None else float(floor)
    hi = float(np.percentile(x, pct))
    if hi - lo <= 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def resample_curve(values, src_hz: float, fps: int, duration: float, offset: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = int(round(duration * fps))
    if values.size == 0:
        return np.zeros(n)
    src_t = offset + np.arange(values.size) / src_hz
    return np.interp(np.arange(n) / fps, src_t, values)


def onset_env(y: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Onset strength with a short window so peaks sit within a few ms of the attack."""
    import librosa

    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_fft=512)
