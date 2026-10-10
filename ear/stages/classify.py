"""Labels the input as music, speech, sfx or mixed so later stages and the report focus correctly."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from core.audio import ANALYSIS_SR, CURVE_HOP, CURVE_HZ, load_mono
from core.context import Context

VERSION = 1
DEPS = ("ingest", "stems", "rhythm")
RHYTHMIC_CONF = 0.6
SPEECH_MOD = 0.45
HARMONIC = 0.6


def speech_modulation(y: np.ndarray, sr: int) -> float:
    """Share of the 300 to 3400 Hz envelope's modulation energy at syllable rates (2 to 8 Hz)."""
    import librosa

    band = sosfilt(butter(4, [300, 3400], btype="band", fs=sr, output="sos"), y)
    env = librosa.feature.rms(y=band, frame_length=2 * CURVE_HOP, hop_length=CURVE_HOP)[0]
    env = env - env.mean()
    if not np.any(env):
        return 0.0
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    f = np.fft.rfftfreq(len(env), 1 / CURVE_HZ)
    total = spec[(f >= 0.5) & (f <= 20)].sum()
    return float(spec[(f >= 2) & (f <= 8)].sum() / total) if total > 0 else 0.0


def harmonic_ratio(y: np.ndarray, sr: int, max_s: float = 60.0) -> float:
    import librosa

    S = np.abs(librosa.stft(y[: int(max_s * sr)]))
    H, P = librosa.decompose.hpss(S)
    total = H.sum() + P.sum()
    return float(H.sum() / total) if total > 0 else 0.0


def classify(beat_conf: float, vocal_ratio: float | None, speech_mod: float, harmonic: float, has_video: bool) -> str:
    if beat_conf >= RHYTHMIC_CONF:
        if vocal_ratio is not None and vocal_ratio >= 0.15 and has_video:
            return "mixed"
        return "music"
    if vocal_ratio is not None:
        if vocal_ratio >= 0.5 or (vocal_ratio >= 0.25 and speech_mod >= SPEECH_MOD):
            return "speech"
    elif speech_mod >= SPEECH_MOD:
        return "speech"
    return "music" if harmonic >= HARMONIC else "sfx"


def _vocal_ratio(ctx: Context, stems: dict) -> float:
    energy = {}
    for name, fname in stems["stems"].items():
        y = load_mono(ctx.path(fname))
        energy[name] = float(np.sum(y.astype(float) ** 2))
    total = sum(energy.values())
    return energy.get("vocals", 0.0) / total if total > 0 else 0.0


def run(ctx: Context) -> dict:
    ingest = ctx.data("ingest") or {}
    rhythm = ctx.data("rhythm")
    stems = ctx.data("stems")
    y = load_mono(ctx.path("master.wav"))
    beat_conf = float(rhythm["confidence"]) if rhythm else 0.0
    vocal_ratio = _vocal_ratio(ctx, stems) if stems else None
    mod = speech_modulation(y, ANALYSIS_SR)
    harm = harmonic_ratio(y, ANALYSIS_SR)
    kind = classify(beat_conf, vocal_ratio, mod, harm, bool(ingest.get("has_video")))
    return {
        "kind": kind,
        "features": {
            "beat_conf": round(beat_conf, 3),
            "vocal_ratio": None if vocal_ratio is None else round(vocal_ratio, 3),
            "speech_mod": round(mod, 3),
            "harmonic": round(harm, 3),
        },
    }
