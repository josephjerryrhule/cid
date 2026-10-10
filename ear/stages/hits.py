"""Per-instrument hits: kick, snare and hat from the drums stem (or the mix), bass hits and vocal entries."""
from __future__ import annotations

import numpy as np

from core.audio import ANALYSIS_SR, load_mono
from core.context import Context

VERSION = 1
DEPS = ("ingest", "stems")
HOP = 128
N_FFT = 1024
# Snare band stops at 3 kHz so hat leakage (hats sit above 5 kHz) stays under the gate.
DRUM_BANDS = {"kick": (20, 100), "snare": (250, 3000), "hat": (5000, 12000)}
BASS_BAND = (20, 400)
# Onset-strength peaks land about 10 ms after the attack with a 1024-sample centred window.
ONSET_BIAS_S = 0.01
GATE = 1e-3  # ignore onsets where the band is more than 30 dB below its loudest frame


def band_onsets(y: np.ndarray, sr: int, lo: float, hi: float) -> list[dict]:
    import librosa

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band = S[(freqs >= lo) & (freqs < hi)]
    if band.size == 0:
        return []
    level = band.sum(axis=0)
    if level.max() <= 0:
        return []
    env = librosa.onset.onset_strength(S=librosa.power_to_db(band + 1e-10), sr=sr, hop_length=HOP)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=HOP, units="frames", delta=0.1, wait=int(0.06 * sr / HOP)
    )
    frames = [f for f in frames if level[min(f + 1, len(level) - 1)] >= GATE * level.max()]
    if not frames:
        return []
    peak = float(np.percentile(env[frames], 95)) or 1.0
    return [
        {
            "t": round(max(0.0, float(librosa.frames_to_time(f, sr=sr, hop_length=HOP)) - ONSET_BIAS_S), 4),
            "v": round(float(min(1.0, env[f] / peak)), 3),
        }
        for f in frames
    ]


def vocal_entries(y: np.ndarray, sr: int, gap_s: float = 0.3) -> list[dict]:
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    if rms.size == 0 or rms.max() <= 0:
        return []
    ref = float(np.percentile(rms, 99))
    on = rms > 0.1 * ref
    frame_s = HOP / sr
    entries, off_run = [], gap_s
    for i, active in enumerate(on):
        if active and off_run >= gap_s:
            entries.append({"t": round(i * frame_s, 4), "v": round(float(min(1.0, rms[i] / ref)), 3)})
        off_run = 0.0 if active else off_run + frame_s
    return entries


def run(ctx: Context) -> dict:
    stems = ctx.data("stems")
    if stems:
        drums = load_mono(ctx.path(stems["stems"]["drums"]))
    else:
        drums = load_mono(ctx.path("master.wav"))
    result = {name: band_onsets(drums, ANALYSIS_SR, lo, hi) for name, (lo, hi) in DRUM_BANDS.items()}
    if stems:
        result["bass"] = band_onsets(load_mono(ctx.path(stems["stems"]["bass"])), ANALYSIS_SR, *BASS_BAND)
        result["vocal"] = vocal_entries(load_mono(ctx.path(stems["stems"]["vocals"])), ANALYSIS_SR)
    else:
        result["bass"], result["vocal"] = [], []
    return {"from": "stems" if stems else "mix", "hits": result}
