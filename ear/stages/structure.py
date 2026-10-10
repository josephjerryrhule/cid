"""Stage 3a: energy and band curves, sections and key."""
from __future__ import annotations

import numpy as np

from core.audio import ANALYSIS_SR, CURVE_HOP, CURVE_HZ, load_mono, norm01
from core.context import Context

VERSION = 1
DEPS = ("ingest", "rhythm", "stems", "classify")
CURVES_FILE = "curves_structure.npz"
SECTION_TARGET_S = 20
MIN_SECTION_S = 4.0
BANDS = {"low": (20, 250), "mid": (250, 4000), "high": (4000, 12000)}

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def energy_curves(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=2 * CURVE_HOP, hop_length=CURVE_HOP)[0]
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=CURVE_HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    out = {"rms": rms, "energy": norm01(rms, floor=0.0)}
    for name, (lo, hi) in BANDS.items():
        out[name] = norm01(np.sqrt(S[(freqs >= lo) & (freqs < hi)].sum(axis=0)), floor=0.0)
    n = min(len(v) for v in out.values())
    return {k: v[:n] for k, v in out.items()}


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)


def _merge_short(starts: list[float], duration: float) -> list[float]:
    out = [0.0]
    for s in starts[1:]:
        if s - out[-1] >= MIN_SECTION_S:
            out.append(s)
    if len(out) > 1 and duration - out[-1] < MIN_SECTION_S:
        out.pop()
    return out


def find_sections(y, sr, downbeats, duration: float, k: int | None = None) -> list[tuple[float, float]]:
    import librosa

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)
    n = min(chroma.shape[1], mfcc.shape[1], rms.shape[1])
    feat = np.vstack([_z(chroma[:, :n]), _z(mfcc[:, :n]), 2 * _z(rms[:, :n])])
    grid = np.asarray(downbeats, dtype=float) if len(downbeats) >= 4 else np.arange(1.0, duration, 1.0)
    frames = librosa.time_to_frames(grid, sr=sr, hop_length=hop)
    bounds = librosa.util.fix_frames(frames[frames < n], x_min=0, x_max=n, pad=True)
    synced = librosa.util.sync(feat, bounds, aggregate=np.median, pad=False)
    seg_starts = librosa.frames_to_time(bounds[:-1], sr=sr, hop_length=hop)
    k = k or int(np.clip(round(duration / SECTION_TARGET_S), 2, 10))
    k = max(1, min(k, synced.shape[1]))
    idx = librosa.segment.agglomerative(synced, k)
    starts = _merge_short(sorted({round(float(seg_starts[i]), 3) for i in idx} | {0.0}), duration)
    ends = starts[1:] + [round(float(duration), 3)]
    return list(zip(starts, ends))


def auto_labels(energies: list[float]) -> list[str]:
    if not energies:
        return []
    hi = max(energies) or 1.0
    n = len(energies)
    labels = []
    for i, e in enumerate(energies):
        if n > 1 and i == 0 and e < 0.6 * hi:
            labels.append("intro")
        elif n > 2 and i == n - 1 and e < 0.6 * hi:
            labels.append("outro")
        elif e >= 0.8 * hi:
            labels.append("peak")
        elif i + 1 < n and energies[i + 1] >= 0.8 * hi and e < energies[i + 1]:
            labels.append("build")
        else:
            labels.append("body")
    return labels


def estimate_key(y: np.ndarray, sr: int) -> dict:
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    scores = []
    for mode, profile in (("major", MAJOR), ("minor", MINOR)):
        for i in range(12):
            r = float(np.corrcoef(chroma, np.roll(profile, i))[0, 1])
            scores.append((r, NOTES[i], mode))
    scores.sort(reverse=True)
    best = scores[0]
    return {"tonic": best[1], "mode": best[2], "confidence": round(float(np.clip(best[0], 0, 1)), 3)}


def run(ctx: Context) -> dict:
    import librosa

    ingest = ctx.data("ingest")
    rhythm = ctx.data("rhythm") or {}
    stems = ctx.data("stems")
    kind = (ctx.data("classify") or {}).get("kind", "music")
    duration = float(ingest["duration"])
    y = load_mono(ctx.path("master.wav"))
    curves = energy_curves(y, ANALYSIS_SR)
    n = len(curves["energy"])
    if stems:
        for name, fname in stems["stems"].items():
            ys = load_mono(ctx.path(fname))
            r = librosa.feature.rms(y=ys, frame_length=2 * CURVE_HOP, hop_length=CURVE_HOP)[0]
            curves[f"stem_{name}"] = _fit(norm01(r, floor=0.0), n)
    np.savez_compressed(ctx.path(CURVES_FILE), **curves)
    spans = find_sections(y, ANALYSIS_SR, rhythm.get("downbeats", []), duration)
    energies = []
    for a, b in spans:
        i, j = int(a * CURVE_HZ), max(int(a * CURVE_HZ) + 1, int(b * CURVE_HZ))
        energies.append(float(curves["energy"][i:j].mean()))
    labels = auto_labels(energies)
    sections = [
        {"id": f"s{i + 1}", "label": lab, "start": a, "end": b, "energy": round(e, 3)}
        for i, ((a, b), e, lab) in enumerate(zip(spans, energies, labels))
    ]
    key = estimate_key(y, ANALYSIS_SR) if kind in ("music", "mixed") else None
    return {"curves_file": CURVES_FILE, "curve_hz": CURVE_HZ, "sections": sections, "key": key}
