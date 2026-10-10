"""Stage 2: beats, downbeats, tempo and meter, refined to the nearest onset peak."""
from __future__ import annotations

from collections import Counter

import numpy as np

from core.audio import ANALYSIS_SR, load_mono, onset_env
from core.context import Context

VERSION = 1
DEPS = ("ingest",)
REFINE_HOP = 128


def _librosa_beats(path) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    y = load_mono(path)
    _, frames = librosa.beat.beat_track(y=y, sr=ANALYSIS_SR, units="frames")
    beats = librosa.frames_to_time(frames, sr=ANALYSIS_SR)
    return beats, beats[::4]


def detect(path) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from beat_this.inference import File2Beats
    except ImportError:
        beats, downs = _librosa_beats(path)
        return beats, downs, "librosa"
    f2b = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = f2b(str(path))
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float), "beat_this"


def refine(beats: np.ndarray, env: np.ndarray, times: np.ndarray, window: float = 0.03) -> np.ndarray:
    """Move each beat to the strongest onset within +/- window, when that onset is clearly a peak."""
    out = np.asarray(beats, dtype=float).copy()
    floor = 1.5 * float(np.median(env)) + 1e-9
    for i, b in enumerate(out):
        lo, hi = np.searchsorted(times, [b - window, b + window])
        if hi - lo < 2:
            continue
        seg = env[lo:hi]
        if seg.max() >= floor:
            out[i] = times[lo + int(seg.argmax())]
    return out


def number_beats(beats, downbeats, tol: float = 0.035) -> list[tuple[int, int]]:
    db = np.sort(np.asarray(downbeats, dtype=float))
    out, bar, pos = [], 0, 0
    for t in beats:
        j = int(np.searchsorted(db, t))
        near = any(0 <= k < len(db) and abs(db[k] - t) < tol for k in (j - 1, j))
        if near:
            bar, pos = bar + 1, 1
        else:
            pos += 1
        out.append((bar, pos))
    return out


def meter(numbered: list[tuple[int, int]]) -> str:
    per_bar: dict[int, int] = {}
    for bar, pos in numbered:
        if bar >= 1:
            per_bar[bar] = max(per_bar.get(bar, 0), pos)
    complete = [n for b, n in per_bar.items() if b != max(per_bar, default=0)]
    if not complete:
        return "unknown"
    return f"{Counter(complete).most_common(1)[0][0]}/4"


def confidence(beats) -> float:
    beats = np.asarray(beats, dtype=float)
    if beats.size < 8:
        return 0.0
    ibi = np.diff(beats)
    cv = float(np.std(ibi) / np.mean(ibi))
    return float(np.clip(1 - 4 * cv, 0, 1))


def tempo_changes(beats, bpm: float, window: int = 8, threshold: float = 0.04) -> list[dict]:
    """A change counts only once the new tempo holds: windows straddling a switch report an
    in-between median and must not be recorded as their own tempo."""
    beats = np.asarray(beats, dtype=float)
    if beats.size < window * 2 or bpm <= 0:
        return []
    ibi = np.diff(beats)
    local = np.array([60.0 / float(np.median(ibi[i : i + window])) for i in range(len(ibi) - window + 1)])
    changes, current = [], bpm
    for i, b in enumerate(local):
        settled = local[i : i + window // 2]
        if abs(b - current) / current > threshold and np.all(np.abs(settled - b) / b <= threshold / 2):
            changes.append({"t": round(float(beats[i]), 3), "bpm": round(float(b), 1)})
            current = b
    return changes


def run(ctx: Context) -> dict:
    import librosa

    path = ctx.path("master.wav")
    beats, downbeats, source = detect(path)
    y = load_mono(path)
    env = onset_env(y, ANALYSIS_SR, REFINE_HOP)
    times = librosa.times_like(env, sr=ANALYSIS_SR, hop_length=REFINE_HOP)
    beats = refine(beats, env, times)
    if beats.size and downbeats.size:
        downbeats = np.array([beats[np.argmin(np.abs(beats - d))] for d in downbeats])
    numbered = number_beats(beats, downbeats)
    peak = float(np.percentile(env, 99)) or 1.0
    idx = np.clip(np.searchsorted(times, beats), 0, len(env) - 1)
    strengths = np.clip(env[idx] / peak, 0, 1)
    bpm = 60.0 / float(np.median(np.diff(beats))) if beats.size > 1 else 0.0
    conf = confidence(beats)
    return {
        "source": source,
        "bpm": round(bpm, 2),
        "confidence": round(conf, 3),
        "low_confidence": conf < 0.5,
        "meter": meter(numbered),
        "changes": tempo_changes(beats, bpm),
        "beats": [
            {"t": round(float(t), 4), "bar": b, "beat": p, "strength": round(float(s), 3)}
            for t, (b, p), s in zip(beats, numbered, strengths)
        ],
        "downbeats": [round(float(d), 4) for d in downbeats],
    }
