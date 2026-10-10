"""Stage 3b: drops, impacts, risers, silences, vocal entries, cut and hold points.

Every threshold is relative to the track itself, so quiet masters behave like loud ones.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from core.audio import ANALYSIS_SR, CURVE_HOP, load_mono, onset_env
from core.context import Context
from core.pipeline import StageUnavailable

VERSION = 1
DEPS = ("ingest", "rhythm", "structure", "hits", "classify")
SILENCE_REL_DB = -50.0
SILENCE_MIN_S = 0.05
DROP_JUMP = 0.25
DROP_MIN_AFTER = 0.4
DROP_LOW_JUMP = 0.2
WIN_S = 2.0


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(n) / n, mode="same")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _dedupe(events: list[dict], min_gap: float) -> list[dict]:
    kept: list[dict] = []
    for e in sorted(events, key=lambda e: -e["strength"]):
        if all(abs(e["t"] - k["t"]) >= min_gap for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e["t"])


def _window_means(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the w samples before and after each index (NaN where the window falls off the edge)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    idx = np.arange(n)
    before = np.full(n, np.nan)
    after = np.full(n, np.nan)
    ok_b = idx >= w
    ok_a = idx + w <= n
    before[ok_b] = (c[idx[ok_b]] - c[idx[ok_b] - w]) / w
    after[ok_a] = (c[idx[ok_a] + w] - c[idx[ok_a]]) / w
    return before, after


def detect_silences(rms: np.ndarray, hz: int, min_s: float = SILENCE_MIN_S) -> list[dict]:
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    ref = float(np.percentile(db, 95))
    quiet = db < ref + SILENCE_REL_DB
    out = []
    for a, b in _runs(quiet):
        if (b - a) / hz >= min_s:
            out.append({"t": round(a / hz, 3), "end": round(b / hz, 3), "type": "silence",
                        "strength": round(min(1.0, (b - a) / hz), 3)})
    return out


def detect_drops(energy, low, hz: int, candidates, win_s: float = WIN_S) -> list[dict]:
    w = int(win_s * hz)
    out = []
    for t in candidates:
        i = int(round(t * hz))
        if i - w < 0 or i + w > len(energy):
            continue
        jump = energy[i : i + w].mean() - energy[i - w : i].mean()
        low_jump = low[i : i + w].mean() - low[i - w : i].mean()
        if jump >= DROP_JUMP and energy[i : i + w].mean() >= DROP_MIN_AFTER and low_jump >= DROP_LOW_JUMP:
            out.append({"t": round(float(t), 4), "type": "drop", "strength": round(float(min(1.0, 2 * jump)), 3)})
    return _dedupe(out, 4.0)


def step_candidates(energy: np.ndarray, hz: int, win_s: float = 0.5) -> list[float]:
    before, after = _window_means(np.asarray(energy, dtype=float), int(win_s * hz))
    diff = np.nan_to_num(after - before)
    peaks, _ = find_peaks(diff, height=0.2, distance=max(1, 2 * hz))
    return [round(p / hz, 3) for p in peaks]


def detect_impacts(env: np.ndarray, hz: int, exclude, top: int = 30, min_gap: float = 1.0) -> list[dict]:
    if env.size == 0 or env.max() <= 0:
        return []
    thr = max(float(np.percentile(env, 98)), 3 * float(np.median(env)) + 1e-9)
    peaks, _ = find_peaks(env, height=thr, distance=max(1, int(min_gap * hz)))
    if peaks.size == 0:
        return []
    top_val = float(env[peaks].max())
    events = [
        {"t": round(p / hz, 4), "type": "impact", "strength": round(float(env[p] / top_val), 3)}
        for p in peaks
        if all(abs(p / hz - x) > 0.2 for x in exclude)
    ]
    return sorted(sorted(events, key=lambda e: -e["strength"])[:top], key=lambda e: e["t"])


def detect_risers(high: np.ndarray, hz: int, targets, min_s=2.0, max_s=16.0, min_rise=0.15, min_corr=0.85) -> list[dict]:
    """Best window scores rise * correlation: inside a ramp the rise grows with length, and once the
    window reaches back past the ramp's start the rise stops growing while correlation falls."""
    smooth = _smooth(high, int(0.25 * hz))
    half = int(0.5 * hz)
    unique = []
    for t in sorted(targets):
        if all(abs(t - u) > 0.5 for u in unique):
            unique.append(t)
    out = []
    for t in unique:
        end = int(round(t * hz)) - 1
        best = None
        for length in np.arange(min_s, max_s + 1e-9, 0.5):
            start = end - int(length * hz)
            if start < 0:
                break
            seg = smooth[start:end]
            rise = seg[-half:].mean() - seg[:half].mean()
            if rise < min_rise or np.std(seg) == 0:
                continue
            corr = float(np.corrcoef(np.arange(len(seg)), seg)[0, 1])
            if corr >= min_corr and (best is None or rise * corr > best[2]):
                best = (start, rise, rise * corr)
        if best:
            out.append({"t": round(best[0] / hz, 3), "end": round(float(t), 3), "type": "riser",
                        "strength": round(float(min(1.0, best[1])), 3)})
    return out


def detect_vocal_changes(curve: np.ndarray, hz: int, on: float = 0.15, min_gap_s: float = 2.0) -> list[dict]:
    active = _smooth(curve, int(0.3 * hz)) > on
    merged: list[tuple[int, int]] = []
    for a, b in _runs(active):
        if merged and (a - merged[-1][1]) / hz < min_gap_s:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    events = []
    for a, b in merged:
        if (b - a) / hz < 0.5:
            continue
        events.append({"t": round(a / hz, 3), "type": "vocal_in", "strength": 1.0})
        events.append({"t": round(b / hz, 3), "type": "vocal_out", "strength": 1.0})
    return events


def rank_cuts(downbeats, section_starts, energy: np.ndarray, hz: int, duration: float) -> list[dict]:
    downbeats = list(downbeats)
    if not downbeats:
        return []
    scored = []
    for i, t in enumerate(downbeats):
        phrase = 1.0 if i % 4 == 0 else 0.0
        boundary = 1.0 if any(abs(t - s) < 0.15 for s in section_starts) else 0.0
        j, w = int(t * hz), int(hz)
        change = abs(energy[j : j + w].mean() - energy[j - w : j].mean()) if j >= w and j + w <= len(energy) else 0.0
        score = 0.4 * phrase + 0.4 * boundary + 0.2 * min(1.0, 4 * change)
        scored.append({"t": round(float(t), 4), "type": "cut", "strength": round(score, 3)})
    limit = max(4, int(duration / 15))
    min_gap = 2 * (downbeats[1] - downbeats[0]) if len(downbeats) > 1 else 4.0
    kept: list[dict] = []
    for e in sorted(scored, key=lambda e: (-e["strength"], e["t"])):
        if len(kept) >= limit:
            break
        if all(abs(e["t"] - k["t"]) >= min_gap - 1e-6 for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e["t"])


def detect_holds(sections, energy: np.ndarray, hz: int, busy_times, min_s=4.0, max_std=0.06) -> list[dict]:
    out = []
    for s in sections:
        a, b = s["start"], s["end"]
        if b - a < min_s:
            continue
        seg = energy[int(a * hz) : int(b * hz)]
        if seg.size and seg.std() < max_std and not any(a < t < b for t in busy_times):
            out.append({"t": a, "end": b, "type": "hold", "strength": round(float(1 - seg.std() / max_std), 3)})
    return out


def run(ctx: Context) -> dict:
    ingest = ctx.data("ingest")
    rhythm = ctx.data("rhythm") or {}
    structure = ctx.data("structure")
    kind = (ctx.data("classify") or {}).get("kind", "music")
    if not structure:
        raise StageUnavailable("structure stage unavailable")
    c = np.load(ctx.path(structure["curves_file"]))
    hz = structure["curve_hz"]
    duration = float(ingest["duration"])
    energy, low, high, rms = c["energy"], c["low"], c["high"], c["rms"]
    downbeats = rhythm.get("downbeats", [])
    section_starts = [s["start"] for s in structure["sections"][1:]]
    candidates = sorted(set(downbeats) | set(section_starts)) if downbeats else step_candidates(energy, hz)
    drops = detect_drops(energy, low, hz, candidates)
    env = onset_env(load_mono(ctx.path("master.wav")), ANALYSIS_SR, CURVE_HOP)
    impacts = detect_impacts(env, hz, [d["t"] for d in drops], top=10 if kind == "music" else 30)
    risers = detect_risers(high, hz, [d["t"] for d in drops] + section_starts)
    silences = detect_silences(rms, hz)
    vocal = detect_vocal_changes(c["stem_vocals"], hz) if "stem_vocals" in c.files else []
    cuts = rank_cuts(downbeats, section_starts, energy, hz, duration)
    holds = detect_holds(structure["sections"], energy, hz, [e["t"] for e in drops + impacts])
    events = drops + impacts + risers + silences + vocal + cuts + holds
    return {"moments": sorted(events, key=lambda e: (e["t"], e["type"]))}
