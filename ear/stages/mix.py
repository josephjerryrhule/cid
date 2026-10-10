"""Stage 4: loudness, true peak, dynamics, spectral balance, stereo image and clipping.

Band flags compare equal-width octave bands: pink noise has equal power per octave,
so a balanced mix reads roughly flat and a bump or dip shows up directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt, welch

from core.context import Context
from core.media import ffmpeg
from core.pipeline import Fatal

VERSION = 1
DEPS = ("ingest",)
ST_WINDOW_S = 3.0
ST_HZ = 10
OCTAVES = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def _k_weight(y: np.ndarray, sr: int) -> np.ndarray:
    meter = pyln.Meter(sr)
    out = np.array(y, dtype=float, copy=True)
    for stage in meter._filters.values():  # pyloudnorm's BS.1770 K-weighting stages
        for ch in range(out.shape[1]):
            out[:, ch] = stage.apply_filter(out[:, ch])
    return out


def short_term(y: np.ndarray, sr: int) -> np.ndarray:
    sq = (_k_weight(y, sr) ** 2).sum(axis=1)
    c = np.concatenate([[0.0], np.cumsum(sq)])
    win = min(int(ST_WINDOW_S * sr), len(sq))
    hop = int(sr / ST_HZ)
    starts = np.arange(0, len(sq) - win + 1, hop)
    ms = (c[starts + win] - c[starts]) / win
    return -0.691 + 10 * np.log10(np.maximum(ms, 1e-12))


def loudness_range(st: np.ndarray) -> float:
    st = st[st > -70]
    if st.size < 2:
        return 0.0
    rel = 10 * np.log10(np.mean(10 ** (st / 10))) - 20
    st = st[st > rel]
    return float(np.percentile(st, 95) - np.percentile(st, 10)) if st.size > 1 else 0.0


def true_peak_db(y: np.ndarray) -> float:
    up = resample_poly(y.astype(np.float32), 4, 1, axis=0)
    return float(20 * np.log10(max(float(np.max(np.abs(up))), 1e-12)))


def clipped_samples(y: np.ndarray, thr: float = 0.999, min_run: int = 3) -> int:
    total = 0
    for ch in y.T:
        hot = np.abs(ch) >= thr
        d = np.diff(np.concatenate([[0], hot.astype(np.int8), [0]]))
        lens = np.where(d == -1)[0] - np.where(d == 1)[0]
        total += int(lens[lens >= min_run].sum())
    return total


def _band_db(f, p, lo, hi) -> float:
    return float(10 * np.log10(p[(f >= lo) & (f < hi)].sum() + 1e-20))


def measure(y: np.ndarray, sr: int) -> dict:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    integrated = float(pyln.Meter(sr).integrated_loudness(y))
    st = short_term(y, sr)
    lra = loudness_range(st)
    tp = true_peak_db(y)
    clipped = clipped_samples(y)
    mid = y.mean(axis=1)
    f, p = welch(mid, fs=sr, nperseg=8192)
    bands = {f"{c:g} Hz": round(_band_db(f, p, c / np.sqrt(2), c * np.sqrt(2)), 2) for c in OCTAVES}
    mono = bool(np.max(np.abs(y[:, 0] - y[:, 1])) < 1e-6)
    if mono:
        width, low_corr, corr = 0.0, 1.0, 1.0
    else:
        side = (y[:, 0] - y[:, 1]) / 2
        width = float(np.sqrt(np.mean(side**2)) / (np.sqrt(np.mean(mid**2)) + 1e-12))
        sos = butter(4, 120, btype="low", fs=sr, output="sos")
        lows = sosfilt(sos, y, axis=0)
        low_corr = float(np.corrcoef(lows[:, 0], lows[:, 1])[0, 1])
        corr = float(np.corrcoef(y[:, 0], y[:, 1])[0, 1])
    flags = []
    if clipped:
        flags.append(f"clipping: {clipped} samples at full scale")
    if tp > -1.0:
        flags.append(f"true peak above -1 dBTP ({tp:.1f})")
    if 0 < lra < 3:
        flags.append(f"very compressed dynamics (LRA {lra:.1f} LU)")
    mud = _band_db(f, p, 200, 400) - np.mean([_band_db(f, p, 100, 200), _band_db(f, p, 400, 800)])
    if mud > 3:
        flags.append(f"low-mid build-up 200 to 400 Hz (+{mud:.1f} dB)")
    harsh = _band_db(f, p, 2500, 5000) - np.mean([_band_db(f, p, 1250, 2500), _band_db(f, p, 5000, 10000)])
    if harsh > 3:
        flags.append(f"harsh upper mids 2.5 to 5 kHz (+{harsh:.1f} dB)")
    if _band_db(f, p, 8000, 16000) - _band_db(f, p, 1000, 2000) < -18:
        flags.append("dull top end above 8 kHz")
    if _band_db(f, p, 40, 120) - _band_db(f, p, 250, 750) < -10:
        flags.append("thin low end below 120 Hz")
    if not mono and low_corr < 0.8:
        flags.append(f"low end not mono below 120 Hz (correlation {low_corr:.2f})")
    if not mono and corr < 0:
        flags.append("phase problem: channels largely out of phase")
    return {
        "lufs": round(integrated, 2),
        "true_peak": round(tp, 2),
        "lra": round(lra, 2),
        "clipped_samples": clipped,
        "width": round(width, 3),
        "low_corr": round(low_corr, 3),
        "mono": mono,
        "bands": bands,
        "flags": flags,
        "short_term": {"hz": ST_HZ, "offset": ST_WINDOW_S / 2, "values": [round(float(v), 2) for v in st]},
    }


def _compare(own: dict, ref: dict) -> list[str]:
    deltas = []
    for band, level in own["bands"].items():
        d = (level - own["lufs"]) - (ref["bands"][band] - ref["lufs"])
        if abs(d) > 3:
            deltas.append(f"{d:+.1f} dB at {band} vs reference (loudness matched)")
    for key, unit in (("lufs", "LU"), ("lra", "LU"), ("true_peak", "dB")):
        d = own[key] - ref[key]
        if abs(d) > 1:
            deltas.append(f"{key} {d:+.1f} {unit} vs reference")
    return deltas


def _measure_ref(ctx: Context, ref_path: Path, own: dict) -> dict:
    out = ctx.path("ref_decoded.wav")
    try:
        ffmpeg(["-i", str(ref_path), "-map", "0:a:0", "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", str(out)])
    except Fatal as e:
        return {"file": ref_path.name, "error": str(e)}
    y, sr = sf.read(str(out), dtype="float32", always_2d=True)
    ref = measure(y, sr)
    return {
        "file": ref_path.name,
        "lufs": ref["lufs"],
        "true_peak": ref["true_peak"],
        "lra": ref["lra"],
        "width": ref["width"],
        "bands": ref["bands"],
        "deltas": _compare(own, ref),
    }


def run(ctx: Context) -> dict:
    y, sr = sf.read(str(ctx.path("master.wav")), dtype="float32", always_2d=True)
    out = measure(y, sr)
    if ctx.options.get("ref"):
        out["ref"] = _measure_ref(ctx, Path(ctx.options["ref"]), out)
    return out
