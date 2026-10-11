"""Stage: melody notes per stem with Spotify basic-pitch, isolated through uvx."""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from core.context import Context
from core.pipeline import StageUnavailable

VERSION = 2
DEPS = ("ingest", "stems")
STEMS = ("vocals", "bass", "other")
RUNNER = Path(__file__).resolve().parent.parent / "tools" / "basic_pitch_runner.py"
BASIC_PITCH_CMD = [
    "uvx", "--python", "3.12",
    "--with", "basic-pitch[onnx]==0.3.0", "--with", "setuptools<70", "--with", "scipy<1.13", "--with", "numpy<2",
    "python",
]
MIN_NOTE_S = 0.06
BASS_MAX_PITCH = 60
VOCAL_RANGE = (36, 84)  # C2 to C6
GHOST_INTERVALS = (12, 19, 24)  # octave, twelfth, two octaves: overtones read as notes
BLEED_RATIO = 0.1  # a note where the stem is under 10% of its loud level is bleed
RMS_HZ = 100
MAX_HOOK_LEAP = 12
SILENT_RMS = 1e-3
TIMEOUT_S = 15 * 60


def _name(pitch: int) -> str:
    import librosa

    return librosa.midi_to_note(int(pitch), unicode=False)


def _ghost(n: dict, others: list[dict]) -> bool:
    dur = n["end"] - n["t"]
    for m in others:
        if m is n or m["pitch"] >= n["pitch"] or (n["pitch"] - m["pitch"]) not in GHOST_INTERVALS:
            continue
        overlap = min(n["end"], m["end"]) - max(n["t"], m["t"])
        if overlap >= 0.5 * dur and abs(n["t"] - m["t"]) <= 0.05 and m["velocity"] >= n["velocity"]:
            return True
    return False


def clean_notes(raw: list[dict], stem: str, rms: np.ndarray | None = None, rms_hz: int = RMS_HZ) -> list[dict]:
    notes = [n for n in sorted(raw, key=lambda n: n["t"]) if n["end"] - n["t"] >= MIN_NOTE_S]
    if stem == "bass":
        notes = [n for n in notes if n["pitch"] <= BASS_MAX_PITCH]
    if stem == "vocals":
        notes = [n for n in notes if VOCAL_RANGE[0] <= n["pitch"] <= VOCAL_RANGE[1]]
    if rms is not None and len(rms):
        loud = float(np.percentile(rms, 95)) or 1.0
        def level(n):
            a, b = int(n["t"] * rms_hz), max(int(n["t"] * rms_hz) + 1, int(n["end"] * rms_hz))
            seg = rms[a:b]
            return float(seg.mean()) if seg.size else 0.0
        notes = [n for n in notes if level(n) >= BLEED_RATIO * loud]
    if stem in ("other", "vocals"):
        notes = [n for n in notes if not _ghost(n, notes)]
    return [{"t": round(n["t"], 3), "end": round(n["end"], 3), "pitch": int(n["pitch"]),
             "name": _name(n["pitch"]), "velocity": round(float(min(1.0, n["velocity"])), 3)} for n in notes]


def skyline(notes: list[dict], window: float = 0.03) -> list[dict]:
    """Top line of a polyphonic part: the highest note of each group of near-simultaneous onsets."""
    out: list[dict] = []
    for n in sorted(notes, key=lambda n: (n["t"], -n["pitch"])):
        if out and n["t"] - out[-1]["t"] <= window:
            if n["pitch"] > out[-1]["pitch"]:
                out[-1] = n
            continue
        out.append(n)
    return out


def find_hook(notes: list[dict], n: int = 4, min_times: int = 3) -> dict | None:
    """Most repeated n-note phrase, transposition invariant (by intervals), non-overlapping."""
    if len(notes) < n * min_times:
        return None
    pitches = [x["pitch"] for x in notes]
    starts = defaultdict(list)
    for i in range(len(notes) - n + 1):
        key = tuple(pitches[i + k + 1] - pitches[i + k] for k in range(n - 1))
        if any(abs(step) > MAX_HOOK_LEAP for step in key):
            continue  # leaps over an octave are overtone artefacts, not a sung or played line
        if not any(key):
            continue  # one repeated pitch is a pedal note, not a hook
        if starts[key] and i < starts[key][-1] + n:
            continue
        starts[key].append(i)
    if not starts:
        return None
    key, idx = max(starts.items(), key=lambda kv: (len(kv[1]), -kv[1][0]))
    if len(idx) < min_times:
        return None
    first = idx[0]
    return {"pattern": [notes[first + k]["name"] for k in range(n)], "times": [notes[i]["t"] for i in idx]}


def _silent(path: Path) -> bool:
    y, _ = sf.read(str(path), dtype="float32", always_2d=True)
    if y.size == 0:
        return True
    hop = 2048
    frames = y[: len(y) // hop * hop].mean(axis=1).reshape(-1, hop)
    return bool(frames.size == 0 or np.sqrt((frames ** 2).mean(axis=1)).max() < SILENT_RMS)


def _rms(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = y.mean(axis=1)
    hop = sr // RMS_HZ
    frames = mono[: len(mono) // hop * hop].reshape(-1, hop)
    return np.sqrt((frames ** 2).mean(axis=1)) if frames.size else np.zeros(0)


def run(ctx: Context) -> dict:
    stems = ctx.data("stems")
    if not stems:
        raise StageUnavailable("melody needs stems (run without --fast)")
    notes, summary = {}, {}
    for stem in STEMS:
        src = ctx.path(stems["stems"][stem])
        if _silent(src):
            notes[stem] = []
        else:
            dest = ctx.path(f"notes_{stem}.json")
            try:
                r = subprocess.run([*BASIC_PITCH_CMD, str(RUNNER), str(src), str(dest)],
                                   capture_output=True, text=True, timeout=TIMEOUT_S)
            except FileNotFoundError:
                raise StageUnavailable("uvx not found; install uv")
            except subprocess.TimeoutExpired:
                raise StageUnavailable(f"basic-pitch timed out on {stem}")
            if r.returncode != 0:
                raise StageUnavailable(f"basic-pitch failed on {stem}: {r.stderr.strip()[-300:]}")
            notes[stem] = clean_notes(json.loads(dest.read_text()), stem, rms=_rms(src))
        ns = notes[stem]
        summary[stem] = {
            "count": len(ns),
            "range": [min(ns, key=lambda x: x["pitch"])["name"], max(ns, key=lambda x: x["pitch"])["name"]] if ns else [],
            "hook": find_hook(skyline(ns)),
        }
    return {"notes": notes, "summary": summary}
