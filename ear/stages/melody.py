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

VERSION = 1
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
SILENT_RMS = 1e-3
TIMEOUT_S = 15 * 60


def _name(pitch: int) -> str:
    import librosa

    return librosa.midi_to_note(int(pitch), unicode=False)


def clean_notes(raw: list[dict], stem: str) -> list[dict]:
    out = []
    for n in sorted(raw, key=lambda n: n["t"]):
        if n["end"] - n["t"] < MIN_NOTE_S:
            continue
        if stem == "bass" and n["pitch"] > BASS_MAX_PITCH:
            continue
        out.append({"t": round(n["t"], 3), "end": round(n["end"], 3), "pitch": int(n["pitch"]),
                    "name": _name(n["pitch"]), "velocity": round(float(min(1.0, n["velocity"])), 3)})
    return out


def find_hook(notes: list[dict], n: int = 4, min_times: int = 3) -> dict | None:
    """Most repeated n-note phrase, transposition invariant (by intervals), non-overlapping."""
    if len(notes) < n * min_times:
        return None
    pitches = [x["pitch"] for x in notes]
    starts = defaultdict(list)
    for i in range(len(notes) - n + 1):
        key = tuple(pitches[i + k + 1] - pitches[i + k] for k in range(n - 1))
        if starts[key] and i < starts[key][-1] + n:
            continue
        starts[key].append(i)
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
            notes[stem] = clean_notes(json.loads(dest.read_text()), stem)
        ns = notes[stem]
        summary[stem] = {
            "count": len(ns),
            "range": [min(ns, key=lambda x: x["pitch"])["name"], max(ns, key=lambda x: x["pitch"])["name"]] if ns else [],
            "hook": find_hook(ns),
        }
    return {"notes": notes, "summary": summary}
