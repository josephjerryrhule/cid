# /listen Phase 2 Implementation Plan: melody and tags

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `melody` stage (notes per stem) and a `tags` stage (what is playing when, instruments and SFX) to /listen, and wire both into cues, curves, report, images, the hearing prompt and the Remotion helper.

**Architecture:** Two new pipeline stages in the existing cached registry. `melody` runs Spotify basic-pitch in an isolated uvx environment over the vocals, bass and other stems. `tags` runs PANNs Cnn14_DecisionLevelMax framewise sound event detection over the mix in the main environment. Assembly, images, hearing and the Remotion helper read their JSON.

**Tech Stack:** basic-pitch 0.3.0 (ONNX backend, isolated via uvx), panns_inference 0.1.1 + torchlibrosa, existing librosa/numpy/scipy/matplotlib, Node 24 test runner for TypeScript.

**Spec:** `ear/docs/specs/2026-10-10-listen-design.md` section 13 (phase 2), on top of sections 1 to 12.

## Global Constraints

- Everything from the phase 1 plan's Global Constraints still applies (`docs/plans/2026-10-10-listen.md`): Python 3.12 via uv, argument-list subprocesses, no em or en dashes anywhere, frame numbers `f = round(t * fps)`, values 0..1, report under 2,000 tokens, images capped at 16, commits without Claude attribution.
- basic-pitch runs only via: `uvx --python 3.12 --with "basic-pitch[onnx]==0.3.0" --with "setuptools<70" --with "scipy<1.13" --with "numpy<2" python tools/basic_pitch_runner.py IN OUT`. Verified 2026-10-11: these pins are required (resampy needs pkg_resources; basic-pitch uses scipy.signal.gaussian). The ONNX model is `str(ICASSP_2022_MODEL_PATH) + ".onnx"`.
- PANNs model files live in `~/panns_data/` (`Cnn14_DecisionLevelMax.pth`, `class_labels_indices.csv`, `ontology.json`) and are downloaded with `urllib`, never wget. `panns_inference` reads the CSV at import time, so assets must exist before the first import.
- Both new stages are skipped with `--fast`. Nothing new is uploaded anywhere.

## Review Focus

1. **Silent or near-silent stems** (an instrumental has an empty vocals stem): melody must return no notes for it, not hallucinated ones. Pinned by `test_silent_stem_gives_no_notes` in Task 2.
2. **Long tracks** (10 minutes): tags must process in chunks without loading the whole framewise matrix of a long file at once. Pinned by `test_chunked_inference_matches_length` in Task 3.
3. **Missing model files or no network on first run**: tags must become unavailable with a clear reason, not crash the run. Pinned by `test_missing_assets_is_unavailable` in Task 3.
4. **Gemini names an instrument with different wording than AudioSet** ("synth plucks" vs "Synthesizer"): must not raise a false conflict. Pinned by `test_instrument_matching_is_fuzzy` in Task 4.
5. **Tracks with no melody or no tags at all** (pure noise, speech only): report and cues must still assemble cleanly. Pinned by `test_assemble_without_notes_or_tags` in Task 4.

---

### Task 1: Model assets

**Files:**
- Create: `core/models.py`
- Modify: `pyproject.toml` (add `panns-inference==0.1.1`, `torchlibrosa>=0.1`), `uv.lock`, `setup.py` (`--prefetch` also fetches PANNs assets)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `core.models.PANNS_DIR: Path` (`~/panns_data`, override with env `LISTEN_PANNS_DIR`), `core.models.ensure_panns() -> dict[str, Path]` with keys `checkpoint`, `labels`, `ontology`; raises `core.pipeline.StageUnavailable` with the failing URL on download error.

- [ ] **Step 1: Add dependencies**

Run: `uv add "panns-inference==0.1.1" "torchlibrosa>=0.1"`
Expected: `pyproject.toml` and `uv.lock` updated, install succeeds.

- [ ] **Step 2: Write the failing test**

`tests/test_models.py`:

```python
import pytest

from core import models
from core.pipeline import StageUnavailable


def test_existing_assets_are_not_downloaded(tmp_path, monkeypatch):
    monkeypatch.setenv("LISTEN_PANNS_DIR", str(tmp_path))
    for name in models.ASSETS:
        (tmp_path / name).write_bytes(b"x")
    calls = []
    monkeypatch.setattr(models.urllib.request, "urlretrieve", lambda *a: calls.append(a))
    paths = models.ensure_panns()
    assert calls == []
    assert paths["checkpoint"] == tmp_path / "Cnn14_DecisionLevelMax.pth"


def test_download_failure_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("LISTEN_PANNS_DIR", str(tmp_path))

    def boom(url, dest):
        raise OSError("offline")

    monkeypatch.setattr(models.urllib.request, "urlretrieve", boom)
    with pytest.raises(StageUnavailable, match="could not download"):
        models.ensure_panns()
    assert not list(tmp_path.glob("*.tmp"))
```

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'models' from 'core'`.

- [ ] **Step 3: Implement**

`core/models.py`:

```python
"""Downloads model assets once. PANNs expects them in ~/panns_data; urllib, never wget."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from core.pipeline import StageUnavailable

ASSETS = {
    "Cnn14_DecisionLevelMax.pth": "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1",
    "class_labels_indices.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
    "ontology.json": "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json",
}


def panns_dir() -> Path:
    return Path(os.environ.get("LISTEN_PANNS_DIR", Path.home() / "panns_data"))


def ensure_panns() -> dict[str, Path]:
    root = panns_dir()
    root.mkdir(parents=True, exist_ok=True)
    for name, url in ASSETS.items():
        dest = root / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(url, tmp)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise StageUnavailable(f"could not download {name} from {url}: {e}")
        os.replace(tmp, dest)
    return {
        "checkpoint": root / "Cnn14_DecisionLevelMax.pth",
        "labels": root / "class_labels_indices.csv",
        "ontology": root / "ontology.json",
    }
```

In `setup.py`, at the end of `prefetch()`, add:

```python
    subprocess.run([*py, "from core.models import ensure_panns; print(ensure_panns())"], check=False, cwd=str(HERE))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_models.py tests/test_setup.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add core/models.py tests/test_models.py pyproject.toml uv.lock setup.py
git commit -m "feat(ear): PANNs model assets downloaded with urllib"
```

---

### Task 2: Melody stage

**Files:**
- Create: `tools/basic_pitch_runner.py`, `stages/melody.py`
- Modify: `tests/synth.py` (add `melody`)
- Test: `tests/test_melody.py`

**Interfaces:**
- Consumes: `ctx.data("stems")["stems"]` mapping stem name to wav file name.
- Produces: `stages.melody.VERSION = 1`, `DEPS = ("ingest", "stems")`, `STEMS = ("vocals", "bass", "other")`, `run(ctx) -> {"notes": {stem: [note]}, "summary": {stem: {"count", "range", "hook"}}}` where note is `{"t", "end", "pitch", "name", "velocity"}`; helpers `clean_notes(raw, stem) -> list[dict]`, `find_hook(notes, n=4, min_times=3) -> dict | None` (`{"pattern": [names], "times": [t]}`).
- `tests.synth.melody(pitches, note_s=0.4, gap_s=0.1, sr=SR) -> (y, truth)` where truth is a list of `(t, end, pitch)`.

- [ ] **Step 1: Add the synthetic melody generator**

Append to `tests/synth.py`:

```python
def melody(pitches, note_s=0.4, gap_s=0.1, sr=SR, start=0.5):
    """Harmonic tones with short attack and release. Returns (signal, [(t, end, midi)])."""
    n_total = int((start + len(pitches) * (note_s + gap_s) + 0.5) * sr)
    y = np.zeros(n_total)
    truth = []
    t = start
    n = int(note_s * sr)
    tt = np.arange(n) / sr
    env = np.minimum(1, np.minimum(tt / 0.01, (note_s - tt) / 0.03))
    for p in pitches:
        f = midi_hz(p)
        tone = sum(np.sin(2 * np.pi * f * h * tt) / h for h in (1, 2, 3, 4))
        _add(y, tone * env, t, sr, 1.0)
        truth.append((t, t + note_s, p))
        t += note_s + gap_s
    return (0.6 * y / np.max(np.abs(y))).astype(np.float32), truth
```

- [ ] **Step 2: Write the failing tests**

`tests/test_melody.py`:

```python
import numpy as np
import pytest
import soundfile as sf

from core.pipeline import StageUnavailable
from stages import melody
from tests.synth import SR, melody as synth_melody


def _stems(ctx, other=None, vocals=None, bass=None, seconds=6.0):
    silent = np.zeros(int(seconds * SR), np.float32)
    for name, y in (("vocals", vocals), ("bass", bass), ("other", other), ("drums", None)):
        y = silent if y is None else y
        sf.write(str(ctx.path(f"stem_{name}.wav")), np.stack([y, y], 1), SR, subtype="FLOAT")
    ctx.write("stems", {"_v": 1, "stems": {s: f"stem_{s}.wav" for s in ("vocals", "bass", "other", "drums")}})


def test_clean_notes_sorts_filters_and_names():
    raw = [{"t": 1.0, "end": 1.5, "pitch": 72, "velocity": 0.6},
           {"t": 0.0, "end": 0.03, "pitch": 70, "velocity": 0.9},
           {"t": 0.1, "end": 0.6, "pitch": 69, "velocity": 0.7}]
    out = melody.clean_notes(raw, "other")
    assert [n["name"] for n in out] == ["A4", "C5"]
    assert out[0]["t"] == 0.1


def test_bass_drops_high_notes():
    raw = [{"t": 0.0, "end": 0.5, "pitch": 40, "velocity": 0.5}, {"t": 0.5, "end": 1.0, "pitch": 76, "velocity": 0.5}]
    assert [n["pitch"] for n in melody.clean_notes(raw, "bass")] == [40]


def test_find_hook_repeated_pattern():
    phrase = [60, 62, 64, 60]
    notes, t = [], 0.0
    for rep in range(3):
        for p in phrase + [rep + 50]:
            notes.append({"t": t, "end": t + 0.2, "pitch": p, "name": "", "velocity": 0.5})
            t += 0.25
    hook = melody.find_hook(notes)
    assert hook is not None
    assert len(hook["times"]) == 3
    assert hook["times"][0] == 0.0


def test_find_hook_none_when_nothing_repeats():
    notes = [{"t": i * 0.3, "end": i * 0.3 + 0.2, "pitch": 40 + i, "name": "", "velocity": 0.5} for i in range(10)]
    assert melody.find_hook(notes) is None


def test_runner_failure_is_unavailable(ctx, monkeypatch):
    y, _ = synth_melody([69, 72])
    _stems(ctx, other=y)

    class R:
        returncode, stdout, stderr = 1, "", "boom"

    monkeypatch.setattr(melody.subprocess, "run", lambda *a, **k: R())
    with pytest.raises(StageUnavailable, match="basic-pitch failed"):
        melody.run(ctx)


def test_needs_stems(ctx):
    with pytest.raises(StageUnavailable, match="needs stems"):
        melody.run(ctx)


@pytest.mark.models
def test_silent_stem_gives_no_notes(ctx):
    y, _ = synth_melody([69, 72, 76])
    _stems(ctx, other=y)
    out = melody.run(ctx)
    assert out["notes"]["vocals"] == [] and out["notes"]["bass"] == []
    assert out["summary"]["vocals"]["count"] == 0


@pytest.mark.models
def test_synthetic_melody_recovered(ctx):
    pitches = [69, 72, 76, 74, 72, 69, 67, 69]
    y, truth = synth_melody(pitches)
    _stems(ctx, other=y, seconds=len(y) / SR)
    notes = melody.run(ctx)["notes"]["other"]
    for t, end, p in truth:
        match = [n for n in notes if abs(n["t"] - t) <= 0.05]
        assert match, f"no note near {t}"
        assert any(n["pitch"] == p for n in match), f"pitch {p} at {t}: got {[n['pitch'] for n in match]}"
```

Run: `uv run pytest tests/test_melody.py -v`
Expected: FAIL with `ImportError: cannot import name 'melody' from 'stages'`.

- [ ] **Step 3: Implement the runner and stage**

`tools/basic_pitch_runner.py`:

```python
"""Runs inside an isolated uvx environment. Usage: basic_pitch_runner.py IN.wav OUT.json"""
import json
import sys

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

_, _, events = predict(sys.argv[1], str(ICASSP_2022_MODEL_PATH) + ".onnx")
notes = [{"t": float(s), "end": float(e), "pitch": int(p), "velocity": float(a)} for s, e, p, a, *_ in events]
with open(sys.argv[2], "w") as f:
    json.dump(sorted(notes, key=lambda n: n["t"]), f)
```

`stages/melody.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_melody.py -v`
Expected: 8 passed (first run downloads the isolated basic-pitch environment).

If `test_synthetic_melody_recovered` misses pitches by exactly 12 (octave errors), print the returned notes and report them; do not loosen the assertion.

- [ ] **Step 5: Commit**

```bash
git add tools/basic_pitch_runner.py stages/melody.py tests/test_melody.py tests/synth.py
git commit -m "feat(ear): melody stage, notes per stem with basic-pitch"
```

---

### Task 3: Tags stage

**Files:**
- Create: `stages/tags.py`
- Test: `tests/test_tags.py`

**Interfaces:**
- Consumes: `master.wav`; `core.models.ensure_panns()`.
- Produces: `stages.tags.VERSION = 1`, `DEPS = ("ingest",)`, `CURVES_FILE = "curves_tags.npz"`, `run(ctx) -> {"spans": [{"label", "group", "t", "end", "confidence"}], "curve_hz": 100, "curves_file": str, "labels": [label ranked by active seconds]}`. Helpers: `class_groups(ontology: list[dict], labels: list[str]) -> dict[str, str]`, `to_spans(prob, hz, thresh=0.3, min_s=0.3, smooth_s=0.3, merge_s=0.5) -> list[tuple[float, float, float]]`, `framewise(y, sed, chunk_s=60.0) -> np.ndarray` (frames x 527 at 100 Hz).

- [ ] **Step 1: Write the failing tests**

`tests/test_tags.py`:

```python
import platform
import subprocess

import numpy as np
import pytest

from core.pipeline import StageUnavailable
from stages import tags
from tests.helpers import ingest_array
from tests.synth import drum_loop

ONTOLOGY = [
    {"id": "/m/music", "name": "Music", "child_ids": ["/m/inst", "/m/genre"]},
    {"id": "/m/inst", "name": "Musical instrument", "child_ids": ["/m/guitar"]},
    {"id": "/m/guitar", "name": "Guitar", "child_ids": []},
    {"id": "/m/genre", "name": "Music genre", "child_ids": ["/m/house"]},
    {"id": "/m/house", "name": "House music", "child_ids": []},
    {"id": "/m/human", "name": "Human sounds", "child_ids": ["/m/speech"]},
    {"id": "/m/speech", "name": "Speech", "child_ids": []},
    {"id": "/m/things", "name": "Sounds of things", "child_ids": ["/m/glass"]},
    {"id": "/m/glass", "name": "Glass", "child_ids": []},
    {"id": "/m/env", "name": "Channel, environment and background", "child_ids": ["/m/hum"]},
    {"id": "/m/hum", "name": "Hum", "child_ids": []},
]


def test_class_groups_from_ontology():
    g = tags.class_groups(ONTOLOGY, ["Guitar", "House music", "Speech", "Glass", "Hum", "Music", "Unknown"])
    assert g == {"Guitar": "instrument", "House music": "genre", "Speech": "voice", "Glass": "sfx",
                 "Hum": "ambience", "Music": "music", "Unknown": "sfx"}


def test_to_spans_threshold_min_length_and_merge():
    p = np.zeros(1000)
    p[100:300] = 0.9   # 2 s span
    p[320:400] = 0.9   # gap 0.2 s: merges
    p[600:610] = 0.9   # 0.1 s: too short
    spans = tags.to_spans(p, 100)
    assert len(spans) == 1
    a, b, conf = spans[0]
    assert abs(a - 1.0) <= 0.05 and abs(b - 4.0) <= 0.05 and conf > 0.8


def test_chunked_inference_matches_length():
    class FakeSED:
        def inference(self, audio):
            n = audio.shape[1] // 320 + 1
            return np.full((1, n, 527), 0.5, dtype=np.float32)

    y = np.zeros(int(150.3 * tags.SR), np.float32)
    fw = tags.framewise(y, FakeSED(), chunk_s=60.0)
    assert fw.shape[1] == 527
    assert abs(fw.shape[0] - 150.3 * tags.FRAME_HZ) <= 3


def test_missing_assets_is_unavailable(ctx, monkeypatch):
    ingest_array(ctx, np.zeros(44100, np.float32))

    def unavailable():
        raise StageUnavailable("could not download Cnn14_DecisionLevelMax.pth from https://example: offline")

    monkeypatch.setattr(tags, "ensure_panns", unavailable)
    with pytest.raises(StageUnavailable, match="could not download"):
        tags.run(ctx)


@pytest.mark.models
@pytest.mark.skipif(platform.system() != "Darwin", reason="uses macOS say")
def test_speech_is_tagged(ctx, tmp_path):
    import librosa

    aiff = tmp_path / "s.aiff"
    subprocess.run(["say", "-o", str(aiff), "This is a short spoken sentence for the tagger to hear clearly."], check=True)
    y, _ = librosa.load(str(aiff), sr=44100)
    ingest_array(ctx, y)
    out = tags.run(ctx)
    speech = [s for s in out["spans"] if s["label"] == "Speech"]
    assert speech and speech[0]["group"] == "voice"
    assert sum(s["end"] - s["t"] for s in speech) >= 0.5 * len(y) / 44100


@pytest.mark.models
def test_drum_loop_is_tagged(ctx):
    y, _ = drum_loop(120, bars=8)
    ingest_array(ctx, y)
    labels = {s["label"] for s in tags.run(ctx)["spans"]}
    drumish = {"Drum", "Drum kit", "Snare drum", "Bass drum", "Percussion", "Drum machine", "Hi-hat", "Drum roll"}
    assert labels & drumish, labels
```

Run: `uv run pytest tests/test_tags.py -v`
Expected: FAIL with `ImportError: cannot import name 'tags' from 'stages'`.

- [ ] **Step 2: Implement**

`stages/tags.py`:

```python
"""Stage: what is playing when. PANNs framewise AudioSet tagging (527 classes, 100 Hz)."""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import median_filter

from core.context import Context
from core.models import ensure_panns

VERSION = 1
DEPS = ("ingest",)
SR = 32000
FRAME_HZ = 100
CURVES_FILE = "curves_tags.npz"
THRESH = 0.3
MIN_SPAN_S = 0.3
MAX_LABELS = 40
CURVE_MIN_S = 2.0
GROUP_BY_ROOT = {
    "Musical instrument": "instrument",
    "Music genre": "genre",
    "Music mood": "genre",
    "Music": "music",
    "Human sounds": "voice",
    "Animal": "sfx",
    "Sounds of things": "sfx",
    "Natural sounds": "sfx",
    "Source-ambiguous sounds": "sfx",
    "Channel, environment and background": "ambience",
}


def class_groups(ontology: list[dict], labels: list[str]) -> dict[str, str]:
    by_id = {o["id"]: o for o in ontology}
    parent: dict[str, str] = {}
    for o in ontology:
        for c in o.get("child_ids", []):
            parent.setdefault(c, o["id"])
    name_to_id = {o["name"]: o["id"] for o in ontology}
    out = {}
    for label in labels:
        node, group = name_to_id.get(label), None
        while node is not None:
            name = by_id[node]["name"]
            if name in GROUP_BY_ROOT and (group is None or GROUP_BY_ROOT[name] != "music"):
                group = group or GROUP_BY_ROOT[name]
                if GROUP_BY_ROOT[name] != "music":
                    break
            node = parent.get(node)
        out[label] = group or "sfx"
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def to_spans(prob, hz, thresh=THRESH, min_s=MIN_SPAN_S, smooth_s=0.3, merge_s=0.5) -> list[tuple[float, float, float]]:
    p = median_filter(np.asarray(prob, dtype=float), size=max(1, int(smooth_s * hz)))
    merged: list[list[int]] = []
    for a, b in _runs(p > thresh):
        if merged and (a - merged[-1][1]) / hz < merge_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a / hz, b / hz, float(p[a:b].max())) for a, b in merged if (b - a) / hz >= min_s]


def framewise(y: np.ndarray, sed, chunk_s: float = 60.0) -> np.ndarray:
    step = int(chunk_s * SR)
    parts = []
    for start in range(0, max(len(y), 1), step):
        chunk = y[start : start + step]
        if len(chunk) < SR // 10:
            break
        out = sed.inference(chunk[None, :].astype(np.float32))[0]
        parts.append(out[: int(round(len(chunk) / SR * FRAME_HZ))])
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 527))


def run(ctx: Context) -> dict:
    import librosa

    paths = ensure_panns()
    from panns_inference import SoundEventDetection, labels

    sed = SoundEventDetection(checkpoint_path=str(paths["checkpoint"]), device="cpu")
    y, _ = librosa.load(str(ctx.path("master.wav")), sr=SR, mono=True)
    fw = framewise(y, sed)
    groups = class_groups(json.loads(paths["ontology"].read_text()), list(labels))
    found = []
    for i, label in enumerate(labels):
        spans = to_spans(fw[:, i], FRAME_HZ)
        if spans:
            found.append((sum(b - a for a, b, _ in spans), label, spans, i))
    found.sort(key=lambda x: -x[0])
    found = found[:MAX_LABELS]
    curves = {label: np.clip(fw[:, i], 0, 1).astype(np.float32) for total, label, _, i in found if total >= CURVE_MIN_S}
    np.savez_compressed(ctx.path(CURVES_FILE), **curves)
    spans_out = [
        {"label": label, "group": groups[label], "t": round(a, 2), "end": round(b, 2), "confidence": round(c, 3)}
        for _, label, spans, _ in found
        for a, b, c in spans
    ]
    return {
        "spans": sorted(spans_out, key=lambda s: (s["t"], s["label"])),
        "curve_hz": FRAME_HZ,
        "curves_file": CURVES_FILE,
        "labels": [label for _, label, _, _ in found],
    }
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_tags.py -v`
Expected: 6 passed.

If `test_class_groups_from_ontology` fails because a label sits under both Music and a more specific root, the specific root must win (instrument, genre) and "music" only applies when nothing more specific is found. Fix `class_groups` accordingly; keep the test.

- [ ] **Step 4: Commit**

```bash
git add stages/tags.py tests/test_tags.py
git commit -m "feat(ear): tags stage, instruments and sfx over time with PANNs"
```

---

### Task 4: Wiring (registry, hear, assemble, see, Remotion)

**Files:**
- Modify: `core/registry.py`, `stages/hear.py`, `stages/assemble.py`, `stages/see.py`, `remotion/cues-core.ts`, `SKILL.md`
- Test: `tests/test_wiring_phase2.py`, `remotion/cues-core.test.ts`

**Interfaces:**
- Consumes: `melody` and `tags` stage JSON from Tasks 2 and 3.
- Produces:
  - Registry: `melody` after `stems` (enabled: not fast, analysable, stems available), `tags` after `ingest` (enabled: not fast, playable). Order: ingest, stems, melody, rhythm, hits, classify, tags, structure, moments, mix, speech, hear, see, assemble.
  - `stages.hear`: `DEPS` adds `"tags"`; `VERSION = 2`; `PROMPT_VERSION = 2`; skeleton gains an `instruments measured:` block (top 8 instrument or voice labels with their first span); `instrument_conflicts(heard: list[str], tag_labels: list[str]) -> list[str]` (fuzzy: a heard instrument matches when any word of 4+ letters shares its first 5 letters with any word of any tag label).
  - `stages.assemble`: `DEPS` adds `"melody", "tags"`; cues gain `notes` and `tags` (each item with `f` and `f1`); curves gain `tags.<label>`; report gains "What's playing when (measured)" (top 8 instrument/voice labels and top 8 sfx spans) and "Melody (measured)" (per stem count, range, hook); conflicts include `instrument_conflicts`.
  - `stages.see`: adds `notes.png` piano roll (one lane per stem with notes) when melody has notes.
  - `remotion/cues-core.ts`: types `Note = { t; f; end; f1; pitch; name; velocity }`, `Tag = { label; group; t; f; end; f1; confidence }`; `noteAt(frame, notes) -> Note | null`; `tagActive(frame, tags, label) -> Tag | null`; `Cues` gains optional `notes?: Record<string, Note[]>` and `tags?: Tag[]`.

- [ ] **Step 1: Write the failing Python tests**

`tests/test_wiring_phase2.py`:

```python
import json

import numpy as np

from core.audio import CURVE_HZ
from core.registry import build_stages
from stages import assemble, hear
from tests.test_assemble import _seed


def _seed_phase2(ctx, notes=True, tag=True):
    _seed(ctx)
    if notes:
        n = [{"t": 1.0 + i, "end": 1.4 + i, "pitch": 69, "name": "A4", "velocity": 0.6} for i in range(5)]
        ctx.write("melody", {"_v": 1, "notes": {"vocals": [], "bass": [], "other": n},
                             "summary": {"vocals": {"count": 0, "range": [], "hook": None},
                                         "bass": {"count": 0, "range": [], "hook": None},
                                         "other": {"count": 5, "range": ["A4", "A4"], "hook": {"pattern": ["A4", "A4", "A4", "A4"], "times": [1.0]}}}})
    if tag:
        np.savez_compressed(ctx.path("curves_tags.npz"), Guitar=np.linspace(0, 1, 6000).astype(np.float32))
        ctx.write("tags", {"_v": 1, "curve_hz": 100, "curves_file": "curves_tags.npz", "labels": ["Guitar", "Whoosh"],
                           "spans": [{"label": "Guitar", "group": "instrument", "t": 0.0, "end": 60.0, "confidence": 0.8},
                                     {"label": "Whoosh", "group": "sfx", "t": 6.0, "end": 6.8, "confidence": 0.5}]})


def test_registry_order():
    names = [s.name for s in build_stages()]
    assert names.index("melody") == names.index("stems") + 1
    assert names.index("tags") < names.index("hear") < names.index("assemble")


def test_instrument_matching_is_fuzzy():
    tags_ = ["Synthesizer", "Electric guitar", "Drum kit"]
    assert hear.instrument_conflicts(["synth plucks", "guitar", "drums"], tags_) == []
    assert hear.instrument_conflicts(["marimba"], tags_) == ["marimba"]


def test_skeleton_lists_measured_instruments(ctx):
    _seed_phase2(ctx)
    data = {s: ctx.data(s) for s in hear.DEPS}
    prompt = hear.build_prompt(data, 0.0, 60.0)
    assert "instruments measured:" in prompt and "Guitar 0.0 to 60.0 s" in prompt


def test_cues_curves_report_include_notes_and_tags(ctx):
    _seed_phase2(ctx)
    assemble.run(ctx)
    cues = json.loads(ctx.path("cues.json").read_text())
    assert cues["notes"]["other"][0] == {"t": 1.0, "f": 60, "end": 1.4, "f1": 84, "pitch": 69, "name": "A4", "velocity": 0.6}
    assert cues["tags"][0]["label"] == "Guitar" and cues["tags"][0]["f1"] == 3600
    curves = json.loads(ctx.path("curves.json").read_text())
    assert "tags.Guitar" in curves["curves"] and len(curves["curves"]["tags.Guitar"]) == curves["frames"]
    report = ctx.path("report.md").read_text()
    assert "## What's playing when (measured)" in report and "Guitar 0:00 to 1:00" in report and "Whoosh" in report
    assert "## Melody (measured)" in report and "other: 5 notes, A4 to A4" in report
    assert len(report) / 4 < 2000


def test_assemble_without_notes_or_tags(ctx):
    _seed_phase2(ctx, notes=False, tag=False)
    assemble.run(ctx)
    cues = json.loads(ctx.path("cues.json").read_text())
    assert cues["notes"] == {} and cues["tags"] == []
    assert "What's playing when" not in ctx.path("report.md").read_text()


def test_piano_roll_image(ctx):
    from stages import see
    from tests.helpers import ingest_array
    from tests.synth import pink_noise

    ingest_array(ctx, pink_noise(8))
    _seed_phase2(ctx)
    assert "notes.png" in see.run(ctx)["images"]
```

Run: `uv run pytest tests/test_wiring_phase2.py -v`
Expected: FAIL (registry has no melody stage; `hear.instrument_conflicts` missing).

- [ ] **Step 2: Registry**

In `core/registry.py`, import `melody, tags` alongside the other stages, and change `build_stages()` to:

```python
def _has_stems(ctx: Context) -> bool:
    return ctx.data("stems") is not None


def build_stages() -> list[Stage]:
    return [
        _stage(ingest, "ingest"),
        _stage(stems, "stems", lambda c: not c.fast and _analysable(c)),
        _stage(melody, "melody", lambda c: not c.fast and _analysable(c) and _has_stems(c)),
        _stage(rhythm, "rhythm", _analysable),
        _stage(hits, "hits", _analysable),
        _stage(classify, "classify", _playable),
        _stage(tags, "tags", lambda c: not c.fast and _playable(c)),
        _stage(structure, "structure", _analysable),
        _stage(moments, "moments", _analysable),
        _stage(mix, "mix", _playable),
        _stage(speech, "speech", _speech_enabled),
        _stage(hear, "hear", lambda c: not c.fast and not c.no_cloud and _playable(c)),
        _stage(see, "see", _playable),
        _stage(assemble, "assemble"),
    ]
```

- [ ] **Step 3: Hearing skeleton and instrument conflicts**

In `stages/hear.py`: set `VERSION = 2`, `PROMPT_VERSION = 2`, `DEPS = ("ingest", "rhythm", "structure", "moments", "classify", "tags")`. Add:

```python
def _words(text: str) -> list[str]:
    return [w for w in "".join(ch.lower() if ch.isalpha() else " " for ch in text).split() if len(w) >= 4]


def instrument_conflicts(heard: list[str], tag_labels: list[str]) -> list[str]:
    tag_words = {w[:5] for label in tag_labels for w in _words(label)}
    return [h for h in heard if _words(h) and not any(w[:5] in tag_words for w in _words(h))]
```

In `build_prompt`, after the moments block and before the video line, add:

```python
    spans = (data.get("tags") or {}).get("spans", [])
    shown = []
    for s in spans:
        if s["group"] in ("instrument", "voice") and s["label"] not in [x["label"] for x in shown] and start <= s["t"] < end:
            shown.append(s)
        if len(shown) == 8:
            break
    if shown:
        lines.append("instruments measured:")
        lines += [f"  {s['label']} {s['t'] - start:.1f} to {s['end'] - start:.1f} s" for s in shown]
```

- [ ] **Step 4: Assemble**

In `stages/assemble.py`: `DEPS` gains `"melody", "tags"` (insert after `"stems"` and after `"classify"` respectively). Add to `build_cues` return dict:

```python
        "notes": {
            stem: [{"t": n["t"], "f": _f(n["t"], fps), "end": n["end"], "f1": _f(n["end"], fps),
                    "pitch": n["pitch"], "name": n["name"], "velocity": n["velocity"]} for n in ns]
            for stem, ns in ((d.get("melody") or {}).get("notes") or {}).items() if ns
        },
        "tags": [
            {"label": s["label"], "group": s["group"], "t": s["t"], "f": _f(s["t"], fps), "end": s["end"],
             "f1": _f(s["end"], fps), "confidence": s["confidence"]}
            for s in (d.get("tags") or {}).get("spans", [])
        ],
```

In `build_curves`, before computing `n`, add:

```python
    tg = ctx.data("tags")
    if tg and ctx.path(tg["curves_file"]).exists():
        tc = np.load(ctx.path(tg["curves_file"]))
        for label in tc.files:
            out[f"tags.{label}"] = resample_curve(tc[label], tg["curve_hz"], fps, duration)
```

In `find_conflicts`, add a parameter `tag_labels: list[str]` (default `()`) and before the return:

```python
    if hear.get("overall") and tag_labels:
        for inst in instrument_conflicts(hear["overall"].get("instruments", []), list(tag_labels)):
            out.append(f"heard {inst}, no matching measured tag")
```

with `from stages.hear import instrument_conflicts` at the top of the module, and in `run` pass `[s["label"] for s in (d.get("tags") or {}).get("spans", [])]` as `tag_labels`.

In `render_report`, insert after the Timeline block:

```python
    tg = cues.get("tags") or []
    if tg:
        lines.append("## What's playing when (measured)")
        firsts = {}
        for s in tg:
            if s["group"] in ("instrument", "voice"):
                firsts.setdefault(s["label"], []).append(s)
        ranked = sorted(firsts.items(), key=lambda kv: -sum(x["end"] - x["t"] for x in kv[1]))[:8]
        for label, ss in ranked:
            spans = ", ".join(f"{fmt_time(x['t'])} to {fmt_time(x['end'])}" for x in ss[:3])
            more = f" (+{len(ss) - 3})" if len(ss) > 3 else ""
            lines.append(f"- {label} {spans}{more}")
        sfx = sorted([s for s in tg if s["group"] == "sfx"], key=lambda s: -s["confidence"])[:8]
        if sfx:
            lines.append("SFX: " + "; ".join(f"{s['label']} {fmt_time(s['t'], precise=True)}" for s in sorted(sfx, key=lambda s: s["t"])))
        lines.append("")
    mel = (d.get("melody") or {}).get("summary") or {}
    if any(v.get("count") for v in mel.values()):
        lines.append("## Melody (measured)")
        for stem, v in mel.items():
            if not v.get("count"):
                continue
            hook = v.get("hook")
            hk = f"; hook {'-'.join(hook['pattern'])} x{len(hook['times'])} from {fmt_time(hook['times'][0], precise=True)}" if hook else ""
            lines.append(f"- {stem}: {v['count']} notes, {v['range'][0]} to {v['range'][1]}{hk}")
        lines.append("")
```

- [ ] **Step 5: Piano roll**

In `stages/see.py`: `DEPS` gains `"melody"`. Add:

```python
def _piano_roll(melody: dict, path) -> bool:
    lanes = [(s, ns) for s, ns in melody.get("notes", {}).items() if ns]
    if not lanes:
        return False
    fig, axes = plt.subplots(len(lanes), 1, figsize=(16, 1.8 * len(lanes)), dpi=100, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (stem, ns) in zip(axes, lanes):
        for n in ns:
            ax.add_patch(plt.Rectangle((n["t"], n["pitch"] - 0.4), n["end"] - n["t"], 0.8, color="#e8590c", lw=0))
        lo, hi = min(n["pitch"] for n in ns), max(n["pitch"] for n in ns)
        ax.set_ylim(lo - 2, hi + 2)
        ax.set_ylabel(stem, fontsize=9)
        ax.xaxis.set_major_formatter(_mmss)
    axes[-1].set_xlim(0, max(n["end"] for _, ns in lanes for n in ns) + 0.5)
    axes[0].set_title("notes (MIDI pitch)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True
```

and in `run`, before the ref image block:

```python
    mel = ctx.data("melody")
    if mel and _piano_roll(mel, ctx.path("notes.png")):
        images.append("notes.png")
```

- [ ] **Step 6: Remotion helpers (TDD)**

Append to `remotion/cues-core.test.ts`:

```ts
import { noteAt, tagActive, type Note, type Tag } from "./cues-core.ts";

test("noteAt returns the sounding note", () => {
  const notes: Note[] = [
    { t: 1, f: 60, end: 1.4, f1: 84, pitch: 69, name: "A4", velocity: 0.6 },
    { t: 2, f: 120, end: 2.4, f1: 144, pitch: 72, name: "C5", velocity: 0.7 },
  ];
  assert.equal(noteAt(70, notes)?.name, "A4");
  assert.equal(noteAt(100, notes), null);
  assert.equal(noteAt(120, notes)?.name, "C5");
});

test("tagActive finds a span for a label", () => {
  const tags: Tag[] = [{ label: "Guitar", group: "instrument", t: 0, f: 0, end: 10, f1: 600, confidence: 0.8 }];
  assert.equal(tagActive(300, tags, "Guitar")?.label, "Guitar");
  assert.equal(tagActive(700, tags, "Guitar"), null);
  assert.equal(tagActive(300, tags, "Speech"), null);
});
```

Run: `node --test remotion/cues-core.test.ts`. Expected: FAIL (no export `noteAt`).

Append to `remotion/cues-core.ts`:

```ts
export type Note = { t: number; f: number; end: number; f1: number; pitch: number; name: string; velocity: number };
export type Tag = { label: string; group: string; t: number; f: number; end: number; f1: number; confidence: number };

/** The note sounding at this frame (binary search on start frame, then range check). */
export function noteAt(frame: number, notes: Note[]): Note | null {
  const i = lastAtOrBefore(notes, frame);
  return i >= 0 && frame < notes[i].f1 ? notes[i] : null;
}

export function tagActive(frame: number, tags: Tag[], label: string): Tag | null {
  return tags.find((s) => s.label === label && s.f <= frame && frame < s.f1) ?? null;
}
```

and add `notes?: Record<string, Note[]>; tags?: Tag[];` to the `Cues` type.

- [ ] **Step 7: SKILL.md**

In `SKILL.md`, in "## Motion work", add after the curves sentence: "`cues.json` also has `notes.{vocals,bass,other}` (every melody note with frames) and `tags` (instruments, voices and SFX with start and end frames); `curves.json` has `tags.<label>`. Use `noteAt(frame, cues.notes.other)` to hit animation on each note and `tagActive(frame, cues.tags, "Electric guitar")` to drive layers by instrument." In "## Step 2: read, then look", add "Open `notes.png` for melody questions."

- [ ] **Step 8: Run all tests**

Run: `uv run pytest tests/test_wiring_phase2.py tests/test_assemble.py tests/test_hear.py tests/test_see.py tests/test_cli.py -v` then `node --test remotion/cues-core.test.ts`
Expected: all pass. Existing tests must still pass unchanged except where a stage's `DEPS` tuple appears in an assertion.

- [ ] **Step 9: Commit**

```bash
git add core/registry.py stages/hear.py stages/assemble.py stages/see.py remotion/cues-core.ts remotion/cues-core.test.ts SKILL.md tests/test_wiring_phase2.py
git commit -m "feat(ear): wire melody and tags into cues, report, images, hearing and Remotion"
```

---

### Task 5: Real-world acceptance

**Files:**
- Modify: `docs/specs/2026-10-10-listen-design.md` (section 13.5 results), `../README.md`

- [ ] **Step 1: Run on real tracks**

```bash
uv run python listen.py /Users/joeseph/Desktop/motiondev/sagobar-buildlog/public/kulakovka-good-vibes-281099.mp3
uv run python listen.py /Users/joeseph/Desktop/motiondev/techsgiving-launch/music/mp3/176432.mp3
uv run python listen.py tests/golden/your-heart-is-a-muscle/audio.wav
```

- [ ] **Step 2: Check against the acceptance criteria in spec 13.4**

For kulakovka: record which labels PANNs reports during the guitar (0:00 to 1:20; a spike saw Organ and Telephone bell, not Guitar) and whether Scratching appears near 0:06 (Gemini heard "turntable scratch"). Confirm the report lists "heard electric guitar, no matching measured tag" under Conflicts if Guitar is missing. Read `notes.png` and confirm the guitar riff shows as a repeated pattern in the "other" lane. For the pop song: vocals notes span a sung range (roughly A2 to C6) and Speech or Singing tags cover the verses. Record what matched and what did not.

- [ ] **Step 3: Full suite**

Run: `uv run pytest -q` (includes golden; about 20 minutes).
Expected: all pass.

- [ ] **Step 4: Record and commit**

Add "### 13.5 Results (date)" to the spec with the findings from Step 2, add one line each for melody and tags to the root README feature list, then:

```bash
git add docs/specs/2026-10-10-listen-design.md ../README.md
git commit -m "docs(ear): phase 2 acceptance results"
git push
```
