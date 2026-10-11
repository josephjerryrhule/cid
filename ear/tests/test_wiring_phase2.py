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
