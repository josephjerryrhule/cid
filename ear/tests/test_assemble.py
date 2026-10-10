import json

import numpy as np

from core.audio import CURVE_HZ
from stages import assemble


def _seed(ctx, n_sections=3, n_moments=6, words=40, hear=True):
    duration = 60.0
    ctx.write("ingest", {"_v": 1, "file": "track.wav", "duration": duration, "has_video": False, "warnings": [], "silent": False, "short": False})
    ctx.write("rhythm", {"_v": 1, "bpm": 124.0, "confidence": 0.94, "meter": "4/4", "changes": [],
                         "beats": [{"t": 0.484, "bar": 1, "beat": 1, "strength": 0.9}], "downbeats": [0.484]})
    ctx.write("hits", {"_v": 1, "from": "mix", "hits": {"kick": [{"t": 0.484, "v": 0.8}], "snare": [], "hat": [], "bass": [], "vocal": []}})
    ctx.write("classify", {"_v": 1, "kind": "music", "features": {}})
    edges = np.linspace(0, duration, n_sections + 1)
    sections = [{"id": f"s{i + 1}", "label": "body", "start": float(a), "end": float(b), "energy": 0.4}
                for i, (a, b) in enumerate(zip(edges, edges[1:]))]
    n = int(duration * CURVE_HZ)
    np.savez_compressed(ctx.path("curves_structure.npz"), energy=np.linspace(0, 1, n), rms=np.ones(n),
                        low=np.zeros(n), mid=np.zeros(n), high=np.zeros(n), stem_vocals=np.zeros(n))
    ctx.write("structure", {"_v": 1, "curves_file": "curves_structure.npz", "curve_hz": CURVE_HZ, "sections": sections,
                            "key": {"tonic": "A", "mode": "minor", "confidence": 0.81}})
    moments = [{"t": 30.97, "type": "drop", "strength": 0.97, "note": "full kit"}]
    moments += [{"t": 0.5 * i, "type": "impact", "strength": 0.5} for i in range(1, n_moments)]
    ctx.write("moments", {"_v": 1, "moments": moments})
    ctx.write("mix", {"_v": 1, "lufs": -9.1, "true_peak": -0.3, "lra": 5.2, "clipped_samples": 0, "width": 0.4,
                      "flags": ["low-mid build-up 200 to 400 Hz (+3.4 dB)"], "bands": {},
                      "short_term": {"hz": 10, "offset": 1.5, "values": [-12.0] * 570}})
    text = " ".join(["word"] * words)
    ctx.write("speech", {"_v": 1, "language": "en", "segments": [{"t0": 1.0, "t1": 3.0, "text": text, "words": []}], "pauses": []})
    if hear:
        ctx.write("hear", {"_v": 1, "kind": "music",
                           "overall": {"genre": "house", "mood": "dark", "instruments": ["pluck"], "vocals": "none", "production_style": "clean"},
                           "sections": [{"id": s["id"], "label": "drop" if s["id"] == "s2" else "intro", "description": "pad swells", "what_changes": ""} for s in sections],
                           "extra_moments": [{"approx_t": 50.0, "what": "second drop"}], "mix_impressions": ["wide pad"],
                           "speech": {"tone": "", "emotion": "", "room": "", "noise": ""}, "notice": None, "warnings": []})
    ctx.write("see", {"_v": 1, "images": ["spec_full.png", "spec_s2.png"]})
    ctx.write("stems", {"_v": 1, "_unavailable": "demucs failed: out of memory"})


def test_cues_have_frames_and_heard_labels(ctx):
    _seed(ctx)
    d = {s: ctx.data(s) for s in assemble.DEPS}
    cues = assemble.build_cues(d, 60)
    assert cues["beats"][0] == {"t": 0.484, "f": 29, "bar": 1, "beat": 1, "strength": 0.9}
    assert cues["hits"]["kick"][0]["f"] == 29
    assert cues["sections"][1]["label"] == "drop" and cues["sections"][1]["label_source"] == "gemini"
    assert cues["sections"][1]["f0"] == 1200
    assert any(m["type"] == "drop" and m["f"] == 1858 for m in cues["moments"])
    assert cues["key"] == {"tonic": "A", "mode": "minor", "confidence": 0.81}
    assert cues["source"]["fps"] == 60 and cues["source"]["kind"] == "music"


def test_partial_hearing_sections_fall_back_to_auto_labels(ctx):
    _seed(ctx)
    h = ctx.read("hear")
    h["sections"] = [{"id": "s9", "label": "ghost", "description": "", "what_changes": ""}]
    ctx.write("hear", h)
    d = {s: ctx.data(s) for s in assemble.DEPS}
    cues = assemble.build_cues(d, 60)
    assert all(s["label_source"] == "auto" for s in cues["sections"])
    conflicts = assemble.find_conflicts(cues, d["hear"], "music", ["spec_full.png"])
    assert any("no label" in c for c in conflicts)


def test_conflicts_for_unmeasured_heard_moment(ctx):
    _seed(ctx)
    d = {s: ctx.data(s) for s in assemble.DEPS}
    cues = assemble.build_cues(d, 60)
    conflicts = assemble.find_conflicts(cues, d["hear"], "speech", ["spec_full.png", "spec_s3.png"])
    assert any("second drop" in c and "spec_s3.png" in c for c in conflicts)
    assert any("measured kind is speech" in c for c in conflicts)


def test_curves_resampled_to_fps(ctx):
    _seed(ctx)
    curves = assemble.build_curves(ctx, 30, 60.0)
    assert curves["fps"] == 30 and curves["frames"] == 1800
    assert set(curves["curves"]) >= {"energy", "loudness", "bands.low", "bands.mid", "bands.high", "stems.vocals"}
    assert all(len(v) == 1800 for v in curves["curves"].values())
    assert all(0 <= x <= 1 for v in curves["curves"].values() for x in v)


def test_run_writes_files_and_report_is_lean(ctx):
    _seed(ctx, n_sections=10, n_moments=60, words=1500)
    out = assemble.run(ctx)
    assert set(out["files"]) == {"report.md", "cues.json", "curves.json"}
    report = ctx.path("report.md").read_text()
    assert len(report) / 4 < 2000
    assert chr(0x2014) not in report and chr(0x2013) not in report
    assert "0:00 to 0:06" in report
    assert "unavailable: demucs failed" in report
    assert "second drop" in report
    assert "-9.1 LUFS" in report
    json.loads(ctx.path("cues.json").read_text())


def test_report_without_hearing_says_so(ctx):
    _seed(ctx, hear=False)
    assemble.run(ctx)
    assert "not run" in ctx.path("report.md").read_text()
