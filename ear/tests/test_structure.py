import numpy as np

from core.audio import ANALYSIS_SR
from stages import structure
from tests.helpers import ingest_array
from tests.synth import SR, chords, drum_loop


def _three_part_track():
    """Pad (0 to 20.5 s), drums and bass (20.5 to 40.5 s), pad (40.5 to 60.5 s)."""
    total = int(60.5 * SR)
    y = np.zeros(total)
    pad = chords([[57, 60, 64], [62, 65, 69]], seconds_each=2.0, repeats=6)
    y[: int(20.5 * SR)] = 0.3 * pad[: int(20.5 * SR)]
    y[int(40.5 * SR):] = 0.3 * pad[: total - int(40.5 * SR)]
    loop, _ = drum_loop(120, bars=10)
    a, b = int(20.0 * SR), int(40.5 * SR)
    seg = loop[: b - a].astype(float)
    seg[: int(0.5 * SR)] = 0
    y[a:b] = seg + 0.4 * np.sin(2 * np.pi * 55 * np.arange(b - a) / SR)
    beats = 0.5 + 0.5 * np.arange(120)
    return (0.8 * y / np.max(np.abs(y))).astype(np.float32), beats, beats[::4]


def test_sections_found_at_part_boundaries():
    import librosa

    y, beats, downbeats = _three_part_track()
    y24 = librosa.resample(y, orig_sr=SR, target_sr=ANALYSIS_SR)
    spans = structure.find_sections(y24, ANALYSIS_SR, beats, downbeats, 60.5)
    starts = [a for a, _ in spans]
    assert len(spans) == 3
    assert starts[0] == 0.0
    assert abs(starts[1] - 20.5) <= 2.1
    assert abs(starts[2] - 40.5) <= 2.1
    assert spans[-1][1] == 60.5


def test_section_count_comes_from_the_music():
    import librosa

    y, beats, downbeats = _three_part_track()
    two = y[: int(40.5 * SR)]
    y24 = librosa.resample(two, orig_sr=SR, target_sr=ANALYSIS_SR)
    spans = structure.find_sections(y24, ANALYSIS_SR, beats[beats < 40.5], downbeats[downbeats < 40.5], 40.5)
    assert len(spans) == 2
    assert abs(spans[1][0] - 20.5) <= 2.1


def test_boundaries_snap_to_downbeats():
    import librosa

    y, beats, downbeats = _three_part_track()
    y24 = librosa.resample(y, orig_sr=SR, target_sr=ANALYSIS_SR)
    spans = structure.find_sections(y24, ANALYSIS_SR, beats, downbeats, 60.5)
    for start, _ in spans[1:]:
        assert np.min(np.abs(downbeats - start)) < 1e-6


def test_auto_labels():
    assert structure.auto_labels([0.2, 0.5, 0.9, 0.3]) == ["intro", "build", "peak", "outro"]
    assert structure.auto_labels([0.9]) == ["peak"]


def test_key_detection():
    import librosa

    a_minor = chords([[57, 60, 64], [62, 65, 69], [52, 56, 59], [57, 60, 64]])
    c_major = chords([[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]])
    for y, tonic, mode in ((a_minor, "A", "minor"), (c_major, "C", "major")):
        key = structure.estimate_key(librosa.resample(y, orig_sr=SR, target_sr=ANALYSIS_SR), ANALYSIS_SR)
        assert (key["tonic"], key["mode"]) == (tonic, mode)
        assert 0 <= key["confidence"] <= 1


def test_run_writes_curves_and_sections(ctx):
    y, beats, downbeats = _three_part_track()
    ingest_array(ctx, y)
    ctx.write("rhythm", {"_v": 1, "beats": [{"t": float(b)} for b in beats], "downbeats": downbeats.tolist(), "confidence": 0.9})
    ctx.write("classify", {"_v": 1, "kind": "music"})
    out = structure.run(ctx)
    c = np.load(ctx.path(out["curves_file"]))
    assert out["curve_hz"] == 100
    assert abs(len(c["energy"]) / 100 - 60.5) < 0.1
    for name in ("energy", "low", "mid", "high"):
        assert c[name].min() >= 0 and c[name].max() <= 1
    ids = [s["id"] for s in out["sections"]]
    assert ids == ["s1", "s2", "s3"]
    middle = out["sections"][1]
    assert middle["energy"] > out["sections"][0]["energy"]
    assert out["key"] is not None
