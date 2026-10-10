import numpy as np

from core.audio import ANALYSIS_SR
from stages import hits
from tests.helpers import ingest_array
from tests.synth import drum_loop


def _match(detected, truth, tol=0.02):
    detected = np.array(detected)
    truth = np.array(truth)
    if detected.size == 0:
        return 0.0, 0.0
    tp_det = sum(np.min(np.abs(truth - d)) <= tol for d in detected)
    tp_true = sum(np.min(np.abs(detected - t)) <= tol for t in truth)
    return tp_det / len(detected), tp_true / len(truth)


def test_drum_hits_from_mix(ctx):
    y, truth = drum_loop(120, bars=8)
    ingest_array(ctx, y)
    out = hits.run(ctx)
    assert out["from"] == "mix"
    for name, key in (("kick", "kicks"), ("snare", "snares")):
        precision, recall = _match([h["t"] for h in out["hits"][name]], truth[key])
        assert precision >= 0.9, name
        assert recall >= 0.9, name
    _, hat_recall = _match([h["t"] for h in out["hits"]["hat"]], truth["hats"])
    assert hat_recall >= 0.9
    assert out["hits"]["bass"] == [] and out["hits"]["vocal"] == []
    assert all(0 <= h["v"] <= 1 for h in out["hits"]["kick"])


def test_vocal_entries_after_gaps():
    sr = ANALYSIS_SR
    y = np.zeros(sr * 6, np.float32)
    t = np.arange(sr) / sr
    for start in (0.5, 2.5, 4.5):
        i = int(start * sr)
        y[i : i + sr] = 0.5 * np.sin(2 * np.pi * 220 * t)
    entries = hits.vocal_entries(y, sr)
    assert [round(e["t"], 1) for e in entries] == [0.5, 2.5, 4.5]
