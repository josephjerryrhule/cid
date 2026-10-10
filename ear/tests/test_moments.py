import numpy as np

from core.audio import ANALYSIS_SR, CURVE_HOP, CURVE_HZ, onset_env
from stages import moments
from stages.structure import energy_curves
from tests.synth import SR, drop_track, drum_loop, impacts_track, pink_noise, riser_track


def _curves(y):
    import librosa

    return energy_curves(librosa.resample(y, orig_sr=SR, target_sr=ANALYSIS_SR), ANALYSIS_SR)


def _with_gap(gain=1.0):
    loop, _ = drum_loop(120, bars=8)
    y = loop.astype(float) + 0.05 * pink_noise(len(loop) / SR)[: len(loop)]
    a, b = int(10.0 * SR), int(10.07 * SR)
    y[a:b] = 0.0
    return (gain * y).astype(np.float32)


def test_silence_gap_found():
    c = _curves(_with_gap())
    found = [s for s in moments.detect_silences(c["rms"], CURVE_HZ) if 9.5 < s["t"] < 10.5]
    assert len(found) == 1
    assert abs(found[0]["t"] - 10.0) <= 0.02
    assert abs(found[0]["end"] - 10.07) <= 0.02


def test_drop_found_on_downbeat():
    y, truth = drop_track()
    c = _curves(y)
    drops = moments.detect_drops(c["energy"], c["low"], CURVE_HZ, list(truth["downbeats"]))
    assert len(drops) == 1
    assert abs(drops[0]["t"] - truth["drop"]) <= 2 / 60


def test_quiet_master_still_finds_drop_and_silence():
    y, truth = drop_track(gain=0.018)  # roughly 33 dB quieter
    c = _curves(y)
    drops = moments.detect_drops(c["energy"], c["low"], CURVE_HZ, list(truth["downbeats"]))
    assert len(drops) == 1
    c2 = _curves(_with_gap(gain=0.02))
    assert any(abs(s["t"] - 10.0) <= 0.02 for s in moments.detect_silences(c2["rms"], CURVE_HZ))


def test_riser_before_drop():
    y, truth = riser_track()
    c = _curves(y)
    risers = moments.detect_risers(c["high"], CURVE_HZ, [truth["drop"]])
    assert len(risers) == 1
    assert abs(risers[0]["t"] - truth["riser_start"]) <= 2.0
    assert risers[0]["end"] == truth["drop"]


def test_impacts_found():
    import librosa

    y, truth = impacts_track()
    env = onset_env(librosa.resample(y, orig_sr=SR, target_sr=ANALYSIS_SR), ANALYSIS_SR, CURVE_HOP)
    found = moments.detect_impacts(env, CURVE_HZ, exclude=[])
    times = [e["t"] for e in found]
    for t in truth["impacts"]:
        assert min(abs(np.array(times) - t)) <= 0.03


def test_step_candidates_without_beats():
    energy = np.concatenate([np.full(500, 0.1), np.full(500, 0.9)])
    cands = moments.step_candidates(energy, 100)
    assert any(abs(t - 5.0) <= 0.1 for t in cands)


def test_vocal_changes_debounced():
    curve = np.zeros(2000)
    curve[300:800] = 0.8
    curve[850:1200] = 0.8  # 0.5 s gap is merged
    curve[1500:1800] = 0.8
    events = moments.detect_vocal_changes(curve, 100)
    assert [(e["type"], round(e["t"])) for e in events] == [
        ("vocal_in", 3), ("vocal_out", 12), ("vocal_in", 15), ("vocal_out", 18)
    ]


def test_cuts_prefer_phrase_starts_and_boundaries():
    downbeats = list(0.5 + 2.0 * np.arange(30))
    energy = np.full(6100, 0.5)
    cuts = moments.rank_cuts(downbeats, [16.5], energy, 100, 61.0)
    times = [c["t"] for c in cuts]
    assert 16.5 in times
    assert all(c["type"] == "cut" and 0 <= c["strength"] <= 1 for c in cuts)
    assert all(b - a >= 4.0 for a, b in zip(times, times[1:]))


def test_holds_only_in_steady_quiet_sections():
    energy = np.concatenate([np.full(1000, 0.3), np.linspace(0, 1, 1000)])
    sections = [{"start": 0.0, "end": 10.0}, {"start": 10.0, "end": 20.0}]
    holds = moments.detect_holds(sections, energy, 100, busy_times=[])
    assert [(h["t"], h["end"]) for h in holds] == [(0.0, 10.0)]
