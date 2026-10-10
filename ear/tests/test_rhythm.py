import numpy as np
import pytest

from stages import rhythm
from tests.helpers import ingest_array
from tests.synth import drum_loop


def test_number_beats_and_meter():
    beats = np.arange(0, 8, 0.5)
    downbeats = beats[::4]
    numbered = rhythm.number_beats(beats, downbeats)
    assert numbered[:5] == [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1)]
    assert rhythm.meter(numbered) == "4/4"
    three = rhythm.number_beats(beats[:12], beats[:12:3])
    assert rhythm.meter(three) == "3/4"


def test_confidence_steady_vs_jittery():
    steady = np.arange(0, 30, 0.5)
    rng = np.random.default_rng(0)
    jittery = np.cumsum(0.5 * (1 + 0.3 * rng.uniform(-1, 1, 60)))
    assert rhythm.confidence(steady) > 0.95
    assert rhythm.confidence(jittery) < 0.5
    assert rhythm.confidence(np.array([0.0, 0.5])) == 0.0


def test_refine_snaps_to_nearby_peak():
    times = np.arange(0, 2, 0.005)
    env = np.zeros_like(times)
    env[np.searchsorted(times, 1.012)] = 10.0
    out = rhythm.refine(np.array([1.0]), env, times)
    assert abs(out[0] - 1.012) <= 0.005  # one grid step


def test_tempo_changes_detects_switch():
    beats = np.concatenate([np.arange(0, 16, 0.5), 16 + np.arange(0, 16, 60 / 140)])
    changes = rhythm.tempo_changes(beats, 120.0)
    assert changes and abs(changes[0]["bpm"] - 140) < 2


@pytest.mark.models
def test_drum_loop_beats_are_frame_accurate(ctx):
    y, truth = drum_loop(120, bars=16, bar_chords=True)
    ingest_array(ctx, y)
    out = rhythm.run(ctx)
    assert abs(out["bpm"] - 120) < 1
    assert out["meter"] == "4/4"
    detected = np.array([b["t"] for b in out["beats"]])
    inner = truth["beats"][4:-4]
    errors = np.array([np.min(np.abs(detected - t)) for t in inner])
    assert np.median(errors) <= 0.010
    assert np.percentile(errors, 95) <= 1 / 60
    downs = np.array(out["downbeats"])
    for t in truth["downbeats"][1:-1]:
        assert np.min(np.abs(downs - t)) < 0.035


@pytest.mark.models
def test_jittery_loop_is_low_confidence(ctx):
    y, _ = drum_loop(120, bars=16, jitter=0.15, seed=3)
    ingest_array(ctx, y)
    assert rhythm.run(ctx)["low_confidence"] is True
