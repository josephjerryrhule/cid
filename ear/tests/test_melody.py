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


def test_ghost_harmonics_removed():
    raw = [{"t": 0.0, "end": 1.0, "pitch": 57, "velocity": 0.8},
           {"t": 0.01, "end": 0.9, "pitch": 69, "velocity": 0.4},   # octave ghost
           {"t": 0.02, "end": 0.8, "pitch": 76, "velocity": 0.3},   # twelfth ghost
           {"t": 0.0, "end": 1.0, "pitch": 60, "velocity": 0.7}]    # real chord tone (minor third)
    pitches = sorted(n["pitch"] for n in melody.clean_notes(raw, "other"))
    assert pitches == [57, 60]


def test_bleed_notes_dropped_where_stem_is_quiet():
    rms = np.r_[np.full(100, 0.5), np.full(100, 0.001)]  # loud first second, near-silent second
    raw = [{"t": 0.2, "end": 0.6, "pitch": 60, "velocity": 0.6}, {"t": 1.2, "end": 1.6, "pitch": 62, "velocity": 0.6}]
    kept = melody.clean_notes(raw, "vocals", rms=rms, rms_hz=100)
    assert [n["pitch"] for n in kept] == [60]


def test_vocal_range_clamped():
    raw = [{"t": 0.0, "end": 0.5, "pitch": 29, "velocity": 0.6}, {"t": 0.5, "end": 1.0, "pitch": 64, "velocity": 0.6},
           {"t": 1.0, "end": 1.5, "pitch": 96, "velocity": 0.6}]
    assert [n["pitch"] for n in melody.clean_notes(raw, "vocals")] == [64]


def test_skyline_keeps_top_line():
    notes = [{"t": 0.0, "end": 0.5, "pitch": 60, "name": "", "velocity": 0.5}, {"t": 0.0, "end": 0.5, "pitch": 67, "name": "", "velocity": 0.5},
             {"t": 0.5, "end": 1.0, "pitch": 62, "name": "", "velocity": 0.5}]
    assert [n["pitch"] for n in melody.skyline(notes)] == [67, 62]


def test_hook_rejects_leaps_over_an_octave():
    notes, t = [], 0.0
    for _ in range(4):
        for p in (88, 45, 88, 88, 50):     # E6-A2-E6-E6: overtone artefact, not a melody
            notes.append({"t": t, "end": t + 0.2, "pitch": p, "name": "", "velocity": 0.5})
            t += 0.25
    assert melody.find_hook(notes) is None


def test_monotone_repetition_is_not_a_hook():
    notes = [{"t": i * 0.25, "end": i * 0.25 + 0.2, "pitch": 47, "name": "", "velocity": 0.5} for i in range(40)]
    assert melody.find_hook(notes) is None
