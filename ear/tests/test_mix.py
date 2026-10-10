import numpy as np

from stages import mix
from tests.helpers import ingest_array
from tests.synth import SR, bandpass, lowpass, pink_noise, to_lufs, write


def _stereo(y):
    return np.stack([y, y], axis=1)


def test_integrated_loudness_matches_target():
    st = to_lufs(_stereo(pink_noise(10)), -14.0)
    m = mix.measure(st, SR)
    assert abs(m["lufs"] - (-14.0)) <= 0.5


def test_true_peak_of_sine():
    t = np.arange(SR * 3) / SR
    m = mix.measure(_stereo(0.5 * np.sin(2 * np.pi * 997 * t)), SR)
    assert abs(m["true_peak"] - (-6.02)) <= 0.2


def test_lra_constant_vs_dynamic():
    steady = to_lufs(_stereo(pink_noise(30)), -20.0)
    assert mix.measure(steady, SR)["lra"] < 1.5
    loud = to_lufs(_stereo(pink_noise(10, seed=1)), -10.0)
    quiet = to_lufs(_stereo(pink_noise(10, seed=2)), -30.0)
    dynamic = np.concatenate([loud, quiet, loud])
    assert mix.measure(dynamic, SR)["lra"] > 10


def test_clipping_is_flagged():
    t = np.arange(SR * 2) / SR
    y = np.clip(2.0 * np.sin(2 * np.pi * 100 * t), -1, 1)
    m = mix.measure(_stereo(y), SR)
    assert m["clipped_samples"] > 0
    assert any(f.startswith("clipping") for f in m["flags"])


def test_low_mid_buildup_flag():
    base = pink_noise(10)
    mud = bandpass(pink_noise(10, seed=9), 200, 400)
    y = base + 3.0 * mud / np.max(np.abs(mud)) * np.max(np.abs(base))
    flagged = mix.measure(_stereo(0.5 * y / np.max(np.abs(y))), SR)["flags"]
    clean = mix.measure(_stereo(base), SR)["flags"]
    assert any("200 to 400 Hz" in f for f in flagged)
    assert not any("200 to 400 Hz" in f for f in clean)


def test_out_of_phase_low_end_is_flagged():
    bass = lowpass(pink_noise(10, seed=4), 100)
    air = pink_noise(10, seed=5)
    left = air + bass
    right = pink_noise(10, seed=6) - bass
    m = mix.measure(np.stack([left, right], axis=1) * 0.4, SR)
    assert any("low end not mono" in f for f in m["flags"])


def test_mono_source_has_no_stereo_flags():
    m = mix.measure(_stereo(pink_noise(5)), SR)
    assert m["mono"] is True
    assert m["width"] == 0.0
    assert not any("mono" in f or "phase" in f for f in m["flags"])


def test_reference_comparison(ctx, tmp_path):
    y = to_lufs(_stereo(pink_noise(8)), -16.0)
    ingest_array(ctx, y[:, 0], stereo=y)
    ref = tmp_path / "ref.wav"
    write(ref, y)
    ctx.options["ref"] = str(ref)
    out = mix.run(ctx)
    assert out["ref"]["file"] == "ref.wav"
    assert abs(out["ref"]["lufs"] - out["lufs"]) < 0.2
    assert out["ref"]["deltas"] == []
