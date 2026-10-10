import pytest

from core.audio import ANALYSIS_SR
from stages import classify
from tests.synth import pink_noise, syllables, chords


@pytest.mark.parametrize(
    "beat_conf, vocal_ratio, speech_mod, harmonic, has_video, expected",
    [
        (0.9, 0.05, 0.3, 0.7, False, "music"),
        (0.9, 0.30, 0.3, 0.7, False, "music"),
        (0.9, 0.30, 0.3, 0.7, True, "mixed"),
        (0.2, 0.80, 0.6, 0.7, False, "speech"),
        (0.2, 0.05, 0.2, 0.8, False, "music"),
        (0.2, 0.05, 0.2, 0.3, False, "sfx"),
        (0.2, None, 0.6, 0.7, False, "speech"),
        (0.2, None, 0.2, 0.3, False, "sfx"),
        (0.9, None, 0.6, 0.5, False, "music"),
    ],
)
def test_classify_rules(beat_conf, vocal_ratio, speech_mod, harmonic, has_video, expected):
    assert classify.classify(beat_conf, vocal_ratio, speech_mod, harmonic, has_video) == expected


def test_speech_modulation_separates_syllables_from_noise():
    import librosa

    speech = librosa.resample(syllables(8), orig_sr=44100, target_sr=ANALYSIS_SR)
    noise = librosa.resample(pink_noise(8), orig_sr=44100, target_sr=ANALYSIS_SR)
    assert classify.speech_modulation(speech, ANALYSIS_SR) >= 0.6
    assert classify.speech_modulation(noise, ANALYSIS_SR) <= 0.4


def test_harmonic_ratio_separates_chords_from_noise():
    import librosa

    tonal = librosa.resample(chords([[57, 60, 64]], 4.0), orig_sr=44100, target_sr=ANALYSIS_SR)
    noise = librosa.resample(pink_noise(4), orig_sr=44100, target_sr=ANALYSIS_SR)
    assert classify.harmonic_ratio(tonal, ANALYSIS_SR) >= 0.6
    assert classify.harmonic_ratio(noise, ANALYSIS_SR) < 0.6
