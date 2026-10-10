import platform
import subprocess

import pytest

from stages import speech
from tests.helpers import ingest_array


def test_clean_segments_drops_non_speech_and_hallucinations():
    raw = [
        {"start": 0.0, "end": 1.0, "text": " Hello there", "no_speech_prob": 0.1, "avg_logprob": -0.3,
         "words": [{"start": 0.0, "end": 0.4, "word": " Hello"}, {"start": 0.5, "end": 1.0, "word": " there"}]},
        {"start": 1.0, "end": 2.0, "text": " Thanks for watching", "no_speech_prob": 0.9, "avg_logprob": -0.3, "words": []},
        {"start": 2.0, "end": 3.0, "text": " la la", "no_speech_prob": 0.2, "avg_logprob": -1.5, "words": []},
    ]
    segs = speech.clean_segments(raw)
    assert [s["text"] for s in segs] == ["Hello there"]
    assert segs[0]["words"] == [{"t0": 0.0, "t1": 0.4, "w": "Hello"}, {"t0": 0.5, "t1": 1.0, "w": "there"}]


def test_find_pauses():
    segs = [{"words": [{"t0": 0.0, "t1": 0.4, "w": "a"}, {"t0": 1.2, "t1": 1.5, "w": "b"}, {"t0": 1.6, "t1": 2.0, "w": "c"}]}]
    assert speech.find_pauses(segs) == [{"t": 0.4, "end": 1.2}]


@pytest.mark.models
@pytest.mark.skipif(platform.system() != "Darwin", reason="uses macOS say")
def test_transcribes_spoken_phrase(ctx, tmp_path):
    import librosa

    aiff = tmp_path / "phrase.aiff"
    subprocess.run(["say", "-o", str(aiff), "Finally home. The candles are lit."], check=True)
    y, _ = librosa.load(str(aiff), sr=44100)
    ingest_array(ctx, y)
    out = speech.run(ctx)
    text = " ".join(s["text"] for s in out["segments"]).lower()
    assert "candles" in text
    assert out["segments"][0]["words"]
