import platform
import subprocess

import numpy as np
import pytest

from core.pipeline import StageUnavailable
from stages import tags
from tests.helpers import ingest_array
from tests.synth import drum_loop

ONTOLOGY = [
    {"id": "/m/music", "name": "Music", "child_ids": ["/m/inst", "/m/genre"]},
    {"id": "/m/inst", "name": "Musical instrument", "child_ids": ["/m/guitar"]},
    {"id": "/m/guitar", "name": "Guitar", "child_ids": []},
    {"id": "/m/genre", "name": "Music genre", "child_ids": ["/m/house"]},
    {"id": "/m/house", "name": "House music", "child_ids": []},
    {"id": "/m/human", "name": "Human sounds", "child_ids": ["/m/speech"]},
    {"id": "/m/speech", "name": "Speech", "child_ids": []},
    {"id": "/m/things", "name": "Sounds of things", "child_ids": ["/m/glass"]},
    {"id": "/m/glass", "name": "Glass", "child_ids": []},
    {"id": "/m/env", "name": "Channel, environment and background", "child_ids": ["/m/hum"]},
    {"id": "/m/hum", "name": "Hum", "child_ids": []},
]


def test_class_groups_from_ontology():
    g = tags.class_groups(ONTOLOGY, ["Guitar", "House music", "Speech", "Glass", "Hum", "Music", "Unknown"])
    assert g == {"Guitar": "instrument", "House music": "genre", "Speech": "voice", "Glass": "sfx",
                 "Hum": "ambience", "Music": "music", "Unknown": "sfx"}


def test_to_spans_threshold_min_length_and_merge():
    p = np.zeros(1000)
    p[100:300] = 0.9   # 2 s span
    p[320:400] = 0.9   # gap 0.2 s: merges
    p[600:610] = 0.9   # 0.1 s: too short
    spans = tags.to_spans(p, 100)
    assert len(spans) == 1
    a, b, conf = spans[0]
    assert abs(a - 1.0) <= 0.05 and abs(b - 4.0) <= 0.05 and conf > 0.8


def test_chunked_inference_matches_length():
    class FakeSED:
        def inference(self, audio):
            n = audio.shape[1] // 320 + 1
            return np.full((1, n, 527), 0.5, dtype=np.float32)

    y = np.zeros(int(150.3 * tags.SR), np.float32)
    fw = tags.framewise(y, FakeSED(), chunk_s=60.0)
    assert fw.shape[1] == 527
    assert abs(fw.shape[0] - 150.3 * tags.FRAME_HZ) <= 3


def test_missing_assets_is_unavailable(ctx, monkeypatch):
    ingest_array(ctx, np.zeros(44100, np.float32))

    def unavailable():
        raise StageUnavailable("could not download Cnn14_DecisionLevelMax.pth from https://example: offline")

    monkeypatch.setattr(tags, "ensure_panns", unavailable)
    with pytest.raises(StageUnavailable, match="could not download"):
        tags.run(ctx)


@pytest.mark.models
@pytest.mark.skipif(platform.system() != "Darwin", reason="uses macOS say")
def test_speech_is_tagged(ctx, tmp_path):
    import librosa

    aiff = tmp_path / "s.aiff"
    subprocess.run(["say", "-o", str(aiff), "This is a short spoken sentence for the tagger to hear clearly."], check=True)
    y, _ = librosa.load(str(aiff), sr=44100)
    ingest_array(ctx, y)
    out = tags.run(ctx)
    speech = [s for s in out["spans"] if s["label"] == "Speech"]
    assert speech and speech[0]["group"] == "voice"
    assert sum(s["end"] - s["t"] for s in speech) >= 0.5 * len(y) / 44100


@pytest.mark.models
def test_drum_loop_is_tagged(ctx):
    y, _ = drum_loop(120, bars=8)
    ingest_array(ctx, y)
    labels = {s["label"] for s in tags.run(ctx)["spans"]}
    drumish = {"Drum", "Drum kit", "Snare drum", "Bass drum", "Percussion", "Drum machine", "Hi-hat", "Drum roll"}
    assert labels & drumish, labels


def test_merge_sources_unions_spans_and_keeps_sources():
    found = {
        ("Drum kit", "drums"): [(8.0, 20.0, 0.6)],
        ("Drum kit", "mix"): [(18.0, 30.0, 0.4)],
        ("Speech", "vocals"): [(5.8, 6.9, 0.5)],
    }
    merged = tags.merge_sources(found)
    assert merged["Drum kit"] == [{"t": 8.0, "end": 30.0, "confidence": 0.6, "sources": ["drums", "mix"]}]
    assert merged["Speech"][0]["sources"] == ["vocals"]


def test_generic_labels_are_dropped():
    assert "Music" in tags.GENERIC and "Musical instrument" in tags.GENERIC
