import json
import os

import pytest

import backends
from backends import aistudio
from core.pipeline import StageUnavailable
from stages import hear
from tests.helpers import ingest_array
from tests.synth import pink_noise

RESULT = {
    "kind": "music",
    "overall": {"genre": "house", "mood": "dark", "instruments": ["kick", "pad"], "vocals": "none", "production_style": "clean"},
    "sections": [{"id": "s1", "label": "intro", "description": "pad swells", "what_changes": "start"}],
    "extra_moments": [{"approx_t": 3.0, "what": "reverse cymbal"}],
    "mix_impressions": ["wide pad"],
    "speech": {"tone": "", "emotion": "", "room": "", "noise": ""},
}


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LISTEN_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LISTEN_BACKEND", "aistudio")
    monkeypatch.delenv("LISTEN_DAILY_CAP", raising=False)


def _prepare(ctx, seconds=6):
    ingest_array(ctx, pink_noise(seconds))
    ctx.write("rhythm", {"_v": 1, "bpm": 124.0, "meter": "4/4", "confidence": 0.9})
    ctx.write("classify", {"_v": 1, "kind": "music"})
    ctx.write("structure", {"_v": 1, "sections": [{"id": "s1", "label": "intro", "start": 0.0, "end": float(seconds), "energy": 0.3}],
                            "key": {"tonic": "A", "mode": "minor", "confidence": 0.8}})
    ctx.write("moments", {"_v": 1, "moments": [{"t": 2.0, "type": "impact", "strength": 0.9}]})


def test_plan_chunks():
    assert hear.plan_chunks(300) == [(0.0, 300.0)]
    chunks = hear.plan_chunks(1300)
    assert chunks[0] == (0.0, 600.0)
    assert chunks[1][0] == 585.0
    assert chunks[-1][1] == 1300.0


def test_prompt_contains_measured_skeleton(ctx):
    _prepare(ctx)
    data = {s: ctx.data(s) for s in ("ingest", "rhythm", "structure", "moments", "classify")}
    prompt = hear.build_prompt(data, 0.0, 6.0)
    assert "124.0 BPM" in prompt and "4/4" in prompt and "A minor" in prompt
    assert "s1 0.00 to 6.00 s" in prompt
    assert "2.00 s impact" in prompt
    assert chr(0x2014) not in prompt and chr(0x2013) not in prompt


def test_one_request_then_cached(ctx, monkeypatch):
    _prepare(ctx)
    calls = []

    def fake(audio, prompt, schema_model, **k):
        calls.append(prompt)
        assert audio.suffix == ".ogg" and audio.exists()
        return RESULT

    monkeypatch.setattr(aistudio, "hear", fake)
    out = hear.run(ctx)
    assert out["overall"]["genre"] == "house"
    assert out["notice"] and "Google" in out["notice"]
    ctx.write("hear", {**out, "_v": hear.VERSION})
    again = hear.run(ctx)
    assert len(calls) == 1
    assert again["prompt_sha"] == out["prompt_sha"]


def test_rate_limit_becomes_unavailable(ctx, monkeypatch):
    _prepare(ctx)

    def limited(*a, **k):
        raise backends.RateLimited("quota")

    monkeypatch.setattr(aistudio, "hear", limited)
    with pytest.raises(StageUnavailable, match="rate limited"):
        hear.run(ctx)


def test_daily_cap(ctx, monkeypatch):
    _prepare(ctx)
    monkeypatch.setenv("LISTEN_DAILY_CAP", "1")
    monkeypatch.setattr(aistudio, "hear", lambda *a, **k: RESULT)
    hear.run(ctx)
    ctx.write("hear", {})
    with pytest.raises(StageUnavailable, match="daily hearing cap"):
        hear.run(ctx)


def test_merge_offsets_chunk_moments():
    second = {**RESULT, "sections": [{"id": "s2", "label": "drop", "description": "", "what_changes": ""}],
              "extra_moments": [{"approx_t": 10.0, "what": "crash"}]}
    merged = hear.merge([RESULT, second], [(0.0, 600.0), (585.0, 900.0)])
    assert [s["id"] for s in merged["sections"]] == ["s1", "s2"]
    assert [m["approx_t"] for m in merged["extra_moments"]] == [3.0, 595.0]


def test_aistudio_retries_on_429(monkeypatch, tmp_path):
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    audio = tmp_path / "a.ogg"
    audio.write_bytes(b"OggS")
    attempts = []

    class FakeModels:
        def generate_content(self, **kwargs):
            attempts.append(1)
            if len(attempts) < 3:
                raise errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
            class R:
                text = json.dumps(RESULT)
            return R()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(aistudio.genai, "Client", FakeClient)
    out = aistudio.hear(audio, "prompt", hear.Hearing, sleep=lambda s: None)
    assert out["overall"]["genre"] == "house"
    assert len(attempts) == 3


def test_aistudio_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(backends.BackendError, match="GEMINI_API_KEY"):
        aistudio.hear(tmp_path / "a.ogg", "p", hear.Hearing)


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("LISTEN_LIVE") != "1", reason="set LISTEN_LIVE=1 to spend one real request")
def test_live_hearing(ctx):
    from core import config

    config.load_env()
    _prepare(ctx, seconds=8)
    out = hear.run(ctx)
    assert out["overall"]["mood"]
