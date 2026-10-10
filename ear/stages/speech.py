"""Stage 5: word-timestamped transcript with mlx-whisper (local, Apple Silicon)."""
from __future__ import annotations

import os

from core.context import Context

VERSION = 1
DEPS = ("ingest", "stems", "classify")
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
MAX_NO_SPEECH = 0.6
MIN_LOGPROB = -1.0


def clean_segments(raw: list[dict]) -> list[dict]:
    out = []
    for s in raw:
        if s.get("no_speech_prob", 0.0) > MAX_NO_SPEECH or s.get("avg_logprob", 0.0) < MIN_LOGPROB:
            continue
        text = s.get("text", "").strip()
        if not text:
            continue
        words = [
            {"t0": round(float(w["start"]), 3), "t1": round(float(w["end"]), 3), "w": w["word"].strip()}
            for w in s.get("words", [])
        ]
        out.append({"t0": round(float(s["start"]), 3), "t1": round(float(s["end"]), 3), "text": text, "words": words})
    return out


def find_pauses(segments: list[dict], min_gap: float = 0.4) -> list[dict]:
    words = [w for s in segments for w in s.get("words", [])]
    return [
        {"t": a["t1"], "end": b["t0"]}
        for a, b in zip(words, words[1:])
        if b["t0"] - a["t1"] >= min_gap
    ]


def run(ctx: Context) -> dict:
    import mlx_whisper

    stems = ctx.data("stems")
    src = ctx.path(stems["stems"]["vocals"]) if stems else ctx.path("speech.wav")
    model = os.environ.get("LISTEN_WHISPER_MODEL", DEFAULT_MODEL)
    res = mlx_whisper.transcribe(
        str(src), path_or_hf_repo=model, word_timestamps=True, condition_on_previous_text=False
    )
    segments = clean_segments(res.get("segments", []))
    return {"language": res.get("language"), "model": model, "segments": segments, "pauses": find_pauses(segments)}
