"""Stage 6: a hearing model describes the sound, given the measured skeleton.

Request discipline: exactly one request per chunk, structured output, and a result
cached by prompt hash so the same audio and skeleton are never asked about twice.
Bump PROMPT_VERSION (and VERSION) whenever PROMPT changes.
"""
from __future__ import annotations

import hashlib
import os
from typing import Callable

from pydantic import BaseModel

import backends
from core import usage
from core.context import Context
from core.media import ffmpeg
from core.pipeline import Fatal, StageUnavailable

VERSION = 1
DEPS = ("ingest", "rhythm", "structure", "moments", "classify")
PROMPT_VERSION = 1
CHUNK_S = 600.0
OVERLAP_S = 15.0


class Overall(BaseModel):
    genre: str
    mood: str
    instruments: list[str]
    vocals: str
    production_style: str


class SectionNote(BaseModel):
    id: str
    label: str
    description: str
    what_changes: str


class ExtraMoment(BaseModel):
    approx_t: float
    what: str


class SpeechNote(BaseModel):
    tone: str
    emotion: str
    room: str
    noise: str


class Hearing(BaseModel):
    kind: str
    overall: Overall
    sections: list[SectionNote]
    extra_moments: list[ExtraMoment]
    mix_impressions: list[str]
    speech: SpeechNote


PROMPT = """You are listening to an audio clip for a motion designer and video editor.

Measured facts about this clip (trust these numbers; do not re-estimate them):
{skeleton}

Listen to the whole clip and fill the JSON schema:
- kind: one of music, speech, sfx, mixed.
- overall: genre, mood, instruments (list), vocals (describe them, or "none"), production_style.
- sections: one entry per measured section id, in order. label: a short name such as intro, verse, build, drop, chorus, break, bridge, outro; for speech use opening, point, aside, close. description: what you hear. what_changes: what differs from the previous section.
- extra_moments: notable events missing from the measured list, with approx_t in seconds from the start of this clip.
- mix_impressions: short observations about balance, space, dynamics and clarity.
- speech: tone, emotion, room, noise. Empty strings when there is no speech.
Be concrete and plain. No praise."""


def plan_chunks(duration: float, size: float = CHUNK_S, overlap: float = OVERLAP_S) -> list[tuple[float, float]]:
    if duration <= size:
        return [(0.0, float(duration))]
    chunks, start = [], 0.0
    while start < duration:
        end = min(start + size, float(duration))
        chunks.append((start, end))
        if end >= duration:
            break
        start = end - overlap
    return chunks


def build_prompt(data: dict, start: float, end: float) -> str:
    ingest = data.get("ingest") or {}
    rhythm = data.get("rhythm") or {}
    structure = data.get("structure") or {}
    kind = (data.get("classify") or {}).get("kind", "unknown")
    lines = [f"duration: {end - start:.1f} s", f"measured kind: {kind}"]
    if rhythm.get("bpm"):
        lines.append(f"tempo: {rhythm['bpm']:.1f} BPM, {rhythm.get('meter', 'unknown')} (confidence {rhythm.get('confidence', 0):.2f})")
    key = structure.get("key")
    if key:
        lines.append(f"key: {key['tonic']} {key['mode']} ({key['confidence']:.2f})")
    lines.append("sections:")
    for s in structure.get("sections", []):
        if s["end"] > start and s["start"] < end:
            lines.append(f"  {s['id']} {max(s['start'], start) - start:.2f} to {min(s['end'], end) - start:.2f} s, energy {s['energy']:.2f}")
    lines.append("moments:")
    for m in (data.get("moments") or {}).get("moments", []):
        if start <= m["t"] < end and m["type"] != "cut":
            lines.append(f"  {m['t'] - start:.2f} s {m['type']}")
    if ingest.get("has_video"):
        lines.append("this is the soundtrack of a video")
    return PROMPT.format(skeleton="\n".join(lines))


def merge(results: list[dict], chunks: list[tuple[float, float]]) -> dict:
    first = results[0]
    merged = {
        "kind": first["kind"],
        "overall": dict(first["overall"]),
        "sections": [],
        "extra_moments": [],
        "mix_impressions": [],
        "speech": first["speech"],
    }
    instruments = list(first["overall"].get("instruments", []))
    seen: set[str] = set()
    for r, (offset, _) in zip(results, chunks):
        for s in r["sections"]:
            if s["id"] not in seen:
                seen.add(s["id"])
                merged["sections"].append(s)
        for m in r["extra_moments"]:
            t = round(m["approx_t"] + offset, 2)
            if all(abs(t - x["approx_t"]) > 2 for x in merged["extra_moments"]):
                merged["extra_moments"].append({"approx_t": t, "what": m["what"]})
        for imp in r["mix_impressions"]:
            if imp not in merged["mix_impressions"]:
                merged["mix_impressions"].append(imp)
        for inst in r["overall"].get("instruments", []):
            if inst not in instruments:
                instruments.append(inst)
    merged["overall"]["instruments"] = instruments
    return merged


def get_backend(name: str) -> Callable:
    if name == "aistudio":
        from backends import aistudio

        return aistudio.hear
    raise StageUnavailable(f"unknown LISTEN_BACKEND {name!r}")


def _model(backend: str) -> str:
    if backend == "aistudio":
        from backends import aistudio

        return aistudio.model_name()
    return backend


def run(ctx: Context) -> dict:
    data = {s: ctx.data(s) for s in DEPS}
    duration = float(data["ingest"]["duration"])
    chunks = plan_chunks(duration)
    backend = os.environ.get("LISTEN_BACKEND", "aistudio")
    prompts = [build_prompt(data, a, b) for a, b in chunks]
    sha = hashlib.sha256((backend + "\n" + "\n".join(prompts)).encode()).hexdigest()[:16]
    prev = ctx.read("hear")
    if prev.get("prompt_sha") == sha and not prev.get("_unavailable"):
        return prev
    hear_fn = get_backend(backend)
    warnings, results = [], []
    for i, ((a, b), prompt) in enumerate(zip(chunks, prompts)):
        warn = usage.check_cap()
        if warn and warn not in warnings:
            warnings.append(warn)
        clip = ctx.path(f"hear_{i}.ogg")
        try:
            ffmpeg(["-i", str(ctx.path("master.wav")), "-ss", f"{a}", "-to", f"{b}",
                    "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k", str(clip)])
        except Fatal as e:
            raise StageUnavailable(f"could not prepare upload: {e}")
        try:
            results.append(hear_fn(clip, prompt, Hearing))
        except backends.RateLimited as e:
            raise StageUnavailable(f"rate limited: {e}")
        except backends.BackendError as e:
            raise StageUnavailable(str(e))
        usage.record()
    return {
        **merge(results, chunks),
        "prompt_sha": sha,
        "prompt_version": PROMPT_VERSION,
        "backend": backend,
        "model": _model(backend),
        "warnings": warnings,
        "notice": usage.first_upload_notice(backend),
    }
