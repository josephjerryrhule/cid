"""The ordered stage list and when each stage runs."""
from __future__ import annotations

from core.context import Context
from core.pipeline import Stage
from stages import assemble, classify, hear, hits, ingest, mix, moments, rhythm, see, speech, stems, structure


def _ingest(ctx: Context) -> dict:
    return ctx.data("ingest") or {}


def _playable(ctx: Context) -> bool:
    return not _ingest(ctx).get("silent")


def _analysable(ctx: Context) -> bool:
    i = _ingest(ctx)
    return not i.get("silent") and not i.get("short")


def _speech_enabled(ctx: Context) -> bool:
    if ctx.fast or not _playable(ctx):
        return False
    c = ctx.data("classify") or {}
    ratio = (c.get("features") or {}).get("vocal_ratio")
    return c.get("kind") in ("speech", "mixed") or (ratio is not None and ratio >= 0.1)


def _stage(mod, name, enabled=None) -> Stage:
    if enabled is None:
        return Stage(name, mod.VERSION, mod.DEPS, mod.run)
    return Stage(name, mod.VERSION, mod.DEPS, mod.run, enabled)


def build_stages() -> list[Stage]:
    return [
        _stage(ingest, "ingest"),
        _stage(stems, "stems", lambda c: not c.fast and _analysable(c)),
        _stage(rhythm, "rhythm", _analysable),
        _stage(hits, "hits", _analysable),
        _stage(classify, "classify", _playable),
        _stage(structure, "structure", _analysable),
        _stage(moments, "moments", _analysable),
        _stage(mix, "mix", _playable),
        _stage(speech, "speech", _speech_enabled),
        _stage(hear, "hear", lambda c: not c.fast and not c.no_cloud and _playable(c)),
        _stage(see, "see", _playable),
        _stage(assemble, "assemble"),
    ]
