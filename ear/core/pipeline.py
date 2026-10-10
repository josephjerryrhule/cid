"""Runs stages in order with per-stage caching and independent failure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from core.context import Context


class Fatal(Exception):
    """Nothing useful can be produced (no audio, unreadable input). Stops the run."""


class StageUnavailable(Exception):
    """One stage cannot run. The reason is shown in the report; other stages continue."""


def _always(ctx: Context) -> bool:
    return True


@dataclass(frozen=True)
class Stage:
    name: str
    version: int
    deps: tuple[str, ...]
    run: Callable[[Context], dict]
    enabled: Callable[[Context], bool] = _always


def run_pipeline(ctx: Context, stages: Iterable[Stage], force: Iterable[str] = ()) -> dict[str, str]:
    force = set(force)
    status: dict[str, str] = {}
    rerun: set[str] = set()
    for st in stages:
        if not st.enabled(ctx):
            status[st.name] = "skipped"
            continue
        cached = ctx.read(st.name)
        stale = (
            st.name in force
            or cached.get("_v") != st.version
            or bool(cached.get("_unavailable"))
            or any(d in rerun for d in st.deps)
        )
        if not stale:
            status[st.name] = "cached"
            continue
        try:
            data = dict(st.run(ctx))
            data["_v"] = st.version
            data.pop("_unavailable", None)
            ctx.write(st.name, data)
            status[st.name] = "ok"
        except Fatal:
            raise
        except StageUnavailable as e:
            ctx.write(st.name, {"_v": st.version, "_unavailable": str(e)})
            status[st.name] = f"unavailable: {e}"
        except Exception as e:  # a broken stage must never take the whole run down
            reason = f"{type(e).__name__}: {e}"
            ctx.write(st.name, {"_v": st.version, "_unavailable": reason})
            status[st.name] = f"unavailable: {reason}"
        rerun.add(st.name)
    return status
