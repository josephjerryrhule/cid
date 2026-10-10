#!/usr/bin/env python3
"""/listen: give Claude an ear. Usage is documented in SKILL.md."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from core import config
from core.cache import cache_root, source_key
from core.context import Context
from core.pipeline import Fatal, run_pipeline
from core.registry import build_stages
from core.timefmt import parse_time
from stages.ingest import resolve_source


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="listen", description="Listen to audio, a URL or a video soundtrack.")
    p.add_argument("source")
    p.add_argument("question", nargs="?")
    p.add_argument("--fast", action="store_true", help="skip stems and the hearing model")
    p.add_argument("--start", help="SS, MM:SS or HH:MM:SS")
    p.add_argument("--end", help="SS, MM:SS or HH:MM:SS")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--ref", help="reference track for mix comparison")
    p.add_argument("--no-cloud", action="store_true", help="never upload audio")
    p.add_argument("--stages", default="", help="comma-separated stages to force rerun")
    p.add_argument("--export", help="copy cues.json and curves.json into this folder")
    return p


def run(argv: list[str]) -> tuple[int, dict]:
    a = _parser().parse_args(argv)
    config.load_env()
    try:
        start = parse_time(a.start) if a.start else None
        end = parse_time(a.end) if a.end else None
        downloads = cache_root() / "downloads"
        src = resolve_source(a.source, downloads)
        workdir = cache_root() / source_key(src, start, end)
        workdir.mkdir(parents=True, exist_ok=True)
        options = {"source": str(src), "start": start, "end": end}
        if a.ref:
            options["ref"] = str(resolve_source(a.ref, downloads))
        ctx = Context(workdir=workdir, fps=a.fps, fast=a.fast, no_cloud=a.no_cloud, options=options)
        force = {s.strip() for s in a.stages.split(",") if s.strip()} | {"assemble"}
        if a.ref or ctx.read("mix").get("ref"):
            force.add("mix")
        status = run_pipeline(ctx, build_stages(), force=force)
    except (Fatal, ValueError) as e:
        print(f"listen: {e}", file=sys.stderr)
        return 2, {"status": {}, "workdir": None}
    if a.export:
        dest = Path(a.export).expanduser()
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("cues.json", "curves.json"):
            shutil.copy2(workdir / name, dest / name)
    return 0, {"status": status, "workdir": workdir}


def main() -> None:
    code, info = run(sys.argv[1:])
    if code == 0:
        work = info["workdir"]
        print("stages: " + ", ".join(f"{k} {v}" for k, v in info["status"].items()))
        print(f"workdir: {work}")
        print(f"cues: {work / 'cues.json'}")
        print(f"curves: {work / 'curves.json'}")
        print()
        print((work / "report.md").read_text())
    sys.exit(code)


if __name__ == "__main__":
    main()
