"""ffmpeg and ffprobe wrappers. Arguments are always lists; nothing goes through a shell."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from core.pipeline import Fatal


def ffmpeg(args: list[str]) -> None:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise Fatal(f"ffmpeg failed: {r.stderr.strip()[-300:]}")


def probe(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise Fatal(f"ffprobe could not read {Path(path).name}")
    info = json.loads(r.stdout)
    streams = info.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    video = [
        s
        for s in streams
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic", 0)
    ]
    if not audio:
        raise Fatal(f"{Path(path).name} has no audio stream")
    fps = None
    if video:
        num, _, den = video[0].get("avg_frame_rate", "0/1").partition("/")
        if float(den or 0):
            fps = round(float(num) / float(den), 3)
    return {
        "duration": float(info.get("format", {}).get("duration", 0.0)),
        "channels": int(audio[0].get("channels", 2)),
        "has_video": bool(video),
        "video_fps": fps,
    }
