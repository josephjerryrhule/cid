"""Stage 0: decode any input (file, URL, video) into the two WAVs every other stage reads."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from core.context import Context
from core.media import ffmpeg, probe
from core.pipeline import Fatal

VERSION = 1
DEPS: tuple[str, ...] = ()
SR = 44100
SPEECH_SR = 16000
LONG_INPUT_S = 20 * 60
SHORT_INPUT_S = 3.0


def resolve_source(src: str, downloads: Path) -> Path:
    """Return a local file for src, downloading URLs with yt-dlp."""
    if src.startswith(("http://", "https://")):
        downloads.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "wav", "--no-playlist",
             "-o", str(downloads / "%(id)s.%(ext)s"), "--print", "after_move:filepath", src],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 or not r.stdout.strip():
            raise Fatal(f"download failed: {r.stderr.strip()[-300:]}")
        return Path(r.stdout.strip().splitlines()[-1])
    p = Path(src).expanduser()
    if not p.exists():
        raise Fatal(f"no such file: {p}")
    return p.resolve()


def run(ctx: Context) -> dict:
    src = Path(ctx.options["source"])
    meta = probe(src)
    trim: list[str] = []
    if ctx.options.get("start") is not None:
        trim += ["-ss", str(ctx.options["start"])]
    if ctx.options.get("end") is not None:
        trim += ["-to", str(ctx.options["end"])]
    master, speech = ctx.path("master.wav"), ctx.path("speech.wav")
    ffmpeg(["-i", str(src), *trim, "-map", "0:a:0", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(master)])
    ffmpeg(["-i", str(master), "-ac", "1", "-ar", str(SPEECH_SR), "-c:a", "pcm_s16le", str(speech)])
    info = sf.info(str(master))
    duration = info.frames / info.samplerate
    if duration < 0.05:
        raise Fatal("the selected range contains no audio")
    y, _ = sf.read(str(speech), dtype="float32")
    silent = bool(y.size == 0 or np.max(np.abs(y)) < 1e-4)
    warnings = []
    if duration > LONG_INPUT_S:
        warnings.append("input is over 20 minutes; consider --start/--end")
    return {
        "source": str(src),
        "file": src.name,
        "duration": round(duration, 3),
        "sr": SR,
        "short": duration < SHORT_INPUT_S,
        "silent": silent,
        "warnings": warnings,
        "has_video": meta["has_video"],
        "video_fps": meta["video_fps"],
    }
