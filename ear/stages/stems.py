"""Stage 1: split the mix into vocals, drums, bass and other with demucs.

demucs runs in its own throwaway environment through uvx, so its torch and
torchaudio pins never conflict with the analysis environment.
"""
from __future__ import annotations

import shutil
import subprocess

from core.context import Context
from core.pipeline import StageUnavailable

VERSION = 1
DEPS = ("ingest",)
STEMS = ("vocals", "drums", "bass", "other")
DEMUCS_CMD = [
    "uvx", "--python", "3.11",
    "--from", "demucs==4.0.1",
    "--with", "torch==2.5.1", "--with", "torchaudio==2.5.1", "--with", "soundfile",
    "demucs",
]
TIMEOUT_S = 30 * 60


def run(ctx: Context) -> dict:
    tmp = ctx.path("stems_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        r = subprocess.run(
            [*DEMUCS_CMD, "-n", "htdemucs", "-o", str(tmp), str(ctx.path("master.wav"))],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except FileNotFoundError:
        raise StageUnavailable("uvx not found; install uv")
    except subprocess.TimeoutExpired:
        raise StageUnavailable("demucs timed out after 30 minutes")
    if r.returncode != 0:
        raise StageUnavailable(f"demucs failed: {r.stderr.strip()[-300:]}")
    produced = tmp / "htdemucs" / "master"
    files = {}
    for s in STEMS:
        src = produced / f"{s}.wav"
        if not src.exists():
            raise StageUnavailable(f"demucs did not write {s}.wav")
        dst = ctx.path(f"stem_{s}.wav")
        shutil.move(str(src), str(dst))
        files[s] = dst.name
    shutil.rmtree(tmp, ignore_errors=True)
    return {"model": "htdemucs", "stems": files}
