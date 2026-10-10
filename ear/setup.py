#!/usr/bin/env python3
"""Preflight and installer for /listen. Stdlib only: it runs before the venv exists.

Exit codes: 0 ready, 2 missing binaries, 3 missing API key, 4 both, 5 Python environment not installed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINARIES = ("ffmpeg", "ffprobe", "yt-dlp", "uv")
BREW_PACKAGES = {"ffmpeg": "ffmpeg", "ffprobe": "ffmpeg", "yt-dlp": "yt-dlp", "uv": "uv"}
CONFIG_DIR = Path(os.environ.get("LISTEN_CONFIG", Path.home() / ".config" / "listen"))
ENV_FILE = CONFIG_DIR / ".env"
READY_MARKER = HERE / ".venv" / ".listen-ready"
ENV_TEMPLATE = """# /listen configuration (this file is private: chmod 600)
# Hearing backend (only aistudio is supported)
# LISTEN_BACKEND=aistudio
# Free key: https://aistudio.google.com/apikey
# GEMINI_API_KEY=
# LISTEN_MODEL=gemini-flash-latest
# LISTEN_DAILY_CAP=200
# LISTEN_WHISPER_MODEL=mlx-community/whisper-large-v3-turbo
"""
MESSAGES = {
    2: "missing binaries: {missing}. Run: python3 {here}/setup.py",
    3: "no GEMINI_API_KEY in {env}. Add one, or run listen with --no-cloud.",
    4: "missing binaries ({missing}) and no GEMINI_API_KEY. Run: python3 {here}/setup.py",
    5: "Python environment not installed. Run: python3 {here}/setup.py",
}


def read_env() -> dict:
    """Shell exports first, then the config file on top (same precedence as core.config)."""
    env = {k: v for k, v in os.environ.items() if k.startswith(("LISTEN_", "GEMINI_"))}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def status() -> dict:
    env = read_env()
    missing = [b for b in BINARIES if not shutil.which(b)]
    backend = env.get("LISTEN_BACKEND", "aistudio")
    has_key = backend != "aistudio" or bool(env.get("GEMINI_API_KEY"))
    env_ready = READY_MARKER.exists()
    if missing and not has_key:
        code = 4
    elif missing:
        code = 2
    elif not env_ready:
        code = 5
    elif not has_key:
        code = 3
    else:
        code = 0
    return {
        "status_code": code,
        "missing_binaries": missing,
        "backend": backend,
        "has_key": has_key,
        "env_ready": env_ready,
        "config_file": str(ENV_FILE),
        "platform": platform.platform(),
    }


def scaffold_env() -> None:
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not ENV_FILE.exists():
        ENV_FILE.write_text(ENV_TEMPLATE)
    os.chmod(ENV_FILE, 0o600)


def install() -> dict:
    missing = [b for b in BINARIES if not shutil.which(b)]
    if missing and shutil.which("brew"):
        subprocess.run(["brew", "install", *sorted({BREW_PACKAGES[b] for b in missing})], check=False)
    elif missing:
        print(f"install these with your package manager: {', '.join(missing)}")
    if shutil.which("uv"):
        r = subprocess.run(["uv", "sync", "--project", str(HERE)], check=False)
        if r.returncode == 0:
            READY_MARKER.parent.mkdir(parents=True, exist_ok=True)
            READY_MARKER.touch()
    scaffold_env()
    return status()


def prefetch() -> None:
    """Download model weights once so the first real run is fast."""
    py = ["uv", "run", "--project", str(HERE), "python", "-c"]
    subprocess.run([*py, "from beat_this.inference import File2Beats; File2Beats(checkpoint_path='final0', device='cpu')"], check=False)
    subprocess.run([*py, "import mlx_whisper, numpy as np; mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), "
                         "path_or_hf_repo='mlx-community/whisper-large-v3-turbo')"], check=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--prefetch", action="store_true")
    a = p.parse_args()
    if a.check or a.json:
        s = status()
        if a.json:
            print(json.dumps(s))
        elif s["status_code"]:
            print(MESSAGES[s["status_code"]].format(missing=", ".join(s["missing_binaries"]), here=HERE, env=ENV_FILE))
        sys.exit(s["status_code"])
    s = install()
    if a.prefetch:
        prefetch()
    print(json.dumps(s, indent=1))
    sys.exit(s["status_code"])


if __name__ == "__main__":
    main()
