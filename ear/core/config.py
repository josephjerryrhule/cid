"""Loads ~/.config/listen/.env into the environment without overriding real env vars."""
from __future__ import annotations

import os
from pathlib import Path


def config_dir() -> Path:
    return Path(os.environ.get("LISTEN_CONFIG", Path.home() / ".config" / "listen"))


def load_env(path: Path | None = None) -> None:
    p = path or config_dir() / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
