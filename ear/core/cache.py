"""Content-hash cache keys."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def cache_root() -> Path:
    return Path(os.environ.get("LISTEN_CACHE", Path.home() / ".cache" / "listen"))


def source_key(path: Path, start: float | None, end: float | None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    key = h.hexdigest()[:16]
    if start is not None or end is not None:
        lo = f"{start:g}" if start is not None else "0"
        hi = f"{end:g}" if end is not None else "end"
        key += f"_{lo}-{hi}"
    return key
