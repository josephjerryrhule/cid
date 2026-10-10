"""Time parsing and formatting (SS, MM:SS, HH:MM:SS)."""
from __future__ import annotations


def parse_time(s: str) -> float:
    parts = str(s).strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"bad time: {s}")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def fmt_time(t: float, precise: bool = False) -> str:
    m, s = divmod(max(float(t), 0.0), 60)
    h, m = divmod(int(m), 60)
    sec = f"{s:05.2f}" if precise else f"{int(s):02d}"
    return f"{h}:{m:02d}:{sec}" if h else f"{m}:{sec}"
