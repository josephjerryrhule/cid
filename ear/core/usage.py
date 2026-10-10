"""Local daily counter for hearing requests, and the one-time data-use notice."""
from __future__ import annotations

import json
import os
from datetime import date

from core.cache import cache_root
from core.pipeline import StageUnavailable

NOTICE = (
    "Audio was sent to Google AI Studio. On the free tier Google may use it to improve its "
    "products and humans may review it. Use a billing-enabled key or --no-cloud for unreleased client audio."
)


def _path():
    return cache_root() / "usage.json"


def _load() -> dict:
    try:
        return json.loads(_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, p)


def cap() -> int:
    return int(os.environ.get("LISTEN_DAILY_CAP", "200"))


def today_count() -> int:
    return int(_load().get("days", {}).get(date.today().isoformat(), 0))


def check_cap() -> str | None:
    n, c = today_count(), cap()
    if n >= c:
        raise StageUnavailable(f"daily hearing cap reached ({n}/{c}); raise LISTEN_DAILY_CAP or wait until tomorrow")
    if n >= 0.8 * c:
        return f"hearing requests today: {n}/{c}"
    return None


def record() -> None:
    data = _load()
    days = data.setdefault("days", {})
    key = date.today().isoformat()
    days[key] = int(days.get(key, 0)) + 1
    _save(data)


def first_upload_notice(backend: str) -> str | None:
    if backend != "aistudio":
        return None
    data = _load()
    if data.get("notice_shown"):
        return None
    data["notice_shown"] = True
    _save(data)
    return NOTICE
