"""Per-run state shared by every stage."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _np_default(o):
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


@dataclass
class Context:
    workdir: Path
    fps: int = 60
    fast: bool = False
    no_cloud: bool = False
    options: dict = field(default_factory=dict)

    def path(self, name: str) -> Path:
        return self.workdir / name

    def read(self, stage: str) -> dict:
        """Raw stage JSON, or {} when missing or unreadable."""
        try:
            return json.loads((self.workdir / f"{stage}.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def data(self, stage: str) -> dict | None:
        """Stage JSON if the stage produced usable output, else None."""
        d = self.read(stage)
        if not d or d.get("_unavailable"):
            return None
        return d

    def write(self, stage: str, data: dict) -> None:
        target = self.workdir / f"{stage}.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1, default=_np_default))
        os.replace(tmp, target)
