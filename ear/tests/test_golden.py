"""Golden tests on human-annotated music (see tests/golden/README.md).

Thresholds are regression floors set from measured accuracy on 2026-10-11 (spec section 12),
not aspirations. Tighten them when accuracy improves; never loosen them to make a run pass.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import listen

GOLDEN = Path(__file__).parent / "golden"
CASES = sorted(p for p in GOLDEN.glob("*") if (p / "truth.json").exists() and list(p.glob("audio.*")))

BPM_TOLERANCE = 1.0
BEAT_AT_DOWNBEAT_P90_S = 0.030  # measured worst 25.4 ms
SECTION_TOLERANCE_S = 2.0
SECTION_F_FLOOR = 0.25  # measured worst 0.27, mean 0.38


def _section_f(est, truth):
    if not len(est) or not len(truth):
        return 0.0
    recall = np.mean([np.min(np.abs(est - t)) <= SECTION_TOLERANCE_S for t in truth])
    precision = np.mean([np.min(np.abs(truth - e)) <= SECTION_TOLERANCE_S for e in est])
    return 2 * recall * precision / (recall + precision) if recall + precision else 0.0


@pytest.mark.models
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_golden(case, tmp_path, monkeypatch):
    monkeypatch.setenv("LISTEN_CACHE", str(tmp_path / "cache"))
    truth = json.loads((case / "truth.json").read_text())
    audio = next(case.glob("audio.*"))
    code, info = listen.run([str(audio), "--no-cloud"])
    assert code == 0
    cues = json.loads((info["workdir"] / "cues.json").read_text())
    assert abs(cues["tempo"]["bpm"] - truth["bpm"]) <= BPM_TOLERANCE
    beats = np.array([b["t"] for b in cues["beats"]])
    errors = np.array([np.min(np.abs(beats - t)) for t in truth["downbeats"]])
    assert np.percentile(errors, 90) <= BEAT_AT_DOWNBEAT_P90_S, f"p90 {np.percentile(errors, 90) * 1000:.1f} ms"
    drops = np.array([m["t"] for m in cues["moments"] if m["type"] == "drop"])
    for t in truth.get("drops", []):
        assert drops.size and np.min(np.abs(drops - t)) <= 2 / 60, f"drop {t} not found in {drops}"
    est = np.array([s["start"] for s in cues["sections"][1:]])
    f = _section_f(est, np.array(truth.get("section_starts", [])))
    assert f >= SECTION_F_FLOOR, f"section F {f:.2f}"
