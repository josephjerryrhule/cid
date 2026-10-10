import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core.pipeline import StageUnavailable
from stages import stems
from tests.helpers import ingest_array
from tests.synth import drum_loop, pink_noise


class _Result:
    def __init__(self, code, stderr=""):
        self.returncode, self.stderr, self.stdout = code, stderr, ""


def test_demucs_failure_is_unavailable(ctx, monkeypatch):
    ingest_array(ctx, pink_noise(2))
    monkeypatch.setattr(stems.subprocess, "run", lambda *a, **k: _Result(1, "boom"))
    with pytest.raises(StageUnavailable, match="demucs failed"):
        stems.run(ctx)


def test_missing_uvx_is_unavailable(ctx, monkeypatch):
    ingest_array(ctx, pink_noise(2))

    def raise_fnf(*a, **k):
        raise FileNotFoundError("uvx")

    monkeypatch.setattr(stems.subprocess, "run", raise_fnf)
    with pytest.raises(StageUnavailable, match="uvx not found"):
        stems.run(ctx)


def test_success_moves_stems_into_workdir(ctx, monkeypatch):
    ingest_array(ctx, pink_noise(2))

    def fake_run(cmd, **k):
        out = Path(cmd[cmd.index("-o") + 1]) / "htdemucs" / "master"
        out.mkdir(parents=True)
        for s in stems.STEMS:
            sf.write(str(out / f"{s}.wav"), np.zeros((100, 2), np.float32), 44100)
        return _Result(0)

    monkeypatch.setattr(stems.subprocess, "run", fake_run)
    out = stems.run(ctx)
    assert out["stems"] == {s: f"stem_{s}.wav" for s in stems.STEMS}
    assert all(ctx.path(f"stem_{s}.wav").exists() for s in stems.STEMS)
    assert not ctx.path("stems_tmp").exists()


@pytest.mark.models
def test_real_demucs_puts_drums_in_drums(ctx):
    y, _ = drum_loop(120, bars=4)
    ingest_array(ctx, y)
    stems.run(ctx)
    energy = {s: float(np.sum(sf.read(str(ctx.path(f"stem_{s}.wav")))[0] ** 2)) for s in stems.STEMS}
    assert max(energy, key=energy.get) == "drums"
