import subprocess

import numpy as np
import pytest
import soundfile as sf

from core.context import Context
from core.pipeline import Fatal
from stages import ingest
from tests.synth import pink_noise, write


def _ctx(tmp_path, src, **opts):
    work = tmp_path / "work"
    work.mkdir()
    return Context(workdir=work, options={"source": str(src), **opts})


def test_weird_filename_is_ingested(tmp_path):
    src = tmp_path / "my track (final) é's & co.wav"
    write(src, pink_noise(4))
    ctx = _ctx(tmp_path, src)
    out = ingest.run(ctx)
    info = sf.info(str(ctx.path("master.wav")))
    assert info.channels == 2 and info.samplerate == 44100
    assert abs(out["duration"] - 4.0) < 0.02
    assert out["file"] == src.name
    assert sf.info(str(ctx.path("speech.wav"))).samplerate == 16000


def test_mono_48k_becomes_stereo_44k(tmp_path):
    src = tmp_path / "mono.wav"
    write(src, pink_noise(3, sr=48000), sr=48000)
    ctx = _ctx(tmp_path, src)
    ingest.run(ctx)
    info = sf.info(str(ctx.path("master.wav")))
    assert (info.channels, info.samplerate) == (2, 44100)


def test_trim_with_start_and_end(tmp_path):
    src = tmp_path / "long.wav"
    write(src, pink_noise(10))
    out = ingest.run(_ctx(tmp_path, src, start=2.0, end=5.0))
    assert abs(out["duration"] - 3.0) < 0.05


def test_video_with_audio_reports_fps(tmp_path):
    src = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2", "-shortest", str(src)],
        check=True,
    )
    out = ingest.run(_ctx(tmp_path, src))
    assert out["has_video"] is True
    assert out["video_fps"] == 30.0


def test_video_without_audio_is_fatal(tmp_path):
    src = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30",
         "-t", "1", str(src)],
        check=True,
    )
    with pytest.raises(Fatal, match="no audio stream"):
        ingest.run(_ctx(tmp_path, src))


def test_short_and_silent_flags(tmp_path):
    short = tmp_path / "short.wav"
    write(short, pink_noise(1))
    assert ingest.run(_ctx(tmp_path, short))["short"] is True
    silent = tmp_path / "silent.wav"
    write(silent, np.zeros(44100 * 5, dtype=np.float32))
    work2 = tmp_path / "w2"
    work2.mkdir()
    out = ingest.run(Context(workdir=work2, options={"source": str(silent)}))
    assert out["silent"] is True


def test_missing_file_is_fatal(tmp_path):
    with pytest.raises(Fatal, match="no such file"):
        ingest.resolve_source(str(tmp_path / "nope.wav"), tmp_path / "dl")
