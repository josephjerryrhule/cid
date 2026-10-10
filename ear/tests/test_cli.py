import json
import subprocess

import pytest

import listen
from tests.synth import drum_loop, write


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LISTEN_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LISTEN_CONFIG", str(tmp_path / "config"))


def test_video_without_audio_exits_2(tmp_path, capsys):
    src = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30",
                    "-t", "1", str(src)], check=True)
    code, _ = listen.run([str(src), "--no-cloud"])
    assert code == 2
    assert "no audio stream" in capsys.readouterr().err


def test_bad_time_exits_2(tmp_path, capsys):
    src = tmp_path / "a.wav"
    write(src, drum_loop(120, bars=2)[0])
    code, _ = listen.run([str(src), "--start", "1:2:3:4"])
    assert code == 2


@pytest.mark.models
def test_end_to_end_fast_then_fps_change(tmp_path):
    src = tmp_path / "loop.wav"
    y, _ = drum_loop(120, bars=8)
    write(src, y)
    code, info = listen.run([str(src), "--fast", "--no-cloud"])
    assert code == 0
    work = info["workdir"]
    for name in ("report.md", "cues.json", "curves.json", "spec_full.png"):
        assert (work / name).exists(), name
    assert info["status"]["stems"] == "skipped" and info["status"]["hear"] == "skipped"

    code, info = listen.run([str(src), "--fast", "--no-cloud", "--fps", "30", "--export", str(tmp_path / "public")])
    assert code == 0
    ran = {k for k, v in info["status"].items() if v == "ok"}
    assert ran == {"assemble"}
    cues = json.loads((tmp_path / "public" / "cues.json").read_text())
    assert cues["source"]["fps"] == 30
    assert all(b["f"] == round(b["t"] * 30) for b in cues["beats"])
    assert (tmp_path / "public" / "curves.json").exists()
