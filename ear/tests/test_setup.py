import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup_mod(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("listen_setup", HERE / "setup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(mod, "READY_MARKER", tmp_path / ".listen-ready")
    for k in ("GEMINI_API_KEY", "LISTEN_BACKEND"):
        monkeypatch.delenv(k, raising=False)
    return mod


def _which(present):
    return lambda name: f"/bin/{name}" if name in present else None


def test_ready(setup_mod, monkeypatch):
    monkeypatch.setattr(setup_mod.shutil, "which", _which(setup_mod.BINARIES))
    setup_mod.ENV_FILE.write_text("GEMINI_API_KEY=abc\n")
    setup_mod.READY_MARKER.touch()
    assert setup_mod.status()["status_code"] == 0


@pytest.mark.parametrize(
    "binaries, key, ready, code",
    [
        (("ffmpeg", "ffprobe", "uv"), True, True, 2),
        (None, False, True, 3),
        (("ffmpeg",), False, True, 4),
        (None, True, False, 5),
    ],
)
def test_exit_codes(setup_mod, monkeypatch, binaries, key, ready, code):
    monkeypatch.setattr(setup_mod.shutil, "which", _which(binaries or setup_mod.BINARIES))
    if key:
        setup_mod.ENV_FILE.write_text("GEMINI_API_KEY=abc\n")
    if ready:
        setup_mod.READY_MARKER.touch()
    assert setup_mod.status()["status_code"] == code


def test_gemini_cli_backend_needs_no_key(setup_mod, monkeypatch):
    monkeypatch.setattr(setup_mod.shutil, "which", _which(setup_mod.BINARIES))
    setup_mod.ENV_FILE.write_text("LISTEN_BACKEND=gemini-cli\n")
    setup_mod.READY_MARKER.touch()
    assert setup_mod.status()["status_code"] == 0


def test_scaffold_env_is_private(setup_mod):
    setup_mod.scaffold_env()
    assert setup_mod.ENV_FILE.exists()
    assert oct(setup_mod.ENV_FILE.stat().st_mode & 0o777) == "0o600"
    assert "GEMINI_API_KEY" in setup_mod.ENV_FILE.read_text()
