import pytest

from core import models
from core.pipeline import StageUnavailable


def test_existing_assets_are_not_downloaded(tmp_path, monkeypatch):
    monkeypatch.setenv("LISTEN_PANNS_DIR", str(tmp_path))
    for name in models.ASSETS:
        (tmp_path / name).write_bytes(b"x")
    calls = []
    monkeypatch.setattr(models.urllib.request, "urlretrieve", lambda *a: calls.append(a))
    paths = models.ensure_panns()
    assert calls == []
    assert paths["checkpoint"] == tmp_path / "Cnn14_DecisionLevelMax.pth"


def test_download_failure_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("LISTEN_PANNS_DIR", str(tmp_path))

    def boom(url, dest):
        raise OSError("offline")

    monkeypatch.setattr(models.urllib.request, "urlretrieve", boom)
    with pytest.raises(StageUnavailable, match="could not download"):
        models.ensure_panns()
    assert not list(tmp_path.glob("*.tmp"))
