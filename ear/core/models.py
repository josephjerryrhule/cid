"""Downloads model assets once. PANNs expects them in ~/panns_data; urllib, never wget."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from core.pipeline import StageUnavailable

ASSETS = {
    "Cnn14_DecisionLevelMax.pth": "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1",
    "class_labels_indices.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
    "ontology.json": "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json",
}


def panns_dir() -> Path:
    return Path(os.environ.get("LISTEN_PANNS_DIR", Path.home() / "panns_data"))


def ensure_panns() -> dict[str, Path]:
    root = panns_dir()
    root.mkdir(parents=True, exist_ok=True)
    for name, url in ASSETS.items():
        dest = root / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(url, tmp)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise StageUnavailable(f"could not download {name} from {url}: {e}")
        os.replace(tmp, dest)
    return {
        "checkpoint": root / "Cnn14_DecisionLevelMax.pth",
        "labels": root / "class_labels_indices.csv",
        "ontology": root / "ontology.json",
    }
