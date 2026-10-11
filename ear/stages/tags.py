"""Stage: what is playing when. PANNs framewise AudioSet tagging (527 classes, 100 Hz)."""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import median_filter

from core.context import Context
from core.models import ensure_panns

VERSION = 1
DEPS = ("ingest",)
SR = 32000
FRAME_HZ = 100
CURVES_FILE = "curves_tags.npz"
THRESH = 0.3
MIN_SPAN_S = 0.3
MAX_LABELS = 40
CURVE_MIN_S = 2.0
GROUP_BY_ROOT = {
    "Musical instrument": "instrument",
    "Music genre": "genre",
    "Music mood": "genre",
    "Music": "music",
    "Human sounds": "voice",
    "Animal": "sfx",
    "Sounds of things": "sfx",
    "Natural sounds": "sfx",
    "Source-ambiguous sounds": "sfx",
    "Channel, environment and background": "ambience",
}


def class_groups(ontology: list[dict], labels: list[str]) -> dict[str, str]:
    by_id = {o["id"]: o for o in ontology}
    parent: dict[str, str] = {}
    for o in ontology:
        for c in o.get("child_ids", []):
            parent.setdefault(c, o["id"])
    name_to_id = {o["name"]: o["id"] for o in ontology}
    out = {}
    for label in labels:
        node, group = name_to_id.get(label), None
        while node is not None:
            name = by_id[node]["name"]
            if name in GROUP_BY_ROOT and (group is None or GROUP_BY_ROOT[name] != "music"):
                group = group or GROUP_BY_ROOT[name]
                if GROUP_BY_ROOT[name] != "music":
                    break
            node = parent.get(node)
        out[label] = group or "sfx"
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def to_spans(prob, hz, thresh=THRESH, min_s=MIN_SPAN_S, smooth_s=0.3, merge_s=0.5) -> list[tuple[float, float, float]]:
    p = median_filter(np.asarray(prob, dtype=float), size=max(1, int(smooth_s * hz)))
    merged: list[list[int]] = []
    for a, b in _runs(p > thresh):
        if merged and (a - merged[-1][1]) / hz < merge_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a / hz, b / hz, float(p[a:b].max())) for a, b in merged if (b - a) / hz >= min_s]


def framewise(y: np.ndarray, sed, chunk_s: float = 60.0) -> np.ndarray:
    step = int(chunk_s * SR)
    parts = []
    for start in range(0, max(len(y), 1), step):
        chunk = y[start : start + step]
        if len(chunk) < SR // 10:
            break
        out = sed.inference(chunk[None, :].astype(np.float32))[0]
        parts.append(out[: int(round(len(chunk) / SR * FRAME_HZ))])
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 527))


def run(ctx: Context) -> dict:
    import librosa

    paths = ensure_panns()
    from panns_inference import SoundEventDetection, labels

    sed = SoundEventDetection(checkpoint_path=str(paths["checkpoint"]), device="cpu")
    y, _ = librosa.load(str(ctx.path("master.wav")), sr=SR, mono=True)
    fw = framewise(y, sed)
    groups = class_groups(json.loads(paths["ontology"].read_text()), list(labels))
    found = []
    for i, label in enumerate(labels):
        spans = to_spans(fw[:, i], FRAME_HZ)
        if spans:
            found.append((sum(b - a for a, b, _ in spans), label, spans, i))
    found.sort(key=lambda x: -x[0])
    found = found[:MAX_LABELS]
    curves = {label: np.clip(fw[:, i], 0, 1).astype(np.float32) for total, label, _, i in found if total >= CURVE_MIN_S}
    np.savez_compressed(ctx.path(CURVES_FILE), **curves)
    spans_out = [
        {"label": label, "group": groups[label], "t": round(a, 2), "end": round(b, 2), "confidence": round(c, 3)}
        for _, label, spans, _ in found
        for a, b, c in spans
    ]
    return {
        "spans": sorted(spans_out, key=lambda s: (s["t"], s["label"])),
        "curve_hz": FRAME_HZ,
        "curves_file": CURVES_FILE,
        "labels": [label for _, label, _, _ in found],
    }
