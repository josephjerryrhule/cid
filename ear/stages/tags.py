"""Stage: what is playing when. PANNs framewise AudioSet tagging (527 classes, 100 Hz)."""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import median_filter

from core.context import Context
from core.models import ensure_panns

VERSION = 2
DEPS = ("ingest", "stems")
SR = 32000
FRAME_HZ = 100
CURVES_FILE = "curves_tags.npz"
THRESH = 0.3
STEM_THRESH = 0.2  # isolated stems give cleaner, lower-probability instrument tags
STEM_SOURCES = ("drums", "bass", "other", "vocals")
MIX_GROUPS = {"sfx", "ambience", "voice"}  # instruments come from stems when stems exist
STEM_GROUPS = {"instrument", "voice"}
GENERIC = {"Music", "Musical instrument"}
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


def merge_sources(found: dict[tuple[str, str], list[tuple[float, float, float]]]) -> dict[str, list[dict]]:
    """Union each label's spans across sources (mix and stems); keep the best confidence and every source."""
    by_label: dict[str, list[tuple[float, float, float, str]]] = {}
    for (label, source), spans in found.items():
        by_label.setdefault(label, []).extend((a, b, c, source) for a, b, c in spans)
    merged: dict[str, list[dict]] = {}
    for label, items in by_label.items():
        out: list[dict] = []
        for a, b, c, src in sorted(items):
            if out and a <= out[-1]["end"] + 0.5:
                out[-1]["end"] = max(out[-1]["end"], b)
                out[-1]["confidence"] = max(out[-1]["confidence"], c)
                if src not in out[-1]["sources"]:
                    out[-1]["sources"] = sorted(out[-1]["sources"] + [src])
            else:
                out.append({"t": a, "end": b, "confidence": c, "sources": [src]})
        merged[label] = out
    return merged


def run(ctx: Context) -> dict:
    import librosa

    paths = ensure_panns()
    from panns_inference import SoundEventDetection, labels

    sed = SoundEventDetection(checkpoint_path=str(paths["checkpoint"]), device="cpu")
    groups = class_groups(json.loads(paths["ontology"].read_text()), list(labels))
    stems = ctx.data("stems")
    sources = [("mix", ctx.path("master.wav"), THRESH, MIX_GROUPS if stems else MIX_GROUPS | STEM_GROUPS)]
    if stems:
        sources += [(s, ctx.path(stems["stems"][s]), STEM_THRESH, STEM_GROUPS) for s in STEM_SOURCES]
    found: dict[tuple[str, str], list] = {}
    curves_raw: dict[str, np.ndarray] = {}
    for source, path, thresh, allowed in sources:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        fw = framewise(y, sed)
        for i, label in enumerate(labels):
            if label in GENERIC or groups[label] not in allowed:
                continue
            spans = to_spans(fw[:, i], FRAME_HZ, thresh=thresh)
            if spans:
                found[(label, source)] = spans
                c = np.clip(fw[:, i], 0, 1).astype(np.float32)
                prev = curves_raw.get(label)
                curves_raw[label] = c if prev is None else np.maximum(prev[: len(c)], c[: len(prev)])
    merged = merge_sources(found)
    ranked = sorted(merged, key=lambda lab: -sum(x["end"] - x["t"] for x in merged[lab]))[:MAX_LABELS]
    curves = {lab: curves_raw[lab] for lab in ranked if sum(x["end"] - x["t"] for x in merged[lab]) >= CURVE_MIN_S}
    np.savez_compressed(ctx.path(CURVES_FILE), **curves)
    spans_out = [
        {"label": lab, "group": groups[lab], "t": round(x["t"], 2), "end": round(x["end"], 2),
         "confidence": round(x["confidence"], 3), "sources": x["sources"]}
        for lab in ranked
        for x in merged[lab]
    ]
    return {
        "spans": sorted(spans_out, key=lambda s: (s["t"], s["label"])),
        "curve_hz": FRAME_HZ,
        "curves_file": CURVES_FILE,
        "labels": ranked,
    }
