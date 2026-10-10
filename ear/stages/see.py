"""Stage 7: images Claude reads to see the sound. Flat styling, no shadows."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from core.context import Context
from core.timefmt import fmt_time

VERSION = 1
DEPS = ("ingest", "rhythm", "structure", "stems", "hear", "mix")
SPEC_SR = 22050
MAX_SECTION_IMAGES = 8
MAX_IMAGES = 16
STEMS = ("vocals", "drums", "bass", "other")
_mmss = FuncFormatter(lambda x, _: fmt_time(x))


def _labels(ctx: Context, sections: list[dict]) -> dict[str, str]:
    heard = {s["id"]: s["label"] for s in (ctx.data("hear") or {}).get("sections", []) if s.get("label")}
    return {s["id"]: heard.get(s["id"], s["label"]) for s in sections}


def _spectrogram(y, t0, path, title, sections, labels, downbeats, figsize):
    import librosa
    import librosa.display

    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SPEC_SR, n_mels=128, fmax=SPEC_SR / 2), ref=np.max)
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    x = t0 + librosa.times_like(S, sr=SPEC_SR)
    librosa.display.specshow(S, sr=SPEC_SR, x_coords=x, x_axis="time", y_axis="mel", fmax=SPEC_SR / 2, ax=ax, cmap="magma")
    t1 = t0 + len(y) / SPEC_SR
    top = ax.get_ylim()[1]
    for s in sections:
        if t0 <= s["start"] < t1:
            ax.axvline(s["start"], color="white", lw=1)
            ax.text(s["start"], top, f" {s['id']} {labels.get(s['id'], '')}", color="white", fontsize=9, va="top")
    db = [d for d in downbeats if t0 <= d < t1]
    if db:
        ax.plot(db, [ax.get_ylim()[0]] * len(db), ls="", marker="|", ms=10, color="#7fd4ff")
    ax.xaxis.set_major_formatter(_mmss)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _lanes(ctx: Context, structure: dict, path) -> None:
    c = np.load(ctx.path(structure["curves_file"]))
    hz = structure["curve_hz"]
    names = [f"stem_{s}" for s in STEMS if f"stem_{s}" in c.files] or ["low", "mid", "high"]
    fig, axes = plt.subplots(len(names), 1, figsize=(16, 1.5 * len(names)), dpi=100, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        v = c[name]
        t = np.arange(len(v)) / hz
        ax.fill_between(t, v, color="#e8590c", lw=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel(name.replace("stem_", ""), fontsize=9)
        ax.xaxis.set_major_formatter(_mmss)
    axes[0].set_title("stem energy" if names[0].startswith("stem_") else "band energy (no stems)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _ref(mix: dict, path) -> None:
    bands = list(mix["bands"])
    own = [mix["bands"][b] - mix["lufs"] for b in bands]
    ref = [mix["ref"]["bands"][b] - mix["ref"]["lufs"] for b in bands]
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.plot(bands, own, marker="o", color="#e8590c", label="this mix")
    ax.plot(bands, ref, marker="o", color="#4c6ef5", label="reference")
    ax.set_ylabel("band level relative to LUFS (dB)")
    ax.set_title("spectral balance, loudness matched", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(ctx: Context) -> dict:
    import librosa

    y, _ = librosa.load(str(ctx.path("master.wav")), sr=SPEC_SR, mono=True)
    structure = ctx.data("structure")
    sections = structure["sections"] if structure else []
    labels = _labels(ctx, sections)
    downbeats = (ctx.data("rhythm") or {}).get("downbeats", [])
    images = ["spec_full.png"]
    _spectrogram(y, 0.0, ctx.path("spec_full.png"), "full track", sections, labels, downbeats, (16, 5))
    if structure:
        _lanes(ctx, structure, ctx.path("stems.png"))
        images.append("stems.png")
        chosen = sorted(sorted(sections, key=lambda s: s["end"] - s["start"], reverse=True)[:MAX_SECTION_IMAGES],
                        key=lambda s: s["start"])
        for s in chosen:
            a, b = int(s["start"] * SPEC_SR), int(s["end"] * SPEC_SR)
            name = f"spec_{s['id']}.png"
            title = f"{s['id']} {labels.get(s['id'], '')}: {fmt_time(s['start'])} to {fmt_time(s['end'])}"
            _spectrogram(y[a:b], s["start"], ctx.path(name), title, [s], labels, downbeats, (12, 4))
            images.append(name)
    mix = ctx.data("mix") or {}
    if mix.get("ref", {}).get("bands"):
        _ref(mix, ctx.path("ref.png"))
        images.append("ref.png")
    return {"images": images[:MAX_IMAGES]}
