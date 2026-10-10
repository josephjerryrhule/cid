import numpy as np
from PIL import Image

from core.audio import CURVE_HZ
from stages import see
from tests.helpers import ingest_array
from tests.synth import pink_noise


def _fake_structure(ctx, duration, n_sections):
    n = int(duration * CURVE_HZ)
    curves = {k: np.random.default_rng(0).random(n) for k in ("energy", "rms", "low", "mid", "high")}
    np.savez_compressed(ctx.path("curves_structure.npz"), **curves)
    edges = np.linspace(0, duration, n_sections + 1)
    sections = [
        {"id": f"s{i + 1}", "label": "body", "start": float(a), "end": float(b), "energy": 0.5}
        for i, (a, b) in enumerate(zip(edges, edges[1:]))
    ]
    ctx.write("structure", {"_v": 1, "curves_file": "curves_structure.npz", "curve_hz": CURVE_HZ, "sections": sections})


def test_images_written_and_capped(ctx):
    ingest_array(ctx, pink_noise(40))
    _fake_structure(ctx, 40.0, 11)
    ctx.write("rhythm", {"_v": 1, "downbeats": list(np.arange(0.5, 40, 2.0))})
    ctx.write("hear", {"_v": 1, "sections": [{"id": "s1", "label": "intro"}]})
    out = see.run(ctx)
    images = out["images"]
    assert images[0] == "spec_full.png"
    assert "stems.png" in images
    zooms = [i for i in images if i.startswith("spec_s")]
    assert len(zooms) == see.MAX_SECTION_IMAGES
    assert len(images) <= see.MAX_IMAGES
    w, h = Image.open(ctx.path("spec_full.png")).size
    assert (w, h) == (1600, 500)


def test_without_structure_only_full_spectrogram(ctx):
    ingest_array(ctx, pink_noise(2))
    out = see.run(ctx)
    assert out["images"] == ["spec_full.png"]


def test_reference_image_when_ref_measured(ctx):
    ingest_array(ctx, pink_noise(5))
    bands = {"125 Hz": -20.0, "1000 Hz": -22.0}
    ctx.write("mix", {"_v": 1, "lufs": -14.0, "bands": bands, "ref": {"lufs": -12.0, "bands": {"125 Hz": -18.0, "1000 Hz": -25.0}}})
    assert "ref.png" in see.run(ctx)["images"]
