"""Stage 8: cues.json and curves.json for code, report.md for Claude."""
from __future__ import annotations

import json

import numpy as np

from core.audio import norm01, resample_curve
from core.context import Context
from core.timefmt import fmt_time
from stages.hear import instrument_conflicts

VERSION = 1
DEPS = ("ingest", "stems", "melody", "rhythm", "hits", "classify", "tags", "structure", "moments", "mix", "speech", "hear", "see")
STEMS = ("vocals", "drums", "bass", "other")
SPARK = "▁▂▃▄▅▆▇█"
MAX_MOMENTS = 12
MAX_WORDS = 250
MAX_DESC = 140


def _f(t: float, fps: int) -> int:
    return int(round(float(t) * fps))


def _framed(e: dict, fps: int, key: str = "t") -> dict:
    out = {key: e[key], "f": _f(e[key], fps)}
    out.update({k: v for k, v in e.items() if k != key})
    return out


def build_cues(d: dict, fps: int) -> dict:
    ing = d.get("ingest") or {}
    rh = d.get("rhythm") or {}
    st = d.get("structure") or {}
    hear = d.get("hear") or {}
    heard = {s["id"]: s["label"] for s in hear.get("sections", []) if s.get("label")}
    sections = []
    for s in st.get("sections", []):
        label = heard.get(s["id"])
        sections.append({
            "id": s["id"], "label": label or s["label"], "start": s["start"], "end": s["end"],
            "f0": _f(s["start"], fps), "f1": _f(s["end"], fps), "energy": s["energy"],
            "label_source": "gemini" if label else "auto",
        })
    mix = d.get("mix")
    return {
        "version": 1,
        "source": {"file": ing.get("file"), "hash": d.get("_key"), "duration": ing.get("duration"), "fps": fps,
                   "kind": (d.get("classify") or {}).get("kind")},
        "tempo": {"bpm": rh.get("bpm"), "confidence": rh.get("confidence"), "meter": rh.get("meter"),
                  "changes": rh.get("changes", [])} if rh else None,
        "key": st.get("key"),
        "beats": [_framed(b, fps) for b in rh.get("beats", [])],
        "hits": {k: [_framed(h, fps) for h in v] for k, v in ((d.get("hits") or {}).get("hits") or {}).items()},
        "sections": sections,
        "moments": [_framed(m, fps) for m in (d.get("moments") or {}).get("moments", [])],
        "speech": [
            {"t0": s["t0"], "t1": s["t1"], "text": s["text"], "words": s.get("words", [])}
            for s in (d.get("speech") or {}).get("segments", [])
        ],
        "mix": {k: mix.get(k) for k in ("lufs", "true_peak", "lra", "clipped_samples", "flags")} if mix else None,
        "notes": {
            stem: [{"t": n["t"], "f": _f(n["t"], fps), "end": n["end"], "f1": _f(n["end"], fps),
                    "pitch": n["pitch"], "name": n["name"], "velocity": n["velocity"]} for n in ns]
            for stem, ns in ((d.get("melody") or {}).get("notes") or {}).items() if ns
        },
        "tags": [
            {"label": s["label"], "group": s["group"], "t": s["t"], "f": _f(s["t"], fps), "end": s["end"],
             "f1": _f(s["end"], fps), "confidence": s["confidence"]}
            for s in (d.get("tags") or {}).get("spans", [])
        ],
    }


def build_curves(ctx: Context, fps: int, duration: float) -> dict:
    out: dict[str, list[float]] = {}
    st = ctx.data("structure")
    if st:
        c = np.load(ctx.path(st["curves_file"]))
        hz = st["curve_hz"]
        out["energy"] = resample_curve(c["energy"], hz, fps, duration)
        for band in ("low", "mid", "high"):
            out[f"bands.{band}"] = resample_curve(c[band], hz, fps, duration)
        for s in STEMS:
            if f"stem_{s}" in c.files:
                out[f"stems.{s}"] = resample_curve(c[f"stem_{s}"], hz, fps, duration)
    mix = ctx.data("mix")
    if mix and mix.get("short_term", {}).get("values"):
        stv = mix["short_term"]
        vals = np.array(stv["values"], dtype=float)
        loud = norm01(vals, floor=max(-60.0, float(vals.min())))
        out["loudness"] = resample_curve(loud, stv["hz"], fps, duration, offset=stv["offset"])
    tg = ctx.data("tags")
    if tg and ctx.path(tg["curves_file"]).exists():
        tc = np.load(ctx.path(tg["curves_file"]))
        for label in tc.files:
            out[f"tags.{label}"] = resample_curve(tc[label], tg["curve_hz"], fps, duration)
    n = int(round(duration * fps))
    return {"fps": fps, "frames": n,
            "curves": {k: [round(float(x), 3) for x in np.clip(v, 0, 1)] for k, v in out.items()}}


def _section_at(sections: list[dict], t: float) -> dict | None:
    return next((s for s in sections if s["start"] <= t < s["end"]), None)


def find_conflicts(cues: dict, hear: dict, kind: str | None, images: list[str], tag_labels: list[str] = ()) -> list[str]:
    out = []
    if not hear:
        return out
    measured = [m["t"] for m in cues["moments"] if m["type"] != "cut"]
    for em in hear.get("extra_moments", []):
        t = em["approx_t"]
        if not any(abs(t - x) <= 2.0 for x in measured):
            sec = _section_at(cues["sections"], t)
            img = f"spec_{sec['id']}.png" if sec and f"spec_{sec['id']}.png" in images else "spec_full.png"
            out.append(f"heard \"{em['what']}\" near {fmt_time(t, precise=True)}, nothing measured within 2 s; check {img}")
    hk = hear.get("kind")
    if hk and kind and hk != kind:
        out.append(f"measured kind is {kind}, hearing model says {hk}")
    missing = [s["id"] for s in cues["sections"] if s["label_source"] == "auto"]
    if missing:
        out.append(f"hearing model gave no label for {', '.join(missing)}; auto labels used")
    if hear.get("overall") and tag_labels:
        for inst in instrument_conflicts(hear["overall"].get("instruments", []), list(tag_labels)):
            out.append(f"heard {inst}, no matching measured tag")
    return out


def _spark(energy: list[float], fps: int, a: float, b: float, bins: int = 4) -> str:
    seg = energy[_f(a, fps):max(_f(a, fps) + 1, _f(b, fps))]
    if not seg:
        return ""
    chunks = np.array_split(np.array(seg), bins)
    return "".join(SPARK[min(7, int(c.mean() * 8))] if c.size else SPARK[0] for c in chunks)


def _clip(text: str, n: int = MAX_DESC) -> str:
    text = " ".join(str(text).split()).replace("|", "/")
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def render_report(cues, d, curves, conflicts, unavailable, image_paths) -> str:
    src, tempo, key = cues["source"], cues.get("tempo"), cues.get("key")
    hear = d.get("hear") or {}
    head = [src.get("file") or "input", fmt_time(src.get("duration") or 0), src.get("kind") or "unknown"]
    if tempo and tempo.get("bpm") and (tempo.get("confidence") or 0) >= 0.5:
        head.append(f"{tempo['bpm']:.1f} BPM {tempo.get('meter')}")
    elif tempo:
        head.append("no steady beat")
    if key:
        head.append(f"{key['tonic']} {key['mode']} ({key['confidence']:.2f})")
    lines = ["# " + "  ·  ".join(head), ""]
    for w in (d.get("ingest") or {}).get("warnings", []) + hear.get("warnings", []):
        lines.append(f"Warning: {w}")
    if (d.get("ingest") or {}).get("silent"):
        lines += ["Input is silent. Nothing to analyse.", ""]

    lines.append("## What it sounds like (heard)")
    if hear.get("overall"):
        o = hear["overall"]
        lines.append(_clip(f"{o['genre']}, {o['mood']}. Instruments: {', '.join(o['instruments'])}. "
                           f"Vocals: {o['vocals']}. Production: {o['production_style']}.", 400))
        if hear.get("notice"):
            lines.append(f"Note: {hear['notice']}")
    else:
        lines.append("Hearing model not run (--fast, --no-cloud, or unavailable). Everything below is measured.")
    lines.append("")

    if cues["sections"]:
        energy = curves["curves"].get("energy", [])
        heard_desc = {s["id"]: s.get("description", "") for s in hear.get("sections", [])}
        lines += ["## Timeline (measured + heard)", "| # | label | time | energy | what is happening |", "|---|---|---|---|---|"]
        for s in cues["sections"]:
            inside = sorted({m["type"] for m in cues["moments"] if s["start"] <= m["t"] < s["end"] and m["type"] not in ("cut", "hold")})
            desc = heard_desc.get(s["id"]) or (f"measured: {', '.join(inside)}" if inside else "measured: steady")
            lines.append(f"| {s['id']} | {s['label']} | {fmt_time(s['start'])} to {fmt_time(s['end'])} | "
                         f"{_spark(energy, src['fps'], s['start'], s['end'])} | {_clip(desc)} |")
        lines.append("")

    tg = cues.get("tags") or []
    if tg:
        lines.append("## What's playing when (measured)")
        firsts = {}
        for s in tg:
            if s["group"] in ("instrument", "voice"):
                firsts.setdefault(s["label"], []).append(s)
        ranked = sorted(firsts.items(), key=lambda kv: -sum(x["end"] - x["t"] for x in kv[1]))[:8]
        for label, ss in ranked:
            spans = ", ".join(f"{fmt_time(x['t'])} to {fmt_time(x['end'])}" for x in ss[:3])
            more = f" (+{len(ss) - 3})" if len(ss) > 3 else ""
            lines.append(f"- {label} {spans}{more}")
        sfx = sorted([s for s in tg if s["group"] == "sfx"], key=lambda s: -s["confidence"])[:8]
        if sfx:
            lines.append("SFX: " + "; ".join(f"{s['label']} {fmt_time(s['t'], precise=True)}" for s in sorted(sfx, key=lambda s: s["t"])))
        lines.append("")
    mel = (d.get("melody") or {}).get("summary") or {}
    if any(v.get("count") for v in mel.values()):
        lines.append("## Melody (measured)")
        for stem, v in mel.items():
            if not v.get("count"):
                continue
            hook = v.get("hook")
            hk = f"; hook {'-'.join(hook['pattern'])} x{len(hook['times'])} from {fmt_time(hook['times'][0], precise=True)}" if hook else ""
            lines.append(f"- {stem}: {v['count']} notes, {v['range'][0]} to {v['range'][1]}{hk}")
        lines.append("")

    events = [m for m in cues["moments"] if m["type"] not in ("cut", "hold")]
    top = sorted(sorted(events, key=lambda m: -m.get("strength", 0))[:MAX_MOMENTS], key=lambda m: m["t"])
    cuts = [m for m in cues["moments"] if m["type"] == "cut"]
    if top or cuts:
        lines.append("## Key moments (measured)")
        for m in top:
            span = f" to {fmt_time(m['end'], precise=True)}" if "end" in m else ""
            note = f": {m['note']}" if m.get("note") else ""
            lines.append(f"- {fmt_time(m['t'], precise=True)}{span} {m['type']} ({m.get('strength', 0):.2f}){note}")
        if len(events) > len(top):
            lines.append(f"- plus {len(events) - len(top)} more in cues.json")
        if cuts:
            lines.append("Best cuts: " + ", ".join(fmt_time(c["t"], precise=True) for c in sorted(cuts, key=lambda c: -c["strength"])[:6]))
        lines.append("")

    mix = d.get("mix")
    if mix:
        lines.append("## Mix (measured)")
        lines.append(f"{mix['lufs']:.1f} LUFS integrated  ·  true peak {mix['true_peak']:.1f} dBTP  ·  LRA {mix['lra']:.1f} LU  ·  "
                     f"width {mix.get('width', 0):.2f}  ·  clipped {mix['clipped_samples']}")
        lines.append("Flags: " + ("; ".join(mix["flags"]) if mix["flags"] else "none"))
        if hear.get("mix_impressions"):
            lines.append("Heard: " + _clip("; ".join(hear["mix_impressions"]), 300))
        ref = mix.get("ref")
        if ref:
            lines.append(f"Reference {ref['file']}: " + (ref.get("error") or ("; ".join(ref["deltas"]) or "no differences over 3 dB")))
        lines.append("")

    segs = (d.get("speech") or {}).get("segments", [])
    if segs:
        words = " ".join(s["text"] for s in segs).split()
        lines.append("## Speech (measured)")
        sp = hear.get("speech") or {}
        if any(sp.get(k) for k in ("tone", "emotion", "room", "noise")):
            lines.append(_clip(f"Heard: tone {sp['tone']}, emotion {sp['emotion']}, room {sp['room']}, noise {sp['noise']}", 300))
        pauses = (d.get("speech") or {}).get("pauses", [])
        lines.append(f"{len(segs)} segments, {len(words)} words, {len(pauses)} pauses over 0.4 s")
        lines.append(" ".join(words[:MAX_WORDS]) + (" ... (full transcript in cues.json)" if len(words) > MAX_WORDS else ""))
        lines.append("")

    lines.append("## Conflicts")
    lines += [f"- {c}" for c in conflicts] or ["none"]
    lines.append("")
    if unavailable:
        lines.append("## Status")
        lines += [f"- {k}: unavailable: {v}" for k, v in unavailable.items()]
        lines.append("")
    lines.append("## Images")
    lines += [f"- {p}" for p in image_paths] or ["none"]
    return "\n".join(lines) + "\n"


def run(ctx: Context) -> dict:
    d = {s: ctx.data(s) for s in DEPS}
    d["_key"] = ctx.workdir.name
    duration = float((d.get("ingest") or {}).get("duration") or 0.0)
    cues = build_cues(d, ctx.fps)
    curves = build_curves(ctx, ctx.fps, duration)
    images = (d.get("see") or {}).get("images", [])
    conflicts = find_conflicts(cues, d.get("hear") or {}, (d.get("classify") or {}).get("kind"), images,
                               [s["label"] for s in (d.get("tags") or {}).get("spans", [])])
    unavailable = {s: ctx.read(s)["_unavailable"] for s in DEPS if ctx.read(s).get("_unavailable")}
    report = render_report(cues, d, curves, conflicts, unavailable, [str(ctx.path(i)) for i in images])
    ctx.path("cues.json").write_text(json.dumps(cues, indent=1))
    ctx.path("curves.json").write_text(json.dumps(curves, separators=(",", ":")))
    ctx.path("report.md").write_text(report)
    return {"files": ["report.md", "cues.json", "curves.json"], "conflicts": len(conflicts)}
