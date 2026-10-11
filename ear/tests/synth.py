"""Synthetic audio with known ground truth. Every generator is deterministic."""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100


def write(path, y, sr: int = SR) -> None:
    sf.write(str(path), np.asarray(y, dtype=np.float32), sr, subtype="FLOAT")


def bandpass(x, lo, hi, sr=SR):
    return sosfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), x)


def highpass(x, f, sr=SR):
    return sosfilt(butter(4, f, btype="high", fs=sr, output="sos"), x)


def lowpass(x, f, sr=SR):
    return sosfilt(butter(4, f, btype="low", fs=sr, output="sos"), x)


def _env(n, decay_s, sr=SR):
    return np.exp(-np.arange(n) / (decay_s * sr))


def kick(sr=SR):
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    freq = 50 + 70 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(freq) / sr) * _env(n, 0.08, sr)


def snare(sr=SR):
    rng = np.random.default_rng(1)
    n = int(0.18 * sr)
    noise = bandpass(rng.standard_normal(n), 300, 4000, sr)
    tone = np.sin(2 * np.pi * 180 * np.arange(n) / sr)
    return (0.7 * noise / np.max(np.abs(noise)) + 0.3 * tone) * _env(n, 0.05, sr)


def hat(sr=SR):
    rng = np.random.default_rng(2)
    n = int(0.05 * sr)
    x = highpass(rng.standard_normal(n), 7000, sr)
    return 0.4 * x / np.max(np.abs(x)) * _env(n, 0.012, sr)


def _add(y, x, t, sr, gain):
    i = int(round(t * sr))
    j = min(len(y), i + len(x))
    if i < len(y):
        y[i:j] += gain * x[: j - i]


def drum_loop(bpm=120.0, bars=16, sr=SR, jitter=0.0, seed=0, bar_chords=False):
    """4/4 loop starting at 0.5 s: kick on 1 and 3, snare on 2 and 4, hats on eighths.

    Kicks on 1 and 3 are nearly identical, so drums alone are genuinely ambiguous between
    2/4 and 4/4. bar_chords=True adds a mid-register chord that changes every bar, which is
    how real music marks the downbeat. Use it whenever a test checks meter or downbeats.

    Returns (signal, truth). truth holds beats, downbeats, kicks, snares, hats (seconds) and duration.
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n_beats = bars * 4
    ibis = beat * (1 + jitter * rng.uniform(-1, 1, n_beats))
    beats = np.concatenate([[0.5], 0.5 + np.cumsum(ibis[:-1])])
    y = np.zeros(int((beats[-1] + beat + 1.0) * sr))
    k, s, h = kick(sr), snare(sr), hat(sr)
    kicks, snares, hats = [], [], []
    for i, b in enumerate(beats):
        pos = i % 4
        if pos in (0, 2):
            _add(y, k, b, sr, 1.0 if pos == 0 else 0.85)
            kicks.append(b)
        else:
            _add(y, s, b, sr, 0.8)
            snares.append(b)
        _add(y, h, b, sr, 0.5)
        hats.append(b)
        off = b + ibis[i] / 2
        _add(y, h, off, sr, 0.35)
        hats.append(off)
    if bar_chords:
        prog = [[60, 64, 67], [57, 60, 64], [65, 69, 72], [67, 71, 74]]
        for bar in range(bars):
            t0 = beats[bar * 4]
            t1 = beats[bar * 4 + 4] if bar * 4 + 4 < len(beats) else t0 + 4 * beat
            _add(y, 0.35 * chords([prog[bar % 4]], seconds_each=t1 - t0, repeats=1, sr=sr), t0, sr, 1.0)
    y = 0.8 * y / np.max(np.abs(y))
    truth = {
        "beats": beats,
        "downbeats": beats[::4],
        "kicks": np.array(kicks),
        "snares": np.array(snares),
        "hats": np.array(hats),
        "duration": len(y) / sr,
    }
    return y.astype(np.float32), truth


def pink_noise(seconds, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return (0.5 * x / np.max(np.abs(x))).astype(np.float32)


def to_lufs(y, target, sr=SR):
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    return pyln.normalize.loudness(y, meter.integrated_loudness(y), target).astype(np.float32)


def midi_hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def chords(progression, seconds_each=2.0, repeats=2, sr=SR):
    out = []
    n = int(seconds_each * sr)
    t = np.arange(n) / sr
    fade = np.minimum(1, np.minimum(t / 0.02, (seconds_each - t) / 0.05))
    for notes in progression:
        x = sum(np.sin(2 * np.pi * midi_hz(m) * h * t) / h for m in notes for h in (1, 2, 3))
        out.append(x * fade)
    y = np.concatenate(out * repeats)
    return (0.5 * y / np.max(np.abs(y))).astype(np.float32)


def drop_track(sr=SR, gain=0.8):
    """Filtered quiet intro, then a full drop with sub bass at downbeat 10 (20.5 s)."""
    loop, truth = drum_loop(120, bars=20, sr=sr)
    drop_t = float(truth["downbeats"][10])
    i = int(round(drop_t * sr))
    y = loop.astype(float).copy()
    y[:i] = lowpass(y[:i], 400, sr) * 0.15
    y[i:] += 0.6 * np.sin(2 * np.pi * 50 * np.arange(len(y) - i) / sr)
    y = gain * y / np.max(np.abs(y))
    return y.astype(np.float32), {**truth, "drop": drop_t}


def riser_track(sr=SR):
    """Quiet bed 0 to 20 s, linear high riser 12 to 20 s, loud drums from 20 s."""
    bed = 0.03 * pink_noise(28, sr, seed=3)
    y = bed.astype(float)
    a, b = int(12 * sr), int(20 * sr)
    rng = np.random.default_rng(4)
    noise = highpass(rng.standard_normal(b - a), 4000, sr)
    noise /= np.max(np.abs(noise))
    y[a:b] += noise * np.linspace(0.02, 0.8, b - a)
    loop, _ = drum_loop(120, bars=4, sr=sr)
    j = int(20 * sr)
    seg = loop[: len(y) - j]
    y[j : j + len(seg)] += seg
    y = 0.8 * y / np.max(np.abs(y))
    return y.astype(np.float32), {"riser_start": 12.0, "riser_end": 20.0, "drop": 20.0}


def impacts_track(sr=SR, times=(2.0, 5.5, 9.0)):
    y = 0.03 * pink_noise(12, sr, seed=5).astype(float)
    rng = np.random.default_rng(6)
    for t in times:
        burst = rng.standard_normal(int(0.3 * sr)) * _env(int(0.3 * sr), 0.06, sr)
        _add(y, burst / np.max(np.abs(burst)), t, sr, 0.9)
        _add(y, kick(sr), t, sr, 0.9)
    y = 0.8 * y / np.max(np.abs(y))
    return y.astype(np.float32), {"impacts": list(times)}


def syllables(seconds=8.0, rate_hz=4.0, sr=SR, seed=7):
    """Speech-like signal: voiced harmonic tone with gliding pitch, amplitude-modulated at syllable rate."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    pitch = 140 + 30 * np.sin(2 * np.pi * 0.7 * t) + 10 * rng.standard_normal()
    phase = 2 * np.pi * np.cumsum(pitch) / sr
    voiced = sum(np.sin(h * phase) / h for h in range(1, 12))
    voiced = bandpass(voiced, 300, 3400, sr)
    am = np.clip(np.sin(2 * np.pi * rate_hz * t), 0, None) ** 2
    y = voiced * am
    return (0.6 * y / np.max(np.abs(y))).astype(np.float32)


def melody(pitches, note_s=0.4, gap_s=0.1, sr=SR, start=0.5):
    """Harmonic tones with short attack and release. Returns (signal, [(t, end, midi)])."""
    n_total = int((start + len(pitches) * (note_s + gap_s) + 0.5) * sr)
    y = np.zeros(n_total)
    truth = []
    t = start
    n = int(note_s * sr)
    tt = np.arange(n) / sr
    env = np.minimum(1, np.minimum(tt / 0.01, (note_s - tt) / 0.03))
    for p in pitches:
        f = midi_hz(p)
        tone = sum(np.sin(2 * np.pi * f * h * tt) / h for h in (1, 2, 3, 4))
        _add(y, tone * env, t, sr, 1.0)
        truth.append((t, t + note_s, p))
        t += note_s + gap_s
    return (0.6 * y / np.max(np.abs(y))).astype(np.float32), truth
