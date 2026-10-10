"""Test helpers that stand in for the ingest stage so stage tests need no ffmpeg."""
from __future__ import annotations

import librosa
import numpy as np
import soundfile as sf


def ingest_array(ctx, y, sr: int = 44100, stereo=None) -> None:
    y = np.asarray(y, dtype=np.float32)
    st = np.stack([y, y], axis=1) if stereo is None else np.asarray(stereo, dtype=np.float32)
    sf.write(str(ctx.path("master.wav")), st, sr, subtype="FLOAT")
    speech = librosa.resample(st.mean(axis=1), orig_sr=sr, target_sr=16000)
    sf.write(str(ctx.path("speech.wav")), speech, 16000, subtype="PCM_16")
    duration = len(st) / sr
    ctx.write(
        "ingest",
        {
            "_v": 1,
            "source": "synthetic.wav",
            "file": "synthetic.wav",
            "duration": round(duration, 3),
            "sr": sr,
            "short": duration < 3.0,
            "silent": bool(np.max(np.abs(st)) < 1e-4),
            "warnings": [],
            "has_video": False,
            "video_fps": None,
        },
    )
