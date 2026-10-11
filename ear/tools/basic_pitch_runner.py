"""Runs inside an isolated uvx environment. Usage: basic_pitch_runner.py IN.wav OUT.json"""
import json
import sys

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

_, _, events = predict(sys.argv[1], str(ICASSP_2022_MODEL_PATH) + ".onnx")
notes = [{"t": float(s), "end": float(e), "pitch": int(p), "velocity": float(a)} for s, e, p, a, *_ in events]
with open(sys.argv[2], "w") as f:
    json.dump(sorted(notes, key=lambda n: n["t"]), f)
