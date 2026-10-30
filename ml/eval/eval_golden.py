#!/usr/bin/env python3
"""§9 gate on the golden set (never trained on): LSD H20-H80 of the network vs. the baseline, per voice.

    .venv/bin/python ml/eval/eval_golden.py data/runs/live/best.pt

Each clip is analysed by pycore and degraded by the training masks with a fixed seed per clip (same
draws for network and baseline). Gate: network beats the baseline by ≥ 3 dB.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / p) for p in ("build/ml/pycore", "ml/data", "ml/eval", "ml/train")]
import pycore  # noqa: E402
from build_shards import FS, bandwidth  # noqa: E402
from model import TCN  # noqa: E402
from train import evaluate  # noqa: E402


def main(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = TCN(**ck["arch"])
    model.load_state_dict(ck["model"])
    clips = []
    for path in sorted((REPO / "golden/8_vozes").glob("*/*.wav")):
        x, fs = sf.read(path, dtype="float32")
        assert fs == FS
        t = pycore.analyze(x, fs)
        t = {k: (v.astype(np.float16) if k in ("h", "noise") else v) for k, v in t.items() if k != "hop" and k != "hz"}
        clips.append({"t": t, "i": t, "bw": bandwidth(x, fs, fs), "speaker": path.parent.name})
    per, lsd, base = evaluate(model, clips, ck["delay"], "cpu")
    for s, (m, b) in sorted(per.items()):
        print(f"{s:22s} net {m:5.2f} dB  baseline {b:5.2f} dB  gain {b - m:+5.2f}")
    print(f"ALL: net {lsd:.2f} dB, baseline {base:.2f} dB, gain {base - lsd:+.2f} dB -> "
          f"{'PASS' if base - lsd >= 3 else 'FAIL'} (≥ 3 dB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
