#!/usr/bin/env python3
"""§9.6 gate: exported JSON run by core/neuralcore (C++, via pycore) == PyTorch, max |diff| <
max(1e-4, 2e-6·max|ref|). The reference runs in float64. float32 rounding grows with the output
magnitude: with the 60k-step weights (outputs up to ~110 dB on random input) PyTorch's own float32
result is 2.1e-4 off the exact one and the C++ path 1.3e-4.

    .venv/bin/python ml/train/check_export.py model.json [ckpt.pt]

Without a checkpoint, builds a random TCN, exports it and compares (tests the C++ inference itself).
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "build/ml/pycore"), str(REPO / "ml/train")]
import pycore  # noqa: E402
from model import TCN, export  # noqa: E402

TOL, REL_TOL = 1e-4, 2e-6


def main(json_path, ckpt=None):
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        m = TCN(**ck["arch"])
        m.load_state_dict(ck["model"])
    else:
        torch.manual_seed(0)
        m = TCN()
        json_path = json_path or tempfile.mktemp(suffix=".json")
        export(m, 0, json_path)
    m.eval()
    x = np.random.default_rng(0).uniform(0, 1, size=(600, 51)).astype(np.float32)  # > RF: ring wrap covered
    with torch.no_grad():
        ref = m.double()(torch.from_numpy(x).double()[None])[0].numpy()
    out = pycore.tcn_run(json.load(open(json_path)), x)
    d, tol = float(np.abs(out - ref).max()), max(TOL, REL_TOL * float(np.abs(ref).max()))
    print(f"max |C++ - torch(float64)| = {d:.2e} (tol {tol:.1e}) over {x.shape[0]} frames -> {'PASS' if d < tol else 'FAIL'}")
    return d < tol


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(0 if main(args[0] if args else None, args[1] if len(args) > 1 else None) else 1)
