#!/usr/bin/env python3
"""§9 gate: h[80] from ml/pycore must match the headless C++ (larynx_analyze --harm) within 0.01 dB.

    PYTHONPATH=build/ml/pycore .venv/bin/python ml/pycore/check_equivalence.py file.wav [...]

Compares 100 random voiced frames per file. Exit 1 on failure.
"""
import subprocess
import sys
import wave

import numpy as np
import pycore

TOL_DB = 0.01


def read_wav(path):
    with wave.open(path) as w:
        n, ch, sw, fs = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, "<i2") / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        x = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8) / 8388608.0
    elif sw == 4:
        x = np.frombuffer(raw, "<i4") / 2147483648.0
    else:
        raise ValueError(f"{path}: {sw * 8}-bit PCM not supported")
    return x.reshape(-1, ch)[:, 0].astype(np.float32), fs


def main(paths):
    rng = np.random.default_rng(0)
    worst = 0.0
    for path in paths:
        x, fs = read_wav(path)
        py = pycore.analyze(x, fs)
        csv = subprocess.run(["build/tools/bench_plugin/larynx_analyze", path, "--harm"],
                             capture_output=True, text=True, check=True).stdout
        cpp = np.loadtxt(csv.splitlines()[1:], delimiter=",", ndmin=2)
        assert len(cpp) == len(py["f0"]), f"{path}: frame count {len(cpp)} vs {len(py['f0'])}"
        voiced = np.flatnonzero(py["flags"] & pycore.VOICED)
        idx = rng.choice(voiced, size=min(100, len(voiced)), replace=False)
        d = np.abs(cpp[idx, 6:] - py["h"][idx]).max() if len(idx) else 0.0
        worst = max(worst, d)
        print(f"{path}: {len(idx)} frames, max |dh| {d:.4f} dB")
    print(f"worst {worst:.4f} dB -> {'PASS' if worst < TOL_DB else 'FAIL'}")
    return worst < TOL_DB


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(0 if main(sys.argv[1:]) else 1)
