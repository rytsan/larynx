#!/usr/bin/env python3
"""Vowel drift (NFR5): F1/F2 of a processed WAV vs. its time-aligned reference, via Praat (parselmouth).

Frames: every 10 ms where Praat finds pitch in the reference and both files have F1 and F2.
Drift per frame = |F_out - F_ref| / F_ref. Reports median and p95 per formant.

    python ml/eval/metrics.py ref.wav out.wav [ref2.wav out2.wav ...]

Pairs are pooled; the last line is the total over all of them.
"""
import sys

import numpy as np
import parselmouth

STEP = 0.01
MAX_FORMANT = 5500.0  # Praat default for female voices; same setting on both sides, so only the drift matters


def formants(snd, times):
    f = snd.to_formant_burg(time_step=STEP, max_number_of_formants=5, maximum_formant=MAX_FORMANT)
    return np.array([[f.get_value_at_time(k, t) for t in times] for k in (1, 2)])


def formant_drift(ref_path, out_path):
    """Relative F1/F2 drift per voiced frame, shape (2, n)."""
    ref, out = parselmouth.Sound(ref_path), parselmouth.Sound(out_path)
    pitch = ref.to_pitch(time_step=STEP)
    times = [t for t in pitch.xs() if pitch.get_value_at_time(t) > 0]
    a, b = formants(ref, times), formants(out, times)
    ok = np.all(np.isfinite(a) & np.isfinite(b), axis=0)
    return np.abs(b[:, ok] - a[:, ok]) / a[:, ok]


def report(name, d):
    if d.shape[1] == 0:
        print(f"{name}: no voiced frames")
        return
    med, p95 = np.median(d, axis=1) * 100, np.percentile(d, 95, axis=1) * 100
    print(f"{name}: frames {d.shape[1]:6d}  F1 med {med[0]:6.2f}% p95 {p95[0]:6.2f}%  "
          f"F2 med {med[1]:6.2f}% p95 {p95[1]:6.2f}%")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or len(args) % 2:
        sys.exit(__doc__)
    all_d = []
    for ref_path, out_path in zip(args[::2], args[1::2]):
        d = formant_drift(ref_path, out_path)
        report(out_path, d)
        all_d.append(d)
    if len(all_d) > 1:
        report("TOTAL", np.concatenate(all_d, axis=1))
