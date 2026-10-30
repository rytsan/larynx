"""Non-neural baseline (§9.5) and the shared LSD metric.

Baseline: per frame, least-squares line of h (dB) vs. log2(n) over the harmonics present in the
input (H1..H40); harmonics missing from the input get the line's value, present ones are kept.
The network must beat it by ≥ 3 dB of LSD (H20-H80) to be accepted.
"""
import numpy as np

FLOOR = -100.0
PRESENT = -90.0  # same threshold as the synth's existing/missing split (§2.3)


def baseline(h_in, n_valid):
    """h_in (T, 80) dB (input, H41+ already at floor) → predicted full envelope (T, 80)."""
    n = np.arange(1, 81)
    x = np.log2(n)
    h = h_in.astype(np.float64)
    w = ((h > PRESENT) & (n <= np.minimum(40, n_valid)[:, None])).astype(np.float64)
    sw, sx, sy = w.sum(1), w @ x, (w * h).sum(1)
    sxx, sxy = w @ (x * x), (w * h) @ x
    den = sw * sxx - sx * sx
    fit = (sw >= 2) & (den > 1e-9)
    a = np.where(fit, (sw * sxy - sx * sy) / np.where(fit, den, 1), 0)
    b = np.where(fit, (sy - a * sx) / np.maximum(sw, 1), 0)
    miss = (h <= PRESENT) & (n <= n_valid[:, None]) & fit[:, None]
    return np.where(miss, np.clip(a[:, None] * x + b[:, None], FLOOR, 0), h).astype(np.float32)


def valid_mask(h_t, f0, voiced, n_valid, bw_hz):
    """Harmonics that count in loss and LSD: voiced frame, n ≤ nValid, n·f0 ≤ bw_hz, target above floor."""
    n = np.arange(1, 81)
    return (voiced[:, None] & (n <= n_valid[:, None]) & (n * f0[:, None] <= bw_hz) & (h_t > FLOOR + 1))


def lsd_terms(h_pred, h_t, mask, lo=20, hi=80):
    """Sum of squared dB error and count over H{lo}..H{hi} — pool them, then sqrt(sum / count)."""
    m = mask[:, lo - 1:hi]
    d = (h_pred[:, lo - 1:hi] - h_t[:, lo - 1:hi])[m]
    return float(np.sum(d * d)), int(m.sum())
