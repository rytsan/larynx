"""Corruptions that turn a rich clip into the network's input (§9.3).

Audio-level masks run before decomposition (build_shards.py stores the corrupted analysis);
frame-level masks run on the fly in training, so every epoch sees a new draw.
Truncation at H40 with taper is not here: it is part of core/features.h (netInput), always on.
"""
import numpy as np
from scipy import signal

FLOOR_DB = -100.0

# name: probability per clip (§9.3 table)
AUDIO_MASKS = {"softclip": 0.2, "lowpass": 0.2}


def soft_clip(x, rng):
    drive = 10 ** (rng.uniform(6, 18) / 20)
    peak = np.max(np.abs(x)) + 1e-9
    return (np.tanh(drive * x / peak) * peak).astype(np.float32)


def lowpass(x, fs, rng):
    sos = signal.butter(8, rng.uniform(6000, 8000), fs=fs, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def draw_audio_mask(rng):
    """At most one audio mask per clip: 'softclip', 'lowpass' or ''."""
    u, acc = rng.uniform(), 0.0
    for name, p in AUDIO_MASKS.items():
        acc += p
        if u < acc:
            return name
    return ""


def apply_audio_mask(name, x, fs, rng):
    return soft_clip(x, rng) if name == "softclip" else lowpass(x, fs, rng) if name == "lowpass" else x


def frame_masks(h, n_valid, rng):
    """Per-clip draw over a (n, 80) dB array, in place: truncation at H8-H32 (40%), dropout of 3-5
    harmonics below H40 (30%)."""
    if rng.uniform() < 0.4:
        h[:, rng.integers(8, 33):] = FLOOR_DB
    if rng.uniform() < 0.3:
        top = max(6, min(40, int(np.median(n_valid))))
        h[:, rng.choice(top, size=rng.integers(3, 6), replace=False)] = FLOOR_DB
    return h


def conditioning_noise(f0, flags, voiced_bit, rng):
    """Detector-error augmentation (§9.4): ±15 cents on f0, 5% of voiced frames flipped to unvoiced.
    Returns new (f0, flags)."""
    f0 = f0 * 2 ** (rng.uniform(-15, 15, size=f0.shape) / 1200)
    drop = (flags & voiced_bit).astype(bool) & (rng.uniform(size=flags.shape) < 0.05)
    return f0.astype(np.float32), np.where(drop, flags & ~voiced_bit, flags).astype(np.uint8)
