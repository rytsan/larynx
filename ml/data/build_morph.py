#!/usr/bin/env python3
"""Morph profiles (§12, M5 v2) from the 12 GTSinger singers outside the golden set. Never training.

    .venv/bin/python ml/data/build_morph.py [--per-singer 16] [--data data] [--out plugin/source/neural/profiles.data]

1. Downloads up to N clips per singer (one 0000.wav per song/group, all techniques) into
   data/gtsinger_new/<singer>/ — the same clips serve the P6 A/B on new voices.
2. Analyses them with pycore; per voiced frame: vowel (core vowel reader), pitch bin (2 semitones
   from 55 Hz), envelope e = h[H1..H40] − rms_db (core/features.h scale).
3. Profile = mean dB envelope per singer × vowel × bin (≥ 20 frames; vowel index 5 = any vowel).
4. PCA over all profiles (8 components); stores mean, components and each profile's coefficients.

File format (little-endian), read by core/morph.h:
  "LXMP" u32 version=1, u32 nHarm, nComp, nVowels, nBins, nSingers; f32 bin0Hz, binSt
  f32 mean[nHarm]; f32 comp[nComp][nHarm]; char name[nSingers][32]
  f32 coef[nSingers][nVowels][nBins][nComp]; u8 valid[nSingers][nVowels][nBins]
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "build/ml/pycore"), str(REPO / "ml/data")]
import fetch_golden as gt  # noqa: E402
import pycore  # noqa: E402

N_HARM, N_COMP, VOWELS = 40, 8, "aeiou"
BIN0_HZ, BIN_ST, N_BINS = 55.0, 2.0, 27  # 55 Hz .. ~1.2 kHz
MIN_FRAMES = 20
FLOOR_REL = -80.0  # envelope value for harmonics absent from a frame (relative to rms_db)


def new_singers():
    golden = {s.split("/")[1] for _, s, _, _ in gt.VOICES}
    out = []
    for lang in gt.subdirs(""):
        if lang == "processed":
            continue
        out += [f"{lang}/{s}" for s in gt.subdirs(lang) if s not in golden]
    return out


def fetch(singer, n, root):
    dst_dir = root / singer.split("/")[1]
    have = sorted(dst_dir.glob("*.wav"))
    if len(have) >= n:
        return have[:n]
    got = []
    for tech in gt.subdirs(singer):
        for song in gt.subdirs(f"{singer}/{tech}"):
            for group in gt.subdirs(f"{singer}/{tech}/{song}"):
                if len(got) >= n:
                    return got
                dst = dst_dir / f"{tech}_{song}_{group}.wav".replace(" ", "_")
                if not dst.exists():
                    try:
                        gt.download(f"{singer}/{tech}/{song}/{group}/0000.wav", str(dst))
                    except Exception as e:  # group without 0000.wav
                        print(f"  skip {tech}/{song}/{group}: {e}")
                        continue
                got.append(dst)
    return got


def frames_of(path):
    x, fs = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    if fs not in (44100, 48000):
        raise ValueError(f"{path}: {fs} Hz")
    f = pycore.analyze(x, fs)
    v, _, _ = pycore.vowel_of(f)
    X = pycore.net_input(f)
    rms = X[:, 50] * 100 - 100
    voiced = (f["flags"] & pycore.VOICED).astype(bool) & (f["f0"] > 0)
    h = f["h"][:, :N_HARM]
    e = np.where(h > -100, h - rms[:, None], FLOOR_REL)
    e = np.maximum(e, FLOOR_REL)
    b = np.round(12 * np.log2(np.maximum(f["f0"], 1) / BIN0_HZ) / BIN_ST).astype(int)
    keep = voiced & (b >= 0) & (b < N_BINS)
    vi = np.array([VOWELS.find(chr(c)) if c else -1 for c in v])
    return e[keep], vi[keep], b[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-singer", type=int, default=16)
    ap.add_argument("--data", type=Path, default=REPO / "data")
    ap.add_argument("--out", type=Path, default=REPO / "plugin/source/neural/profiles.data")
    a = ap.parse_args()

    singers = new_singers()
    print(f"{len(singers)} singers outside the golden set: {', '.join(s.split('/')[1] for s in singers)}")
    nv = len(VOWELS) + 1
    sums = np.zeros((len(singers), nv, N_BINS, N_HARM))
    cnt = np.zeros((len(singers), nv, N_BINS))
    for si, singer in enumerate(singers):
        files = fetch(singer, a.per_singer, a.data / "gtsinger_new")
        for path in files:
            e, vi, b = frames_of(path)
            for v in (vi, np.full_like(vi, nv - 1)):  # its vowel and "any"
                ok = v >= 0
                np.add.at(sums[si], (v[ok], b[ok]), e[ok])
                np.add.at(cnt[si], (v[ok], b[ok]), 1)
        print(f"  {singer}: {len(files)} clips, {int(cnt[si, nv - 1].sum())} voiced frames")

    valid = cnt >= MIN_FRAMES
    prof = sums[valid] / cnt[valid][:, None]
    mean = prof.mean(0)
    _, sv, vt = np.linalg.svd(prof - mean, full_matrices=False)
    comp = vt[:N_COMP]
    explained = (sv[:N_COMP] ** 2).sum() / (sv ** 2).sum()
    coef = np.zeros((len(singers), nv, N_BINS, N_COMP), np.float32)
    coef[valid] = (prof - mean) @ comp.T

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(b"LXMP" + struct.pack("<6I2f", 1, N_HARM, N_COMP, nv, N_BINS, len(singers), BIN0_HZ, BIN_ST))
        f.write(mean.astype("<f4").tobytes() + comp.astype("<f4").tobytes())
        for s in singers:
            f.write(s.split("/")[1].encode()[:31].ljust(32, b"\0"))
        f.write(coef.astype("<f4").tobytes() + valid.astype(np.uint8).tobytes())
    print(f"{valid.sum()} profiles, PCA {N_COMP} comps explain {100 * explained:.1f}% -> {a.out}")


if __name__ == "__main__":
    main()
