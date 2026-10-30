#!/usr/bin/env python3
"""Decompose training corpora into shards through ml/pycore (§9.2, §9.3). Replaces decompose.py +
qc_spectral.py: one pass reads the clip, runs QC and the C++ analysis, writes one .npz.

    .venv/bin/python ml/data/build_shards.py vocalset vctk libritts_r ptdb jvs_music [--data data] [--jobs 8] [--max-hours 10]

Per clip (resampled to 48 kHz, loudness-normalized to -20 LUFS):
  QC   reject on clipping, HNR < 12 dB, inharmonicity, jitter/shimmer too high (pathological) or
       too low ("dead" synthetic voice); RT60 only with --max-rt60 (blind estimate, see rt60()).
       The scripted corpora are documented dry studio recordings.
       Corpora missing from data/ are skipped with a warning.
  out  data/shards/<corpus>/<speaker>/<clip>.npz with the target analysis (clean) and, when an audio
       mask was drawn, the input analysis of the corrupted audio (masks.py).
Manifest data/shards/manifest.csv: one row per clip, kept or rejected, with split (by speaker,
80/10/10), bw_hz, QC numbers, npz sha256 and the core hash. Shards whose core hash differs from
the current one are stale and get rebuilt.
"""
import argparse
import csv
import hashlib
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import parselmouth
import pyloudnorm
import soundfile as sf
from scipy import signal

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "build/ml/pycore"))
sys.path.insert(0, str(REPO / "ml/data"))
import masks  # noqa: E402
import pycore  # noqa: E402

FS = 48000
CORE_FILES = ["analyzer.h", "analyzer.cpp", "skeleton.h", "skeleton.cpp", "features.h", "fft.h", "harm_frame.h"]

# corpus: (glob under data/<corpus>, speaker id from the path relative to data/<corpus>)
CORPORA = {
    # FULL/<singer>/<context>/<technique>/<file>.wav, 44.1 kHz
    "vocalset": ("**/FULL/*/*/*/*.wav", lambda p: p.parts[-4]),
    # wav48_silence_trimmed/<speaker>/<speaker>_<n>_mic1.flac, 48 kHz (mic2 is a second take of the same audio)
    "vctk": ("**/wav48_silence_trimmed/*/*_mic1.flac", lambda p: p.parts[-2]),
    # LibriTTS_R/train-clean-100/<speaker>/<chapter>/<id>.wav, 24 kHz (bw_hz masks above 12 kHz)
    "libritts_r": ("**/train-clean-100/*/*/*.wav", lambda p: p.parts[-3]),
    # SPEECH DATA/<FEMALE|MALE>/MIC/<speaker>/mic_<speaker>_<sentence>.wav, 48 kHz (LAR = laryngograph, unused)
    "ptdb": ("**/MIC/*/mic_*.wav", lambda p: p.parts[-2]),
    # jvs_music_ver1/<jvsNNN>/song_<common|unique>/wav/raw.wav, 24 kHz (hand-extracted)
    "jvs_music": ("**/jvs[0-9][0-9][0-9]/song_*/**/*.wav", lambda p: next(x for x in p.parts if x.startswith("jvs") and x[3:].isdigit())),
}

FIELDS = ["path", "corpus", "speaker", "split", "kept", "reason", "sr", "seconds", "bw_hz", "hnr_db",
          "inharm", "rt60", "jitter", "shimmer", "audio_mask", "frames", "sha256", "core_hash"]


def core_hash():
    h = hashlib.sha256()
    for f in CORE_FILES:
        h.update((REPO / "plugin/source/core" / f).read_bytes())
    return h.hexdigest()[:16]


def split_of(corpus, speaker):
    b = zlib.crc32(f"{corpus}/{speaker}".encode()) % 10
    return "train" if b < 8 else "val" if b == 8 else "test"


def bandwidth(x, fs, sr):
    """min(sr/2, rolloff): highest frequency where the long-term spectrum is within 90 dB of its peak.
    Catches band-limited sources (24 kHz corpora upsampled: nothing above 12 kHz), not the natural
    decay of a low voice. ponytail: a lowpass under 16-bit dither is not detected."""
    f, p = signal.welch(x, fs=fs, nperseg=4096)
    db = 10 * np.log10(p + 1e-20)
    above = np.flatnonzero(db > db.max() - 90)
    return float(min(sr / 2, f[above[-1]] if len(above) else sr / 2))


def hnr_db(fr):
    """Median over voiced frames of harmonic vs. noise-band power, from the C++ analysis itself."""
    v = (fr["flags"] & pycore.VOICED).astype(bool) & (fr["f0"] > 0)
    if v.sum() < 20:
        return -100.0
    hp = np.sum(10 ** (fr["h"][v] / 10) / 2, axis=1)
    npow = np.sum(10 ** (fr["noise"][v] / 10), axis=1) + 1e-20
    return float(np.median(10 * np.log10(hp / npow + 1e-20)))


def inharmonicity(fr):
    """Median relative deviation of the measured partials from a harmonic series fitted per frame
    (least-squares f0' through the origin on harmonics within 30 dB of the loudest, H1-H20). Fitting f0'
    removes detector error; F0 glides inside the window still smear partials (speech up to ~0.085)."""
    v = (fr["flags"] & pycore.VOICED).astype(bool) & (fr["f0"] > 0)
    if v.sum() < 20:
        return 0.0
    h, hz = fr["h"][v, :20], fr["hz"][v, :20]
    n = np.arange(1, 21, dtype=np.float64)
    w = (h > h.max(axis=1, keepdims=True) - 30) & (hz > 0)
    f0 = (w * hz * n).sum(1) / np.maximum((w * n * n).sum(1), 1e-9)
    dev = np.abs(hz - n * f0[:, None]) / np.maximum(f0[:, None], 1e-9)
    return float(np.median(dev[w])) if w.any() else 0.0


def rt60(x, fs):
    """Blind RT60 from the fastest energy fall: 10 ms envelope (30 ms smoothing), drop over the next
    100 ms at every frame; reverb bounds how fast sound can die, so RT60 ≈ 60 dB · 0.1 s / the 99th
    percentile drop. ponytail: crude — legato singing without pauses reads as reverb (VocalSet scales
    0.5-0.9 s), so it is informational unless --max-rt60 is set. Synthetic rooms: 0.3 → 0.35, 0.6 → 0.5,
    1.0 → 0.6, 2.0 → 0.7."""
    hop, w = fs // 100, 10
    p = np.add.reduceat(x.astype(np.float64) ** 2, np.arange(0, len(x) - hop, hop)) / hop
    e = 10 * np.log10(np.convolve(p, np.ones(3) / 3, mode="same") + 1e-12)
    if len(e) <= w:
        return 0.0
    drop = e[:-w] - np.lib.stride_tricks.sliding_window_view(e[1:], w).min(axis=1)
    return float(60 * 0.1 / max(np.percentile(drop, 99), 1e-3))


def praat_qc(x, fs):
    snd = parselmouth.Sound(x.astype(np.float64), fs)
    pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 60, 1200)
    jit = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shim = parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return float(np.nan_to_num(jit, nan=1.0)), float(np.nan_to_num(shim, nan=1.0))


MAX_RT60 = 0.0  # 0 = off; set by --max-rt60


def reject_reason(peak, hnr, inharm, reverb, jit, shim):
    if peak >= 0.999:
        return "clipping"
    if hnr < 12:
        return "hnr"
    if inharm > 0.1:  # speech glides reach ~0.085 on the golden set
        return "inharmonic"
    if MAX_RT60 and reverb > MAX_RT60:
        return "rt60"
    if jit > 0.03 or shim > 0.2:  # pathological voice or tracking failure
        return "jitter_shimmer"
    if jit < 0.001 and shim < 0.01:  # too stable to be a human voice (synthetic, ACE-Step)
        return "too_clean"
    return ""


def frames_dict(fr):
    return {k: fr[k] for k in ("f0", "flags", "n_valid")} | {k: fr[k].astype(np.float16) for k in ("h", "noise")}


def process(job):
    corpus, speaker, src, dst, seed = job
    row = {"path": str(dst), "corpus": corpus, "speaker": speaker, "split": split_of(corpus, speaker),
           "kept": 0, "reason": "", "audio_mask": "", "frames": 0, "sha256": "", "core_hash": CORE_HASH}
    try:
        x, sr = sf.read(src, dtype="float32", always_2d=True)
    except Exception as e:  # unreadable file: record it, keep going
        return row | {"reason": f"read: {e}"}
    x = x[:, 0]
    row |= {"sr": sr, "seconds": round(len(x) / sr, 3)}
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if sr != FS:
        g = np.gcd(sr, FS)
        x = signal.resample_poly(x, FS // g, sr // g).astype(np.float32)
    if len(x) < FS // 2:
        return row | {"reason": "short"}
    lufs = pyloudnorm.Meter(FS).integrated_loudness(x)
    if not np.isfinite(lufs):
        return row | {"reason": "silent"}
    x = (x * 10 ** ((-20 - lufs) / 20)).astype(np.float32)
    row["bw_hz"] = round(bandwidth(x, FS, sr))

    target = pycore.analyze(x, FS)
    hnr, inharm, reverb = hnr_db(target), inharmonicity(target), rt60(x, FS)
    jit, shim = praat_qc(x, FS)
    row |= {"hnr_db": round(hnr, 2), "inharm": round(inharm, 4), "rt60": round(reverb, 3),
            "jitter": round(jit, 4), "shimmer": round(shim, 4)}
    if reason := reject_reason(peak, hnr, inharm, reverb, jit, shim):
        return row | {"reason": reason}

    rng = np.random.default_rng(seed)
    out = {f"t_{k}": v for k, v in frames_dict(target).items()}
    if mask := masks.draw_audio_mask(rng):
        out |= {f"i_{k}": v for k, v in frames_dict(pycore.analyze(masks.apply_audio_mask(mask, x, FS, rng), FS)).items()}
    out["bw_hz"] = np.float32(row["bw_hz"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    return row | {"kept": 1, "audio_mask": mask, "frames": len(target["f0"]),
                  "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}


CORE_HASH = core_hash()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+", choices=CORPORA)
    ap.add_argument("--data", type=Path, default=REPO / "data")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--max-rt60", type=float, default=0, help="reject clips with blind RT60 above this (s); 0 = off")
    ap.add_argument("--max-hours", type=float, default=0, help="per corpus, random subsample by clip (0 = all)")
    a = ap.parse_args()
    global MAX_RT60
    MAX_RT60 = a.max_rt60

    out_dir = a.data / "shards"
    man_path = out_dir / "manifest.csv"
    old = {}
    if man_path.exists():
        with open(man_path) as f:
            old = {r["path"]: r for r in csv.DictReader(f) if r["core_hash"] == CORE_HASH}

    jobs = []
    for corpus in a.corpora:
        pattern, speaker_of = CORPORA[corpus]
        root = a.data / corpus
        files = sorted(f for f in root.glob(pattern) if not f.name.startswith("._") and "__MACOSX" not in f.parts)
        if not files:
            print(f"warning: {corpus}: no files match {root}/{pattern} — skipped (ml/data/download_corpora.sh)")
            continue
        rng = np.random.default_rng(zlib.crc32(corpus.encode()))
        rng.shuffle(files)
        budget = a.max_hours * 3600 if a.max_hours else float("inf")
        for src in files:
            if budget <= 0:
                break
            budget -= sf.info(src).duration if a.max_hours else 0
            rel = src.relative_to(root)
            dst = out_dir / corpus / speaker_of(rel) / (rel.stem + ".npz")
            jobs.append((corpus, speaker_of(rel), src, dst, zlib.crc32(str(rel).encode())))

    rows = [old[str(j[3])] for j in jobs if str(j[3]) in old]
    todo = [j for j in jobs if str(j[3]) not in old]
    print(f"{len(jobs)} clips, {len(todo)} to build (core {CORE_HASH})", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, row in enumerate(ex.map(process, todo, chunksize=8)):
            rows.append(row)
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    # Rows of corpora not rebuilt this run stay in the manifest.
    rows += [r for p, r in old.items() if r["corpus"] not in a.corpora]
    rows.sort(key=lambda r: r["path"])
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by_speaker = {}
    for r in rows:
        by_speaker.setdefault((r["corpus"], r["speaker"]), set()).add(r["split"])
    assert all(len(s) == 1 for s in by_speaker.values()), "speaker in more than one split"
    for corpus in sorted({r["corpus"] for r in rows}):
        rs = [r for r in rows if r["corpus"] == corpus]
        kept = [r for r in rs if int(r["kept"])]
        reasons = {}
        for r in rs:
            if not int(r["kept"]):
                reasons[r["reason"].split(":")[0]] = reasons.get(r["reason"].split(":")[0], 0) + 1
        hours = sum(float(r["seconds"]) for r in kept) / 3600
        print(f"{corpus}: kept {len(kept)}/{len(rs)} ({100 * (1 - len(kept) / max(1, len(rs))):.1f}% discarded, "
              f"{hours:.2f} h) rejects {reasons}")


if __name__ == "__main__":
    main()
