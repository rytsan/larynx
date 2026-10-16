#!/usr/bin/env python3
"""Build (or verify) the frozen golden set from GTSinger. Stdlib only.

8 voices x 3 clips: sung phrase (Control_Group), the voice's technique group,
and paired speech. GTSinger is CC BY-NC-SA 4.0 and is never used for training.

First run selects clips, downloads them and writes golden/manifest.csv and
golden/8_vozes/SHA256SUMS. Later runs never re-select: they re-download only
missing files from the manifest and fail if any hash differs.

    python3 ml/data/fetch_golden.py [--root golden]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
import wave

REPO = "AaronZ345/GTSinger"
REV = "d78cd68d00135b1995da03f25fccb3a221efd8c1"  # pinned dataset revision

# voice id, singer dir, technique dir, substring that picks the technique group
VOICES = [
    ("grave_baixo_it", "Italian/IT-Bass-1", "Vibrato", "Vibrato"),
    ("grave_baixo_es", "Spanish/ES-Bass-1", "Breathy", "Breathy"),
    ("aguda_soprano_fr", "French/FR-Soprano-1", "Vibrato", "Vibrato"),
    ("soprosa_soprano_ko", "Korean/KO-Soprano-1", "Breathy", "Breathy"),
    ("alto_ru", "Russian/RU-Alto-1", "Glissando", "Glissando"),
    ("soprosa_alto_en", "English/EN-Alto-1", "Breathy", "Breathy"),
    ("falsete_tenor_en", "English/EN-Tenor-1", "Mixed_Voice_and_Falsetto", "Falsetto"),
    ("tenor_faringeo_de", "German/DE-Tenor-1", "Pharyngeal", "Pharyngeal"),
]


def api_tree(path):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/" + urllib.parse.quote(path)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def subdirs(path):
    return sorted(e["path"].rsplit("/", 1)[-1] for e in api_tree(path) if e["type"] == "directory")


def select(singer, technique, group_key):
    """First song (sorted) with control, technique and speech groups, each with 0000.wav."""
    base = f"{singer}/{technique}"
    for song in subdirs(base):
        groups = subdirs(f"{base}/{song}")
        tech = [g for g in groups if group_key in g and g not in ("Control_Group", "Paired_Speech_Group")]
        if "Control_Group" in groups and "Paired_Speech_Group" in groups and tech:
            return [
                ("cantado", f"{base}/{song}/Control_Group/0000.wav"),
                ("tecnica", f"{base}/{song}/{tech[0]}/0000.wav"),
                ("fala", f"{base}/{song}/Paired_Speech_Group/0000.wav"),
            ]
    sys.exit(f"no complete song for {base}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(src, dst):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/" + urllib.parse.quote(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, dst)


def check_format(path):
    with wave.open(path) as w:
        return w.getnchannels(), w.getframerate(), w.getsampwidth() * 8, w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="golden")
    args = ap.parse_args()

    clips_dir = os.path.join(args.root, "8_vozes")
    manifest = os.path.join(args.root, "manifest.csv")
    sums = os.path.join(clips_dir, "SHA256SUMS")

    frozen = os.path.exists(manifest)
    if frozen:
        with open(manifest, newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []
        for voice, singer, technique, key in VOICES:
            for kind, src in select(singer, technique, key):
                rows.append({"file": f"{voice}/{kind}.wav", "voice": voice, "singer": singer.split("/")[1],
                             "kind": kind, "source": src})

    for row in rows:
        dst = os.path.join(clips_dir, row["file"])
        if not os.path.exists(dst):
            print("get", row["file"])
            download(row["source"], dst)

    if frozen:
        expected = {}
        with open(sums) as f:
            for line in f:
                digest, name = line.split(maxsplit=1)
                expected[name.strip()] = digest
        bad = [r["file"] for r in rows if sha256(os.path.join(clips_dir, r["file"])) != expected.get(r["file"])]
        if bad:
            sys.exit("SHA256 mismatch (golden set is frozen, do not overwrite): " + ", ".join(bad))
        print(f"ok: {len(rows)} clips match SHA256SUMS")
        return

    warnings = []
    for row in rows:
        ch, sr, bits, dur = check_format(os.path.join(clips_dir, row["file"]))
        row.update(channels=ch, sample_rate=sr, bits=bits, seconds=f"{dur:.2f}")
        if ch != 1 or sr != 48000:
            warnings.append(f"{row['file']}: {ch} ch, {sr} Hz")

    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(sums, "w") as f:
        for row in rows:
            f.write(f"{sha256(os.path.join(clips_dir, row['file']))}  {row['file']}\n")

    print(f"frozen: {len(rows)} clips -> {manifest}, {sums}")
    for msg in warnings:
        print("WARNING (expected mono 48 kHz):", msg)


if __name__ == "__main__":
    main()
