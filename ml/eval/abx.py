#!/usr/bin/env python3
"""Blind ABX on the golden set (§7 gate, §9.5). Replaces tools/abx_player: one stdlib-only CLI.

    python3 ml/eval/abx.py prepare out_dir [--clips 20] [--seconds 8] [-- larynx_render options]
    python3 ml/eval/abx.py run out_dir [--name you]

prepare: cuts the first N seconds of each golden clip (A) and renders them through the DSP path with
larynx_render (B, latency removed). Options after `--` go to larynx_render, e.g. `-- --bright 6`.
run: per pair, X is A or B at random; keys a/b/x replay, then answer a or b. Answers go to
out_dir/results_<name>.csv; the end prints the score and the one-sided binomial p-value
(P3 gate: ≤ 14/20 correct, p ≥ 0.05). Playback via paplay, aplay or ffplay (first found).
"""
import argparse
import csv
import math
import random
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RENDER = REPO / "build/tools/bench_plugin/larynx_render"


def cut(src, dst, seconds):
    with wave.open(str(src)) as r:
        p = r.getparams()
        frames = r.readframes(min(p.nframes, int(seconds * p.framerate)))
    with wave.open(str(dst), "wb") as w:
        w.setparams(p)
        w.writeframes(frames)


def prepare(a):
    clips = sorted((REPO / "golden/8_vozes").glob("*/*.wav"))
    # interleave clip types so 20 of 24 still covers every voice and type
    clips = sorted(clips, key=lambda p: (["cantado", "tecnica", "fala"].index(p.stem), p.parent.name))[:a.clips]
    if not clips:
        sys.exit("no golden clips: python3 ml/data/fetch_golden.py")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(clips):
        A, B = out / f"{i:02d}_A.wav", out / f"{i:02d}_B.wav"
        cut(c, A, a.seconds)
        subprocess.run([str(RENDER), str(A), str(B), *a.render], check=True, capture_output=True)
        print(f"{i:02d} {c.parent.name}/{c.stem}")
    (out / "render_args.txt").write_text(" ".join(a.render) + "\n")


def player():
    for cmd in (["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            return cmd
    sys.exit("no audio player (paplay, aplay, ffplay)")


def p_value(k, n):
    """P(≥ k correct of n | guessing)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def run(a):
    out = Path(a.out_dir)
    pairs = sorted(out.glob("*_A.wav"))
    if not pairs:
        sys.exit(f"no pairs in {out}: run prepare first")
    play = player()
    rng = random.SystemRandom()
    order = rng.sample(pairs, len(pairs))
    path = out / f"results_{a.name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    correct = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "pair", "x_is", "answer", "correct", "plays"])
        for t, A in enumerate(order, 1):
            B = A.with_name(A.name.replace("_A", "_B"))
            x_is = rng.choice("AB")
            files = {"a": A, "b": B, "x": A if x_is == "A" else B}
            plays = 0
            print(f"\ntrial {t}/{len(order)}  [a]=A [b]=B [x]=X, then answer with A or B (upper case)")
            while True:
                k = input("> ").strip()
                if k in files:
                    plays += 1
                    subprocess.run([*play, str(files[k])])
                elif k in ("A", "B"):
                    ok = k == x_is
                    correct += ok
                    w.writerow([t, A.name[:2], x_is, k, int(ok), plays])
                    f.flush()
                    break
    n = len(order)
    print(f"\n{correct}/{n} correct, p = {p_value(correct, n):.3f} (guessing ⇒ p ≥ 0.05) → {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("out_dir")
    p.add_argument("--clips", type=int, default=20)
    p.add_argument("--seconds", type=float, default=8)
    p.add_argument("render", nargs="*", help="after --: larynx_render options")
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("--name", default="me")
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else run(a)


if __name__ == "__main__":
    main()
