#!/usr/bin/env python3
"""Train the TCN on shards (§9.4). Live: --delay 0; Studio: --delay 16 (predicts frame t-D seeing up to t).

    .venv/bin/python ml/train/train.py --name live0 [--delay 0] [--steps 20000] [--data data]

Inputs come from core/features.h through pycore.net_input; frame masks, gain (±12 dB) and
conditioning noise are drawn per example (masks.py). Loss (dB, relative to the input rms_db):
H1-40 and noise as deltas over the input, H41-80 absolute; band weights H1-8 ×2, H9-40 ×1,
H41-60 ×0.8, H61-80 ×0.5, noise ×0.5; harmonics masked by nValid, bw_hz, unvoiced frames and floor.
Validation: LSD H20-H80 per speaker against the baseline (ml/eval/baseline.py); the checkpoint with
the lowest worst-speaker LSD is kept in data/runs/<name>/best.pt.
ponytail: no F1/F2 auxiliary loss yet (needs differentiable formant picking); examples are built in
the main process — move to DataLoader workers if the GPU starves.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for p in ("build/ml/pycore", "ml/data", "ml/eval", "ml/train"):
    sys.path.insert(0, str(REPO / p))
import masks  # noqa: E402
import pycore  # noqa: E402
from baseline import FLOOR, baseline, lsd_terms, valid_mask  # noqa: E402
from model import TCN  # noqa: E402

NH = pycore.NET_HARMONICS  # 40
BAND_W = np.concatenate([np.full(8, 2.0), np.full(32, 1.0), np.full(20, 0.8), np.full(20, 0.5), np.full(8, 0.5)])


def load(data, splits):
    clips = {s: [] for s in splits}
    with open(data / "shards/manifest.csv") as f:
        rows = [r for r in csv.DictReader(f) if int(r["kept"]) and r["split"] in splits]
    for r in rows:
        if r["core_hash"] != rows[0]["core_hash"]:
            sys.exit("shards from different core versions: rebuild with build_shards.py")
        z = np.load(r["path"])
        t = {k[2:]: z[k] for k in z.files if k.startswith("t_")}
        i = {k[2:]: z[k] for k in z.files if k.startswith("i_")} or t
        clips[r["split"]].append({"t": t, "i": i, "bw": float(z["bw_hz"]), "speaker": f"{r['corpus']}/{r['speaker']}"})
    return clips


def example(clip, s, length, rng, train):
    """Network input x (T, 51), target y (T, 88), weight w (T, 88), plus what eval needs."""
    sl = slice(s, s + length)
    t = {k: v[sl] for k, v in clip["t"].items()}
    i = {k: v[sl].astype(np.float32) if v.dtype == np.float16 else v[sl] for k, v in clip["i"].items()}
    i["h"] = masks.frame_masks(i["h"], i["n_valid"], rng)
    cond = dict(i)
    if train:
        cond["f0"], cond["flags"] = masks.conditioning_noise(i["f0"], i["flags"], pycore.VOICED, rng)
    x = pycore.net_input(cond)
    rms = x[:, 50] * 100 + FLOOR
    if train:  # input gain ±12 dB: only the rms_db channel moves, everything else is relative
        x[:, 50] = (np.clip(rms + rng.uniform(-12, 12), FLOOR, 0) - FLOOR) / 100

    h_t, n_t = t["h"].astype(np.float32), t["noise"].astype(np.float32)
    y = np.concatenate([h_t[:, :NH] - i["h"][:, :NH], h_t[:, NH:] - rms[:, None], n_t - i["noise"]], axis=1)
    voiced = (t["flags"] & pycore.VOICED).astype(bool) & (t["f0"] > 0)
    hv = valid_mask(h_t, t["f0"], voiced, t["n_valid"], clip["bw"])
    w = np.concatenate([hv, np.ones((len(y), 8), bool)], axis=1) * BAND_W
    return x, y.astype(np.float32), w.astype(np.float32), {"h_in": i["h"], "h_t": h_t, "rms": rms, "mask": hv,
                                                            "n_valid": i["n_valid"]}


def batch(clips, n, length, rng):
    xs, ys, ws = [], [], []
    while len(xs) < n:
        c = clips[rng.integers(len(clips))]
        T = len(c["t"]["f0"])
        if T < length:
            continue
        x, y, w, _ = example(c, rng.integers(T - length + 1), length, rng, True)
        xs.append(x), ys.append(y), ws.append(w)
    return [torch.from_numpy(np.stack(a)) for a in (xs, ys, ws)]


def loss_fn(pred, y, w, delay):
    if delay:
        pred, y, w = pred[:, delay:], y[:, :-delay], w[:, :-delay]
    return (w * (pred - y) ** 2).sum() / w.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, clips, delay, dev):
    """LSD H20-H80 per speaker for model and baseline; masks drawn with a fixed seed per clip."""
    model.eval()
    acc = {}
    for ci, c in enumerate(clips):
        T = len(c["t"]["f0"])
        x, _, _, e = example(c, 0, T, np.random.default_rng(ci), False)
        out = model(torch.from_numpy(x)[None].to(dev)).float()[0].cpu().numpy()
        pred = np.concatenate([e["h_in"][:, :NH] + out[:, :NH], e["rms"][:, None] + out[:, NH:80]], axis=1)
        h_in = e["h_in"].copy()
        h_in[:, NH:] = FLOOR  # the baseline sees what the network sees
        base = baseline(h_in, e["n_valid"])
        if delay:  # output t is frame t-D
            pred, h_t, mask, base = pred[delay:], e["h_t"][:-delay], e["mask"][:-delay], base[:-delay]
        else:
            h_t, mask = e["h_t"], e["mask"]
        a = acc.setdefault(c["speaker"], np.zeros(4))
        a += [*lsd_terms(pred, h_t, mask), *lsd_terms(base, h_t, mask)]
    model.train()
    per = {s: (math.sqrt(a[0] / max(a[1], 1)), math.sqrt(a[2] / max(a[3], 1))) for s, a in acc.items()}
    tot = np.sum(list(acc.values()), axis=0)
    return per, math.sqrt(tot[0] / max(tot[1], 1)), math.sqrt(tot[2] / max(tot[3], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--length", type=int, default=256, help="frames per example")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--data", type=Path, default=REPO / "data")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clips = load(a.data, ("train", "val"))
    print({s: len(c) for s, c in clips.items()}, flush=True)
    model = TCN(channels=a.channels).to(dev)
    print(f"{sum(p.numel() for p in model.parameters())} params, RF {model.receptive_field()} frames", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    out_dir = a.data / "runs" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    best, t0, run = float("inf"), time.time(), 0.0
    for step in range(1, a.steps + 1):
        x, y, w = (v.to(dev, non_blocking=True) for v in batch(clips["train"], a.batch, a.length, rng))
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            pred = model(x)
        loss = loss_fn(pred.float(), y, w, a.delay)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % a.val_every == 0 or step == a.steps:
            per, lsd, lsd_base = evaluate(model, clips["val"], a.delay, dev)
            worst = max(v[0] for v in per.values())
            print(f"step {step} loss {run:.2f} | val LSD {lsd:.2f} dB (baseline {lsd_base:.2f}), worst speaker "
                  f"{worst:.2f} | {time.time() - t0:.0f} s", flush=True)
            if worst < best:
                best = worst
                torch.save({"model": model.state_dict(), "arch": {"channels": a.channels}, "delay": a.delay,
                            "step": step, "val_lsd": lsd, "val_worst": worst, "baseline_lsd": lsd_base},
                           out_dir / "best.pt")
    print(f"best worst-speaker LSD {best:.2f} dB -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
