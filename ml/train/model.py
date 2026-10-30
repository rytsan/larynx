"""Causal TCN of §2.3 and its export (§9.6).

One residual block of 7 layers, dilations 1..64, kernel 3, GLU, residual + skip, 96 channels:
in 1x1 (51→C) → [causal conv C→2C, GLU, x += g, skip += g] × 7 → out 1x1 (C→88). RF 255 frames.
Layers are plain Conv1d so neuralcore (P6) can map each one onto RTNeural::Conv1D by hand.

    .venv/bin/python ml/train/model.py ckpt.pt model_live.json     # export
"""
import json
import sys

import torch
from torch import nn

N_IN, N_OUT = 51, 88
CONTRACT = 1  # must match kNetContract in plugin/source/core/features.h
DILATIONS = (1, 2, 4, 8, 16, 32, 64)


class TCN(nn.Module):
    def __init__(self, channels=96, dilations=DILATIONS, kernel=3):
        super().__init__()
        self.dilations, self.kernel = tuple(dilations), kernel
        self.inp = nn.Conv1d(N_IN, channels, 1)
        self.convs = nn.ModuleList(nn.Conv1d(channels, 2 * channels, kernel, dilation=d) for d in self.dilations)
        self.out = nn.Conv1d(channels, N_OUT, 1)

    def receptive_field(self):
        return 1 + (self.kernel - 1) * sum(self.dilations)

    def forward(self, x):  # (B, T, 51) → (B, T, 88), causal
        x = self.inp(x.transpose(1, 2))
        skip = 0
        for conv in self.convs:
            g = nn.functional.glu(conv(nn.functional.pad(x, ((self.kernel - 1) * conv.dilation[0], 0))), dim=1)
            x = x + g
            skip = skip + g
        return self.out(skip).transpose(1, 2)


def export(model, delay, path):
    """Weights as nested lists in PyTorch layout: conv w[out][in][k], k oldest→newest (k = kernel-1 is
    the current frame). RTNeural's Conv1D kernel order must be checked against this in the P6
    equivalence test (R2)."""
    layers = [{"name": "in", "w": model.inp.weight[:, :, 0].tolist(), "b": model.inp.bias.tolist()}]
    layers += [{"name": f"conv{i}", "dilation": d, "w": c.weight.tolist(), "b": c.bias.tolist()}
               for i, (d, c) in enumerate(zip(model.dilations, model.convs))]
    layers.append({"name": "out", "w": model.out.weight[:, :, 0].tolist(), "b": model.out.bias.tolist()})
    with open(path, "w") as f:
        json.dump({"contract": CONTRACT, "inputs": N_IN, "outputs": N_OUT, "D": delay,
                   "n_channels": model.inp.out_channels, "kernel": model.kernel,
                   "dilations": list(model.dilations), "layers": layers}, f)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    m = TCN(**ck["arch"])
    m.load_state_dict(ck["model"])
    export(m, ck["delay"], sys.argv[2])
    print(f"{sys.argv[2]}: D={ck['delay']}, {sum(p.numel() for p in m.parameters())} params")
