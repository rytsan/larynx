#pragma once
// M3: causal TCN inference, one frame per hop (§10). Same topology as ml/train/model.py:
//   in 1x1 (51→C) → [causal dilated conv C→2C (kernel k), GLU, x += g, skip += g] × L → out 1x1 (C→88).
// Weights given in PyTorch layout (conv w[out][in][k], k = kernel-1 is the current frame), loaded by the caller
// (the plugin parses JSON with JUCE; pycore takes numpy). Per-layer circular history buffers.
// No JUCE (core/ rule). Allocates only in load(); step() is real-time safe.
#include <vector>
#include "features.h"
#include "harm_frame.h"

namespace larynx {

struct TcnWeights {
    int channels = 0, kernel = 3, delay = 0; // delay D: output t is frame t-D (Studio)
    std::vector<int> dilations;
    std::vector<float> inW, inB;             // C×51, C
    std::vector<std::vector<float>> convW;   // per layer: 2C×C×k
    std::vector<std::vector<float>> convB;   // per layer: 2C
    std::vector<float> outW, outB;           // 88×C, 88
};

class NeuralCore {
public:
    // Returns false (and stays unloaded) on inconsistent shapes.
    bool load(const TcnWeights& w);
    bool loaded() const { return C_ > 0; }
    int delay() const { return w_.delay; }
    void reset();

    // One frame: x = netInput (51) → y (88): delta H1..40, absolute H41..80 (relative to rms_db),
    // delta noise 1..8 — all dB (§2.4).
    void step(const float* x, float* y);

    // Network output → per-harmonic deltas over `in` for blend() (H1..H80 then 8 noise bands).
    // H41+ is reconstructed as rms_db + y and kept only where it exceeds the input (§9.3 note).
    static void toDeltas(const HarmFrame& in, const float* y, float* delta);

private:
    TcnWeights w_;
    int C_ = 0;
    std::vector<std::vector<float>> hist_; // per layer: C × span ring (span = (k-1)·d + 1)
    std::vector<int> pos_;
    std::vector<float> x_, skip_, acc_;
};

} // namespace larynx
