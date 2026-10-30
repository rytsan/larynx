#include "neuralcore.h"
#include <algorithm>
#include <cmath>

namespace larynx {

bool NeuralCore::load(const TcnWeights& w) {
    C_ = 0;
    const int C = w.channels, L = (int) w.dilations.size(), k = w.kernel;
    if (C <= 0 || k <= 0 || L == 0 || (int) w.convW.size() != L || (int) w.convB.size() != L
        || w.inW.size() != (size_t) (C * kNetInputs) || w.inB.size() != (size_t) C
        || w.outW.size() != (size_t) (kNetOutputs * C) || w.outB.size() != (size_t) kNetOutputs)
        return false;
    for (int l = 0; l < L; ++l)
        if (w.convW[(size_t) l].size() != (size_t) (2 * C * C * k) || w.convB[(size_t) l].size() != (size_t) (2 * C)
            || w.dilations[(size_t) l] <= 0)
            return false;
    w_ = w;
    // Re-lay conv weights tap-major, w[j][o][c], so the inner loop over c is contiguous (SIMD).
    for (int l = 0; l < L; ++l) {
        auto& W = w_.convW[(size_t) l];
        const auto& src = w.convW[(size_t) l];
        for (int o = 0; o < 2 * C; ++o)
            for (int c = 0; c < C; ++c)
                for (int j = 0; j < k; ++j)
                    W[((size_t) j * (size_t) (2 * C) + (size_t) o) * (size_t) C + (size_t) c] = src[((size_t) o * (size_t) C + (size_t) c) * (size_t) k + (size_t) j];
    }
    hist_.assign((size_t) L, {});
    for (int l = 0; l < L; ++l) hist_[(size_t) l].assign((size_t) (C * ((k - 1) * w.dilations[(size_t) l] + 1)), 0.0f);
    pos_.assign((size_t) L, 0);
    x_.assign((size_t) C, 0.0f);
    skip_.assign((size_t) C, 0.0f);
    acc_.assign((size_t) (2 * C), 0.0f);
    C_ = C;
    return true;
}

void NeuralCore::reset() {
    for (auto& h : hist_) std::fill(h.begin(), h.end(), 0.0f);
    std::fill(pos_.begin(), pos_.end(), 0);
}

void NeuralCore::step(const float* in, float* y) {
    const int C = C_, k = w_.kernel;
    for (int c = 0; c < C; ++c) {
        float s = w_.inB[(size_t) c];
        const float* wr = &w_.inW[(size_t) (c * kNetInputs)];
        for (int i = 0; i < kNetInputs; ++i) s += wr[i] * in[i];
        x_[(size_t) c] = s;
    }
    std::fill(skip_.begin(), skip_.end(), 0.0f);
    for (size_t l = 0; l < hist_.size(); ++l) {
        const int d = w_.dilations[l], span = (k - 1) * d + 1;
        auto& h = hist_[l];
        int& p = pos_[l];
        // Ring of the last `span` layer inputs, frame-major: h[slot·C + c].
        std::copy(x_.begin(), x_.end(), h.begin() + (ptrdiff_t) p * C);
        const auto& W = w_.convW[l];
        std::copy(w_.convB[l].begin(), w_.convB[l].end(), acc_.begin());
        for (int j = 0; j < k; ++j) {
            // Tap j sees the frame (k-1-j)·d steps back (padding = zeros before the first frame).
            const int slot = ((p - (k - 1 - j) * d) % span + span) % span;
            const float* xs = &h[(size_t) slot * (size_t) C];
            for (int o = 0; o < 2 * C; ++o) {
                const float* wr = &W[((size_t) j * (size_t) (2 * C) + (size_t) o) * (size_t) C];
                float s = 0.0f;
                for (int c = 0; c < C; ++c) s += wr[c] * xs[c];
                acc_[(size_t) o] += s;
            }
        }
        p = (p + 1) % span;
        for (int c = 0; c < C; ++c) {
            const float g = acc_[(size_t) c] / (1.0f + std::exp(-acc_[(size_t) (C + c)])); // GLU: a·σ(b)
            x_[(size_t) c] += g;
            skip_[(size_t) c] += g;
        }
    }
    for (int o = 0; o < kNetOutputs; ++o) {
        float s = w_.outB[(size_t) o];
        const float* wr = &w_.outW[(size_t) (o * C)];
        for (int c = 0; c < C; ++c) s += wr[c] * skip_[(size_t) c];
        y[o] = s;
    }
}

void NeuralCore::toDeltas(const HarmFrame& in, const float* y, float* delta) {
    const float rms = netRmsDb(in);
    for (int i = 0; i < kNumHarmonics; ++i) {
        if (i < kNetHarmonics) delta[i] = y[i];
        else delta[i] = std::max(0.0f, rms + y[i] - std::max(in.h[i], kFloorDb));
    }
    for (int b = 0; b < kNumNoiseBands; ++b) delta[kNumHarmonics + b] = y[kNumHarmonics + b];
}

} // namespace larynx
