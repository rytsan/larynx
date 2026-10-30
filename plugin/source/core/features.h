#pragma once
// Network input contract (§2.4): HarmFrame → 51 normalized channels. Shared by neuralcore (P6) and
// ml/ via pycore, so training and inference see the same numbers. No JUCE (core/ rule).
//   [0..39]  h[0..39] − rms_db, cosine taper on the last 4, (clamp(·, −100, 0) + 100) / 100
//   [40..47] noise − rms_db, same normalization
//   [48]     log_f0_norm: log2(f0/60) / log2(1200/60), 0 when unvoiced (monotonic, no wrap)
//   [49]     voiced 0/1
//   [50]     (clamp(rms_db, −100, 0) + 100) / 100
// rms_db: mean square of what the network sees (H1..H40 + the 8 noise bands), as the dBFS amplitude of
// the sine with that power (same scale as h) — so h − rms_db ≤ 0 and the clamp never cuts a harmonic.
#include <algorithm>
#include <cmath>
#include "harm_frame.h"

namespace larynx {

constexpr int kNetHarmonics = 40;
constexpr int kNetInputs = kNetHarmonics + kNumNoiseBands + 3;      // 51
constexpr int kNetOutputs = kNumHarmonics + kNumNoiseBands;          // 88
constexpr int kNetContract = 1;                                      // bump on any change here
constexpr float kNetF0Min = 60.0f, kNetF0Max = 1200.0f;

// Linear-amplitude cosine taper over H37..H40 (0.905, 0.655, 0.345, 0.095), as a dB offset.
inline float netTaperDb(int i) {
    constexpr int kLen = 4, kStart = kNetHarmonics - kLen;
    if (i < kStart) return 0.0f;
    const float w = 0.5f * (1.0f + std::cos(3.14159265f * (float) (i - kStart + 1) / (float) (kLen + 1)));
    return 20.0f * std::log10(w);
}

inline float netRmsDb(const HarmFrame& f) {
    double ms = 0.0; // 2 × mean square
    const int nh = std::min<int>(kNetHarmonics, f.nValid);
    for (int i = 0; i < nh; ++i) ms += std::pow(10.0, f.h[i] / 10.0);   // A² (×½ for both terms below)
    for (float v : f.noise) ms += 2.0 * std::pow(10.0, v / 10.0);       // noise is mean square
    return std::max(kFloorDb, (float) (10.0 * std::log10(std::max(ms, 1e-30))));
}

inline float netNorm(float db) { return (std::clamp(db, kFloorDb, 0.0f) - kFloorDb) / -kFloorDb; }

inline void netInput(const HarmFrame& f, float* x) {
    const float rms = netRmsDb(f);
    for (int i = 0; i < kNetHarmonics; ++i)
        x[i] = i < f.nValid && f.h[i] > kFloorDb ? netNorm(f.h[i] + netTaperDb(i) - rms) : 0.0f;
    for (int b = 0; b < kNumNoiseBands; ++b) x[kNetHarmonics + b] = f.noise[b] > kFloorDb ? netNorm(f.noise[b] - rms) : 0.0f;
    const bool voiced = (f.flags & kVoiced) && f.f0 > 0.0f;
    x[48] = voiced ? std::max(0.0f, std::log2(f.f0 / kNetF0Min) / std::log2(kNetF0Max / kNetF0Min)) : 0.0f;
    x[49] = voiced ? 1.0f : 0.0f;
    x[50] = netNorm(rms);
}

} // namespace larynx
