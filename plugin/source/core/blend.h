#pragma once
// M4 (§8): target envelope from the input frame, in the order of §2.1:
//   (1) neural delta scaled by Prior (P6; absent until then), (2) manual operators — may cut below the
//   input, (3) spectral Mix in dB. All controls at 0 and Mix = 100% leave target == in bit for bit, so
//   the P3 identity path is untouched. The network may create absent harmonics; the manual operators
//   only shape harmonics present after it (> floor), never conjure partials from nothing. No JUCE (core/ rule), no allocation.
#include <algorithm>
#include <cmath>
#include "harm_frame.h"

namespace larynx {

struct Controls {
    float prior = 0.7f;   // 0..1, scales the neural delta
    float mix = 1.0f;     // 0..1, dB interpolation input → shaped
    float bright = 0.0f;  // dB/oct tilt around H1
    float air = 0.0f;     // dB, shelf at 8 kHz on harmonics and noise bands
    float evenodd = 0.0f; // -1..1 → ±kEvenOddDb on even harmonics, ∓ on odd ones (H1 untouched)
    float formant = 0.0f; // semitones, envelope reindexing (F0 untouched)
    float sub = 0.0f;     // 0..1, M5: handled by Synth::setSub (not an envelope operator)
    float breath = 0.0f;  // 0..1, M5: up to +kBreathDb on the noise bands of voiced frames, more on open vowels
};

constexpr float kEvenOddDb = 6.0f;
constexpr float kAirHz = 8000.0f;
constexpr float kBreathDb = 12.0f;

inline char vowelOf(const HarmFrame& f, float* f1 = nullptr, float* f2 = nullptr);

// Shelf weight 0..1: smoothstep over one octave centred on 8 kHz.
inline float airWeight(float hz) {
    const float t = std::clamp(std::log2(std::max(hz, 1.0f) / kAirHz) + 0.5f, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Centre of ERB noise band b (bands evenly spaced on the ERB-rate scale up to Nyquist, as in skeleton).
inline float noiseBandHz(int b, double fs) {
    const double top = 21.4 * std::log10(1.0 + 0.00437 * fs / 2.0);
    const double e = top * ((double) b + 0.5) / (double) kNumNoiseBands;
    return (float) ((std::pow(10.0, e / 21.4) - 1.0) / 0.00437);
}

// `net`: per-frame neural deltas (H1..H80 then 8 noise bands, dB) or nullptr.
inline void blend(const HarmFrame& in, HarmFrame& t, const Controls& c, double fs, const float* net = nullptr) {
    t = in;
    const int nv = in.nValid;
    if (net && c.prior > 0.0f) {
        for (int i = 0; i < nv; ++i) t.h[i] = in.h[i] + c.prior * net[i];
        for (int b = 0; b < kNumNoiseBands; ++b) t.noise[b] = in.noise[b] + c.prior * net[kNumHarmonics + b];
    }

    if (std::abs(c.formant) > 1e-6f && in.f0 > 0.0f) {
        // h'(n) = h(n / α): the envelope sampled at n·f0/α, linear in dB between harmonics.
        const float a = std::exp2(c.formant / 12.0f);
        HarmFrame src = t;
        for (int i = 0; i < nv; ++i) {
            if (src.h[i] <= kFloorDb) continue;
            const float pos = std::clamp((float) (i + 1) / a - 1.0f, 0.0f, (float) (nv - 1));
            const int k = std::min((int) pos, nv - 2 < 0 ? 0 : nv - 2);
            const float fr = nv > 1 ? pos - (float) k : 0.0f;
            t.h[i] = src.h[k] + fr * (src.h[std::min(k + 1, nv - 1)] - src.h[k]);
        }
    }
    for (int i = 0; i < nv; ++i) {
        if (t.h[i] <= kFloorDb) continue;
        const int n = i + 1;
        float g = c.bright * std::log2((float) n);
        if (n > 1) g += (n % 2 == 0 ? 1.0f : -1.0f) * c.evenodd * kEvenOddDb;
        g += c.air * airWeight((float) n * in.f0);
        t.h[i] += g;
    }
    if (std::abs(c.air) > 1e-6f)
        for (int b = 0; b < kNumNoiseBands; ++b)
            if (t.noise[b] > kFloorDb) t.noise[b] += c.air * airWeight(noiseBandHz(b, fs));

    if (c.breath > 0.0f && (in.flags & kVoiced)) {
        // Open-vowel detector: F1 300 Hz (closed) → 800 Hz (open) maps to 0.5 → 1 of the breath gain.
        float f1 = 0.0f;
        const float open = vowelOf(in, &f1) ? std::clamp((f1 - 300.0f) / 500.0f, 0.0f, 1.0f) : 0.5f;
        for (auto& v : t.noise)
            if (v > kFloorDb) v += c.breath * kBreathDb * (0.5f + 0.5f * open);
    }

    if (c.mix < 1.0f) {
        for (int i = 0; i < nv; ++i) t.h[i] = in.h[i] + c.mix * (t.h[i] - in.h[i]);
        for (int b = 0; b < kNumNoiseBands; ++b) t.noise[b] = in.noise[b] + c.mix * (t.noise[b] - in.noise[b]);
    }
    for (int i = 0; i < nv; ++i) t.h[i] = std::clamp(t.h[i], kFloorDb, 0.0f);
    for (auto& v : t.noise) v = std::clamp(v, kFloorDb, 0.0f);
}

// Vowel reader for the graph (§8): F1/F2 = the two strongest local maxima of the harmonic envelope
// within 250-2500 Hz (parabolic interpolation over neighbouring harmonics), lower one is F1; nearest
// of a/e/i/o/u in log-frequency. Returns 0 when unvoiced or fewer than two maxima. ponytail: harmonic
// sampling is coarse for high voices (f0 > ~400 Hz leaves few points), fine for a display.
inline char vowelOf(const HarmFrame& f, float* f1, float* f2) {
    if (!(f.flags & kVoiced) || f.f0 <= 0.0f) return 0;
    float pk[2] = { 0.0f, 0.0f }, lv[2] = { kFloorDb, kFloorDb };
    for (int i = 1; i + 1 < f.nValid; ++i) {
        const float hz = (float) (i + 1) * f.f0;
        if (hz < 250.0f || hz > 2500.0f || f.h[i] <= kFloorDb || f.h[i] < f.h[i - 1] || f.h[i] <= f.h[i + 1]) continue;
        const float a = f.h[i - 1], b = f.h[i], c = f.h[i + 1], den = a - 2.0f * b + c;
        const float d = den < 0.0f ? std::clamp(0.5f * (a - c) / den, -0.5f, 0.5f) : 0.0f;
        const float at = ((float) (i + 1) + d) * f.f0;
        if (b > lv[0]) pk[1] = pk[0], lv[1] = lv[0], pk[0] = at, lv[0] = b;
        else if (b > lv[1]) pk[1] = at, lv[1] = b;
    }
    if (lv[1] <= kFloorDb) return 0;
    const float F1 = std::min(pk[0], pk[1]), F2 = std::max(pk[0], pk[1]);
    if (f1) *f1 = F1;
    if (f2) *f2 = F2;
    struct V { char c; float f1, f2; };
    constexpr V vowels[] = { { 'a', 750, 1300 }, { 'e', 450, 1950 }, { 'i', 300, 2300 }, { 'o', 480, 900 }, { 'u', 330, 800 } };
    char best = 0;
    float bd = 1e9f;
    for (auto v : vowels) {
        const float dist = std::pow(std::log2(F1 / v.f1), 2.0f) + std::pow(std::log2(F2 / v.f2), 2.0f);
        if (dist < bd) bd = dist, best = v.c;
    }
    return best;
}

} // namespace larynx
