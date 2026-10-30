#include "skeleton.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>

namespace larynx {

namespace {
constexpr float kPi = std::numbers::pi_v<float>;
float erbRate(double hz) { return (float) (21.4 * std::log10(1.0 + 0.00437 * hz)); }
float toDb(float amp) { return std::clamp(20.0f * std::log10(std::max(amp, 1e-6f)), kFloorDb, 0.0f); }
} // namespace

float Skeleton::signedLobe(float d) {
    // Transform of sin(πt) on [0,1], normalized to 1 at δ = 0: cos(πδ) / (1 − 4δ²), π/4 at |δ| = 0.5.
    // Sampled at N = 1024 the discrete sum matches this to O(1/N²) (window is 0 at both ends).
    const float den = 1.0f - 4.0f * d * d;
    return std::abs(den) < 1e-4f ? kPi / 4.0f : std::cos(kPi * d) / den;
}

void Skeleton::prepare(double sampleRate) {
    fs_ = sampleRate;
    win_ = (int) std::lround(1024.0 * fs_ / 48000.0); // §2.7: 941 @44.1k, zero-padded to 1024
    hop_ = (int) std::lround(256.0 * fs_ / 48000.0);
    fft_.prepare(std::bit_ceil((size_t) win_));
    const size_t n = fft_.size();

    w_.resize((size_t) win_);
    sumW_ = sumW2_ = 0.0f;
    for (int i = 0; i < win_; ++i) {
        w_[(size_t) i] = std::sin(kPi * (float) i / (float) win_); // periodic √Hann
        sumW_ += w_[(size_t) i];
        sumW2_ += w_[(size_t) i] * w_[(size_t) i];
    }
    X_.assign(n, {});
    pow_.assign(n / 2 + 1, 0.0f);
    res_.assign(n / 2 + 1, 0.0f);

    // 8 bands evenly spaced on the ERB-rate scale, 0 Hz to Nyquist.
    band_.resize(n / 2 + 1);
    const float top = erbRate(fs_ / 2.0);
    for (size_t k = 0; k <= n / 2; ++k)
        band_[k] = std::min(kNumNoiseBands - 1, (int) ((float) kNumNoiseBands * erbRate((double) k * fs_ / (double) n) / top));
}

void Skeleton::analyze(const float* window, HarmFrame& f) {
    const int n = (int) fft_.size(), half = n / 2;
    for (int i = 0; i < n; ++i) X_[(size_t) i] = i < win_ ? window[i] * w_[(size_t) i] : 0.0f;
    fft_.forward(X_.data());
    for (int k = 0; k <= half; ++k) {
        pow_[(size_t) k] = std::norm(X_[(size_t) k]);
        res_[(size_t) k] = X_[(size_t) k];
    }

    const float binHz = (float) fs_ / (float) n;
    const float s = (float) win_ / (float) n; // fft bin spacing in window bins (1 at 48k, <1 when zero-padded)
    const float ampScale = 2.0f / sumW_;     // peak |X| of a sine of amplitude A is A·Σw/2
    const float f0 = f.f0;

    int nValid = 0;
    if (f0 > 0.0f)
        while (nValid < kNumHarmonics && (float) (nValid + 1) * f0 < 0.95f * (float) fs_ / 2.0f) ++nValid;
    f.nValid = (std::uint8_t) nValid;

    for (int i = 0; i < kNumHarmonics; ++i) {
        f.h[i] = kFloorDb;
        f.hz[i] = (float) (i + 1) * f0;
        peakBin_[(size_t) i] = -1;
    }

    for (int i = 0; i < nValid; ++i) {
        const float c = (float) (i + 1) * f0 / binHz; // expected position, fft bins
        int k;
        float pos; // refined peak position, fft bins
        if (f0 < 100.0f) {
            // Harmonics closer than ~2 bins are not resolved by the 1024 window: take the bin, trust F0.
            k = std::clamp((int) std::lround(c), 1, half - 1);
            pos = c;
        } else {
            const float r = 0.5f * f0 / binHz;
            const int lo = std::max(1, (int) std::ceil(c - r)), hi = std::min(half - 1, (int) std::floor(c + r));
            k = lo;
            for (int j = lo + 1; j <= hi; ++j)
                if (pow_[(size_t) j] > pow_[(size_t) k]) k = j;
            // Exact for the window model: solve L(s − δ) / L(δ) = |X[k±1]| / |X[k]| for δ ∈ [0, s/2].
            const bool right = pow_[(size_t) k + 1] >= pow_[(size_t) k - 1];
            const float ratio = std::sqrt(pow_[(size_t) (right ? k + 1 : k - 1)] / std::max(pow_[(size_t) k], 1e-30f));
            float a = 0.0f, b = 0.5f * s;
            for (int it = 0; it < 24; ++it) {
                const float m = 0.5f * (a + b);
                (std::abs(signedLobe(s - m) / signedLobe(m)) < ratio ? a : b) = m;
            }
            pos = (float) k + (right ? 1.0f : -1.0f) * 0.5f * (a + b) / s;
        }
        const float amp = std::sqrt(pow_[(size_t) k]) * ampScale / std::abs(signedLobe(((float) k - pos) * s));
        f.h[i] = toDb(amp);
        f.hz[i] = pos * binHz;
        peakBin_[(size_t) i] = k;

        // Residual: subtract this harmonic's complex window model C·e^{-iπδ}·L(δ), L(δ) = cos(πδ)/(1−4δ²)
        // (the √Hann window is symmetric about N/2, so its transform has exactly linear phase).
        // Subtracting complex values, not powers, removes the cross terms between overlapping lobes.
        // ponytail: one pass, C taken from the raw peak bin; floor ~40 dB under the harmonics.
        // Re-estimate C from the running residual (2nd pass) if breathy-voice noise needs more.
        const float d0 = ((float) k - pos) * s;
        const std::complex<float> cpx = X_[(size_t) k] / (std::polar(1.0f, -kPi * d0) * signedLobe(d0));
        const int span = (int) std::ceil(8.0f / s); // sidelobes beyond 8 bins are < −48 dB
        for (int j = std::max(1, k - span); j <= std::min(half - 1, k + span); ++j) {
            const float d = ((float) j - pos) * s;
            res_[(size_t) j] -= cpx * std::polar(signedLobe(d), -kPi * d);
        }
    }

    // Noise bands: mean-square of the residual per ERB band (Parseval, one-sided), in dB.
    float band[kNumNoiseBands] = {};
    for (int k = 1; k < half; ++k) band[band_[(size_t) k]] += std::norm(res_[(size_t) k]);
    for (int b = 0; b < kNumNoiseBands; ++b) {
        const float ms = 2.0f * band[b] / ((float) n * sumW2_);
        f.noise[b] = std::clamp(10.0f * std::log10(std::max(ms, 1e-12f)), kFloorDb, 0.0f);
    }
}

} // namespace larynx
