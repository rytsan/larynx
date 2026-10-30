#include "analyzer.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>

namespace larynx {

namespace {
int scaled(int samplesAt48k, double fs) { return (int) std::lround(samplesAt48k * fs / 48000.0); }
} // namespace

void Analyzer::prepare(double sampleRate) {
    fs_ = sampleRate;
    hop_ = scaled(256, fs_);       // §2.7: same 5.33 ms frame at any rate
    win_ = scaled(2048, fs_);
    specWin_ = scaled(1024, fs_);
    minLag_ = std::max(2, (int) std::floor(fs_ / tuning.fMax));
    maxLag_ = std::min(win_ / 2, (int) std::ceil(fs_ / tuning.fMin));
    floorPow_ = std::pow(10.0f, tuning.floorDb / 10.0f);
    centroidCoef_ = 1.0f - std::exp(-(float) hop_ / (0.05f * (float) fs_)); // ~50 ms average

    acFft_.prepare(std::bit_ceil((size_t) (2 * win_))); // zero-padded: linear, not circular, correlation
    specFft_.prepare(std::bit_ceil((size_t) specWin_));
    ring_.assign((size_t) win_, 0.0f);
    ringLp_.assign((size_t) win_, 0.0f);
    lin_.assign((size_t) win_, 0.0f);
    linLp_.assign((size_t) win_, 0.0f);
    {
        // Bright voices with vibrato otherwise lose NSDF clarity: upper partials sweep hundreds of Hz
        // inside the 43 ms window and decorrelate. 4 kHz also gave the fewest octave jumps on the
        // golden set (2/3/4/6 kHz: 32/21/18/22; none: 21).
        const float w0 = 2.0f * std::numbers::pi_v<float> * 4000.0f / (float) fs_;
        const float alpha = std::sin(w0) / std::numbers::sqrt2_v<float>, cw = std::cos(w0); // sin(w0)/(2Q), Q = 1/√2
        const float a0 = 1.0f + alpha;
        b0_ = b2_ = (1.0f - cw) / 2.0f / a0;
        b1_ = (1.0f - cw) / a0;
        a1_ = -2.0f * cw / a0;
        a2_ = (1.0f - alpha) / a0;
    }
    nsdf_.assign((size_t) maxLag_ + 2, 0.0f);
    acBuf_.assign(acFft_.size(), {});
    specBuf_.assign(specFft_.size(), {});
    hann_.resize((size_t) specWin_);
    for (int i = 0; i < specWin_; ++i)
        hann_[(size_t) i] = 0.5f - 0.5f * std::cos(2.0f * std::numbers::pi_v<float> * (float) i / (float) specWin_);
    reset();
}

void Analyzer::reset() {
    std::fill(ring_.begin(), ring_.end(), 0.0f);
    std::fill(ringLp_.begin(), ringLp_.end(), 0.0f);
    x1_ = x2_ = y1_ = y2_ = 0.0f;
    pos_ = count_ = 0;
    voiced_ = false;
    histLen_ = histPos_ = 0;
    prevX_ = eHop_ = edHop_ = 0.0f;
    edHist_.fill(0.0f);
    edPos_ = transientHold_ = 0;
    centroidAvg_ = 1000.0f;
}

bool Analyzer::push(float x, HarmFrame& f) {
    const float lp = b0_ * x + b1_ * x1_ + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;
    x2_ = x1_;
    x1_ = x;
    y2_ = y1_;
    y1_ = lp;
    ring_[(size_t) pos_] = x;
    ringLp_[(size_t) pos_] = lp;
    if (++pos_ == win_) pos_ = 0;
    const float d = x - prevX_;
    prevX_ = x;
    eHop_ += x * x;
    edHop_ += d * d;
    if (++count_ < hop_) return false;
    count_ = 0;
    analyze(f);
    return true;
}

int Analyzer::pickKeys() {
    // MPM: one key maximum per positive region, after leaving the zero-lag lobe.
    // Leave the lobe at the first zero crossing OR local minimum: low-frequency content (breath,
    // rumble at note edges) can keep it positive past the period, hiding the true key at T.
    int t = 1, n = 0;
    while (t <= maxLag_ && nsdf_[(size_t) t] > 0.0f && nsdf_[(size_t) t + 1] <= nsdf_[(size_t) t]) ++t;
    while (t <= maxLag_) {
        while (t <= maxLag_ && nsdf_[(size_t) t] <= 0.0f) ++t;
        int best = -1;
        for (; t <= maxLag_ && nsdf_[(size_t) t] > 0.0f; ++t)
            if (best < 0 || nsdf_[(size_t) t] > nsdf_[(size_t) best]) best = t;
        if (best < minLag_ || n == kMaxKeys) continue;
        const float a = nsdf_[(size_t) best - 1], b = nsdf_[(size_t) best], c = nsdf_[(size_t) best + 1];
        if (c > b) continue; // region cut by maxLag, not a real peak
        const float den = a - 2.0f * b + c;
        const float delta = den < 0.0f ? 0.5f * (a - c) / den : 0.0f;
        keys_[(size_t) n++] = { (float) best + delta, b - 0.25f * (a - c) * delta };
    }
    return n;
}

void Analyzer::analyze(HarmFrame& f) {
    // Linearize the ring: oldest sample first.
    std::copy(ring_.begin() + pos_, ring_.end(), lin_.begin());
    std::copy(ring_.begin(), ring_.begin() + pos_, lin_.begin() + (win_ - pos_));
    std::copy(ringLp_.begin() + pos_, ringLp_.end(), linLp_.begin());
    std::copy(ringLp_.begin(), ringLp_.begin() + pos_, linLp_.begin() + (win_ - pos_));

    double e0 = 0.0;
    for (float v : lin_) e0 += (double) v * v;
    const bool audible = e0 / win_ > floorPow_;

    // --- F0: NSDF n(τ) = 2 r(τ) / m(τ), r via FFT autocorrelation, m incremental.
    float clarity = 0.0f, f0 = 0.0f;
    if (audible) {
        const size_t n = acFft_.size();
        double mean = 0.0;
        for (float v : linLp_) mean += v;
        mean /= win_;
        for (auto& v : linLp_) v -= (float) mean; // DC biases every lag
        e0 = 0.0;
        for (float v : linLp_) e0 += (double) v * v;
        for (size_t i = 0; i < n; ++i) acBuf_[i] = i < (size_t) win_ ? linLp_[i] : 0.0f;
        acFft_.forward(acBuf_.data());
        for (auto& v : acBuf_) v = std::norm(v);
        acFft_.forward(acBuf_.data()); // real, even spectrum: forward == inverse up to 1/n
        double m = 2.0 * e0;
        for (int tau = 0; tau <= maxLag_ + 1; ++tau) {
            nsdf_[(size_t) tau] = m > 0.0 ? (float) (2.0 * acBuf_[(size_t) tau].real() / (double) n / m) : 0.0f;
            const double a = linLp_[(size_t) tau], b = linLp_[(size_t) (win_ - 1 - tau)];
            m -= a * a + b * b;
        }

        const int nk = pickKeys();
        float highest = 0.0f;
        for (int i = 0; i < nk; ++i) highest = std::max(highest, keys_[(size_t) i].val);
        for (int i = 0; i < nk; ++i)
            if (keys_[(size_t) i].val >= tuning.peakK * highest) {
                clarity = keys_[(size_t) i].val;
                f0 = (float) fs_ / keys_[(size_t) i].lag;
                break;
            }

        // Octave guard: the causal median of recent raw F0 only decides the octave. The value
        // always comes from this frame's NSDF, so the median's delay never leaks into F0 (§2.3).
        // Raw history means a real leap is accepted after 3 frames; 1-2 frame jumps are rejected.
        const float raw = f0;
        if (histLen_ >= 3 && f0 > 0.0f) {
            std::array<float, 5> s {};
            std::copy_n(hist_.begin(), histLen_, s.begin());
            std::nth_element(s.begin(), s.begin() + histLen_ / 2, s.begin() + histLen_);
            const float med = s[(size_t) histLen_ / 2];
            if (std::abs(std::log2(f0 / med)) > 0.75f) {
                clarity = 0.0f; // no candidate in the median's octave: unvoiced beats a wrong octave
                for (int i = 0; i < nk; ++i) {
                    const auto& k = keys_[(size_t) i];
                    if (std::abs(std::log2((float) fs_ / k.lag / med)) < 0.25f && k.val > tuning.voiceOff) {
                        clarity = k.val;
                        f0 = (float) fs_ / k.lag;
                        break;
                    }
                }
            }
        }

        voiced_ = clarity > (voiced_ ? tuning.voiceOff : tuning.voiceOn);
        if (voiced_) {
            hist_[(size_t) histPos_] = raw;
            histPos_ = (histPos_ + 1) % (int) hist_.size();
            histLen_ = std::min(histLen_ + 1, (int) hist_.size());
        }
    } else {
        voiced_ = false;
    }
    if (!voiced_) histLen_ = histPos_ = 0;

    // --- Sibilant: noise-like high band in an unvoiced, audible frame.
    bool sibilant = false;
    if (audible && !voiced_) {
        const size_t n = specFft_.size(), off = (size_t) (win_ - specWin_);
        for (size_t i = 0; i < n; ++i)
            specBuf_[i] = i < (size_t) specWin_ ? lin_[off + i] * hann_[i] : 0.0f;
        specFft_.forward(specBuf_.data());
        const auto binOf = [&](double hz) { return (size_t) (hz * (double) n / fs_); };
        const size_t lo = binOf(2000.0), hi = binOf(std::min(16000.0, 0.95 * fs_ / 2.0));
        double logSum = 0.0, sum = 0.0;
        for (size_t k = lo; k <= hi; ++k) {
            const double p = std::norm(specBuf_[k]) + 1e-20;
            logSum += std::log(p);
            sum += p;
        }
        const double cnt = (double) (hi - lo + 1);
        sibilant = std::exp(logSum / cnt) / (sum / cnt) > tuning.flatness;
    }

    // --- Transient: energy of the first difference (a +6 dB/oct high-pass) jumps against the
    // previous hops AND the RMS frequency sqrt(Ediff/E) jumps above its running average.
    // Held for one STFT window (4 frames) so every frame containing the hit is flagged.
    float prevEd = 0.0f;
    for (float v : edHist_) prevEd += v;
    prevEd /= (float) edHist_.size();
    const float eps = floorPow_ * (float) hop_;
    const float fluxDb = 10.0f * std::log10((edHop_ + eps) / (prevEd + eps));
    const bool loud = eHop_ > eps;
    const float centroid = loud ? (float) fs_ / (2.0f * std::numbers::pi_v<float>) * std::sqrt(edHop_ / eHop_) : 0.0f;
    if (fluxDb > tuning.fluxDb && centroid > tuning.centroidJump * centroidAvg_)
        transientHold_ = specWin_ / hop_;
    if (loud) centroidAvg_ += centroidCoef_ * (centroid - centroidAvg_);
    edHist_[(size_t) edPos_] = edHop_;
    edPos_ = (edPos_ + 1) % (int) edHist_.size();
    eHop_ = edHop_ = 0.0f;
    const bool transient = transientHold_ > 0;
    if (transientHold_ > 0) --transientHold_;

    f.f0 = voiced_ ? f0 : 0.0f;
    f.confidence = std::clamp(clarity, 0.0f, 1.0f);
    f.flags = (std::uint8_t) ((voiced_ ? kVoiced : 0) | (sibilant ? kSibilant : 0) | (transient ? kTransient : 0));
}

} // namespace larynx
