#include "synth.h"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace larynx {

namespace {
float dbToGain(float db) { return std::pow(10.0f, db / 20.0f); }
float ampOf(float db) { return db <= kFloorDb ? 0.0f : dbToGain(db); }
// Mask gains are bounded: a mask can only reshape what is there (plan B of §7 tightens this to ±12 dB).
float maskGain(float targetDb, float inDb) { return dbToGain(std::clamp(targetDb - inDb, -60.0f, 24.0f)); }
} // namespace

void Synth::prepare(const Skeleton& sk, double sampleRate) {
    fs_ = sampleRate;
    win_ = sk.win();
    hop_ = sk.hop();
    const int n = sk.fftSize();
    mixStep_ = (float) hop_ / (0.02f * (float) fs_);

    gain_.assign((size_t) n / 2 + 1, 1.0f);
    ola_.assign((size_t) win_, 0.0f);
    out_.assign((size_t) hop_, 0.0f);
    add_.assign((size_t) hop_, 0.0f);
    // OLA normalization: Σ w² over all frames covering each position (period = hop). Exact for any
    // window/hop pair, so 941/235 at 44.1k reconstructs as perfectly as 1024/256.
    norm_.assign((size_t) hop_, 0.0f);
    for (int i = 0; i < win_; ++i) norm_[(size_t) (i % hop_)] += sk.window()[(size_t) i] * sk.window()[(size_t) i];
    // The additive block for frame k spans the centres of frames k−1..k; the OLA emits a centre
    // win/2 samples after its frame. Block is generated one hop early → delay win/2 − hop.
    addDelay_.resize((size_t) (win_ / 2 - hop_));
    reset();
}

void Synth::reset() {
    std::fill(ola_.begin(), ola_.end(), 0.0f);
    std::fill(out_.begin(), out_.end(), 0.0f);
    std::fill(add_.begin(), add_.end(), 0.0f);
    addDelay_.reset();
    readPos_ = 0;
    present_.fill(false);
    mix_.fill(0.0f);
    ampPrev_.fill(0.0f);
    ampCur_.fill(0.0f);
    f0Prev_ = f0Cur_ = 0.0f;
    subPrev_ = subCur_ = 0.0f;
    dblPrev_.fill(0.0f);
    dblCur_.fill(0.0f);
    phase_ = phaseDbl_ = 0.0;
}

void Synth::render(Skeleton& sk, const HarmFrame& in, const HarmFrame& target) {
    auto& X = sk.spectrum();
    const int n = sk.fftSize(), half = n / 2;

    // --- Mask: noise-band gain everywhere, harmonic gain on each existing harmonic's cell.
    for (int k = 0; k <= half; ++k) {
        const int b = sk.bandOf(k);
        gain_[(size_t) k] = maskGain(target.noise[b], in.noise[b]);
    }
    // Existing/absent hysteresis (−90/−84 dB) with a 20 ms crossfade so a harmonic never flickers
    // between mask and additive. Also relative to the loudest harmonic: the √Hann sidelobes put
    // leakage around −60 dB re the peak, and leakage must count as absent, not be masked up.
    // ponytail: fixed 60/54 dB relative floor; make it per-band if noisy recordings misclassify.
    float loudest = kFloorDb;
    for (int i = 0; i < in.nValid; ++i) loudest = std::max(loudest, in.h[i]);
    const float offDb = std::max(-90.0f, loudest - 60.0f), onDb = std::max(-84.0f, loudest - 54.0f);
    // Each harmonic owns its Voronoi cell: bins closer to its measured peak than to a neighbour's.
    // A fixed ±2-bin lobe missed the energy that vibrato smears across the window (6–8 dB short of
    // target on upper harmonics). Harmonic gain replaces the band gain inside the cell.
    // ponytail: noise inside cells follows the harmonic gain; split per bin by residual share if
    // Breath (P8) needs independent control there.
    const float binsPerHz = (float) n / (float) fs_;
    const auto peakAt = [&](int i) { return in.hz[i] * binsPerHz; };
    for (int i = 0; i < kNumHarmonics; ++i) {
        const bool valid = i < in.nValid;
        present_[(size_t) i] = valid && in.h[i] > (present_[(size_t) i] ? offDb : onDb);
        auto& m = mix_[(size_t) i];
        m = present_[(size_t) i] ? std::min(1.0f, m + mixStep_) : std::max(0.0f, m - mixStep_);
        if (!valid || m <= 0.0f) continue;
        const float p = peakAt(i);
        const float lo = i > 0 ? 0.5f * (peakAt(i - 1) + p) : p - 0.5f * (peakAt(1) - p);
        const float hi = i + 1 < in.nValid ? 0.5f * (p + peakAt(i + 1)) : p + 0.5f * (p - (i > 0 ? peakAt(i - 1) : 0.0f));
        const float g = maskGain(target.h[i], in.h[i]);
        for (int k = std::max(0, (int) std::ceil(lo)); k < std::min(half + 1, (int) std::ceil(hi)); ++k)
            gain_[(size_t) k] += m * (g - gain_[(size_t) k]);
    }
    X[0] *= gain_[0];
    X[(size_t) half] *= gain_[(size_t) half];
    for (int k = 1; k < half; ++k) {
        X[(size_t) k] *= gain_[(size_t) k];
        X[(size_t) (n - k)] *= gain_[(size_t) k]; // keep Hermitian symmetry (real output)
    }

    // --- ISTFT: inverse via conj(FFT(conj(X))) / n, synthesis window, overlap-add.
    for (auto& v : X) v = std::conj(v);
    sk.fft().forward(X.data());
    const float scale = 1.0f / (float) n;
    for (int i = 0; i < win_; ++i) ola_[(size_t) i] += X[(size_t) i].real() * scale * sk.window()[(size_t) i];
    for (int j = 0; j < hop_; ++j) out_[(size_t) j] = ola_[(size_t) j] / norm_[(size_t) j];
    std::copy(ola_.begin() + hop_, ola_.end(), ola_.begin());
    std::fill(ola_.end() - hop_, ola_.end(), 0.0f);

    // --- Additive: absent harmonics get only what the target adds above the input (0 at identity).
    f0Prev_ = f0Cur_;
    if (target.f0 > 0.0f) f0Cur_ = target.f0; // unvoiced: keep the last pitch while amplitudes fade
    int active[kNumHarmonics], nActive = 0;
    for (int i = 0; i < kNumHarmonics; ++i) {
        ampPrev_[(size_t) i] = ampCur_[(size_t) i];
        const bool on = target.f0 > 0.0f && i < target.nValid;
        ampCur_[(size_t) i] = on ? (1.0f - mix_[(size_t) i]) * std::max(0.0f, ampOf(target.h[i]) - ampOf(in.h[i])) : 0.0f;
        if (ampPrev_[(size_t) i] > 0.0f || ampCur_[(size_t) i] > 0.0f) active[nActive++] = i;
    }
    subPrev_ = subCur_;
    subCur_ = 0.0f;
    if (subGain_ > 0.0f && target.f0 > 0.0f) {
        float env = 0.0f;
        for (int i = 0; i < std::min(4, (int) target.nValid); ++i) env += ampOf(target.h[i]);
        subCur_ = subGain_ * env / 4.0f;
    }
    dblPrev_ = dblCur_;
    bool dbl = false;
    for (int i = 0; i < kDoubleHarmonics; ++i) {
        dblCur_[(size_t) i] = dblGain_ > 0.0f && target.f0 > 0.0f && i < target.nValid ? dblGain_ * ampOf(target.h[i]) : 0.0f;
        dbl |= dblPrev_[(size_t) i] > 0.0f || dblCur_[(size_t) i] > 0.0f;
    }
    const double dblRatio = std::exp2(kDoubleCents / 1200.0);
    constexpr double twoPi = 2.0 * std::numbers::pi;
    for (int j = 0; j < hop_; ++j) {
        const float t = (float) (j + 1) / (float) hop_;
        phase_ += twoPi * (f0Prev_ + t * (f0Cur_ - f0Prev_)) / fs_;
        if (phase_ >= 4.0 * twoPi) phase_ -= 4.0 * twoPi; // wrap at 8π: f0/4 (phase/4) stays continuous
        float s = 0.0f;
        if (subPrev_ > 0.0f || subCur_ > 0.0f) {
            const float a = subPrev_ + t * (subCur_ - subPrev_);
            s += a * (float) std::sin(0.5 * phase_) + 0.5f * a * (float) std::sin(0.25 * phase_);
        }
        if (dbl) {
            phaseDbl_ += twoPi * dblRatio * (f0Prev_ + t * (f0Cur_ - f0Prev_)) / fs_;
            if (phaseDbl_ >= twoPi) phaseDbl_ -= twoPi;
            for (int i = 0; i < kDoubleHarmonics; ++i)
                s += (dblPrev_[(size_t) i] + t * (dblCur_[(size_t) i] - dblPrev_[(size_t) i])) * (float) std::sin((i + 1) * phaseDbl_);
        }
        for (int a = 0; a < nActive; ++a) {
            const int i = active[a];
            s += (ampPrev_[(size_t) i] + t * (ampCur_[(size_t) i] - ampPrev_[(size_t) i])) * (float) std::sin((i + 1) * phase_);
        }
        add_[(size_t) j] = s;
    }
    readPos_ = 0;
}

} // namespace larynx
