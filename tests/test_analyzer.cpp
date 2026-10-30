#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <functional>
#include <numbers>
#include <random>
#include <tuple>
#include <vector>
#include "core/analyzer.h"

using larynx::Analyzer;
using larynx::HarmFrame;

namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

struct Frame {
    double time; // window centre, seconds
    float f0;
    std::uint8_t flags;
};

std::vector<Frame> analyze(const std::vector<float>& x, double fs) {
    Analyzer a;
    a.prepare(fs);
    std::vector<Frame> out;
    HarmFrame f;
    for (size_t n = 0; n < x.size(); ++n)
        if (a.push(x[n], f))
            out.push_back({ ((double) n + 1.0 - a.window() / 2.0) / fs, f.f0, f.flags });
    return out;
}

// Phase-continuous oscillator for an arbitrary f0(t); `harm(k)` = amplitude of harmonic k.
std::vector<float> tone(double fs, double seconds, const std::function<double(double)>& f0,
                        const std::function<double(int)>& harm, int nHarm) {
    std::vector<float> x((size_t) (fs * seconds));
    double phase = 0.0;
    for (size_t n = 0; n < x.size(); ++n) {
        double v = 0.0;
        for (int k = 1; k <= nHarm; ++k) v += harm(k) * std::sin(k * phase);
        x[n] = (float) v;
        phase += kTwoPi * f0((double) n / fs) / fs;
    }
    return x;
}

double cents(double f, double ref) { return 1200.0 * std::log2(f / ref); }

} // namespace

TEST_CASE("sines 65-1000 Hz: error < 10 cents", "[analyzer]") {
    for (double fs : { 44100.0, 48000.0 }) {
        for (int i = 0; i <= 20; ++i) {
            const double hz = 65.0 * std::pow(1000.0 / 65.0, i / 20.0);
            const auto frames = analyze(tone(fs, 0.5, [=](double) { return hz; }, [](int) { return 0.5; }, 1), fs);
            double worst = 0.0;
            for (const auto& fr : frames) {
                if (fr.time < 0.1) continue;
                REQUIRE((fr.flags & larynx::kVoiced));
                worst = std::max(worst, std::abs(cents(fr.f0, hz)));
            }
            INFO("fs " << fs << " hz " << hz);
            CHECK(worst < 10.0);
        }
    }
}

TEST_CASE("vibrato 6 Hz +-50 cents: instantaneous error < 15 cents", "[analyzer]") {
    const double fs = 48000.0, base = 220.0;
    const auto f0 = [=](double t) { return base * std::pow(2.0, 50.0 / 1200.0 * std::sin(kTwoPi * 6.0 * t)); };
    const auto frames = analyze(tone(fs, 2.0, f0, [](int) { return 0.5; }, 1), fs);
    double worst = 0.0;
    for (const auto& fr : frames) {
        if (fr.time < 0.1) continue;
        REQUIRE((fr.flags & larynx::kVoiced));
        worst = std::max(worst, std::abs(cents(fr.f0, f0(fr.time))));
    }
    CHECK(worst < 15.0);
}

TEST_CASE("sustained vowel 30 s, weak fundamental: no octave jump", "[analyzer]") {
    // /a/-like: formants 700/1200/2600 Hz, H1 20 dB down, 5 Hz vibrato, 0.5% jitter, noise at -45 dB.
    const double fs = 48000.0, base = 110.0;
    std::mt19937 rng(7);
    std::normal_distribution<double> jit(0.0, 0.005);
    std::vector<double> jitter((size_t) (30 * 200) + 2);
    for (auto& j : jitter) j = jit(rng);
    const auto f0 = [&](double t) {
        const double vib = std::pow(2.0, 30.0 / 1200.0 * std::sin(kTwoPi * 5.0 * t));
        const size_t i = (size_t) (t * 200.0); // jitter changes every 5 ms, linearly interpolated
        const double fr = t * 200.0 - (double) i;
        return base * vib * (1.0 + jitter[i] * (1.0 - fr) + jitter[i + 1] * fr);
    };
    const auto formant = [](double hz) {
        double g = 0.0;
        for (auto [fc, bw, amp] : { std::tuple { 700.0, 110.0, 1.0 }, std::tuple { 1200.0, 120.0, 0.5 }, std::tuple { 2600.0, 160.0, 0.15 } })
            g += amp / (1.0 + std::pow((hz - fc) / bw, 2.0));
        return g;
    };
    auto x = tone(fs, 30.0, f0, [&](int k) { return 0.3 * formant(k * base) * (k == 1 ? 0.1 : 1.0); }, 60);
    std::normal_distribution<float> nz(0.0f, 0.3f * 0.0056f);
    for (auto& v : x) v += nz(rng);

    const auto frames = analyze(x, fs);
    int voiced = 0, total = 0, jumps = 0;
    for (const auto& fr : frames) {
        if (fr.time < 0.1) continue;
        ++total;
        if (!(fr.flags & larynx::kVoiced)) continue;
        ++voiced;
        if (std::abs(cents(fr.f0, f0(fr.time))) > 100.0) ++jumps;
    }
    CHECK(jumps == 0);
    CHECK(voiced >= total * 99 / 100);
}

TEST_CASE("silence and noise are unvoiced; noise is sibilant; voicing recovers", "[analyzer]") {
    const double fs = 48000.0;
    std::vector<float> x((size_t) fs * 3, 0.0f);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-0.3f, 0.3f);
    for (size_t n = (size_t) fs; n < (size_t) (2 * fs); ++n) x[n] = u(rng);                         // 1-2 s: noise
    for (size_t n = (size_t) (2 * fs); n < x.size(); ++n) x[n] = 0.5f * (float) std::sin(kTwoPi * 200.0 * (double) n / fs); // 2-3 s: sine

    int noiseFrames = 0, sib = 0;
    for (const auto& fr : analyze(x, fs)) {
        const bool voiced = fr.flags & larynx::kVoiced;
        if (fr.time < 0.9) CHECK(fr.flags == 0);
        else if (fr.time > 1.1 && fr.time < 1.9) {
            CHECK(!voiced);
            ++noiseFrames;
            sib += (fr.flags & larynx::kSibilant) ? 1 : 0;
        } else if (fr.time > 2.1) {
            CHECK(voiced);
            CHECK(!(fr.flags & larynx::kSibilant));
        }
    }
    CHECK(sib >= noiseFrames * 9 / 10);
}

TEST_CASE("noise bursts on a vowel are transients; the vowel itself is not", "[analyzer]") {
    const double fs = 48000.0;
    auto x = tone(fs, 2.0, [](double) { return 220.0; }, [](int k) { return 0.3 / k; }, 20);
    for (size_t n = 0; n < (size_t) (0.02 * fs); ++n) x[n] *= (float) n / (float) (0.02 * fs); // soft onset
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    const double hits[] = { 0.5, 1.0, 1.5 };
    for (double h : hits)
        for (size_t i = 0; i < (size_t) (0.003 * fs); ++i)
            x[(size_t) (h * fs) + i] += u(rng) * std::exp(-(float) i / (float) (0.001 * fs));

    std::vector<bool> found(3, false);
    for (const auto& fr : analyze(x, fs)) {
        const bool t = fr.flags & larynx::kTransient;
        // Frame reports the window ending at time + window/2; the hit is inside the newest hop then.
        const double end = fr.time + 1024.0 / fs;
        bool near = false;
        for (int i = 0; i < 3; ++i)
            if (end >= hits[i] && end < hits[i] + 0.03) {
                near = true;
                found[(size_t) i] = found[(size_t) i] || t;
            }
        if (!near) CHECK(!t);
    }
    for (int i = 0; i < 3; ++i) CHECK(found[(size_t) i]);
}

TEST_CASE("bright voice with vibrato stays voiced", "[analyzer]") {
    // Flat spectrum to H40 (12 kHz): upper partials carry most energy and sweep ±50 cents at 5.5 Hz.
    const double fs = 48000.0, base = 300.0;
    const auto f0 = [=](double t) { return base * std::pow(2.0, 50.0 / 1200.0 * std::sin(kTwoPi * 5.5 * t)); };
    const auto frames = analyze(tone(fs, 2.0, f0, [](int) { return 0.02; }, 40), fs);
    int voiced = 0, total = 0;
    for (const auto& fr : frames) {
        if (fr.time < 0.1) continue;
        ++total;
        voiced += (fr.flags & larynx::kVoiced) ? 1 : 0;
    }
    CHECK(voiced == total);
}
