#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>
#include "core/engine.h"

using larynx::Engine;
using larynx::HarmFrame;

namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

// Harmonic tone, amplitudes in dB per harmonic (index 0 = H1), fixed phases.
std::vector<float> harmonicTone(double fs, double seconds, double f0, const std::vector<float>& db) {
    std::vector<float> x((size_t) (fs * seconds));
    for (size_t n = 0; n < x.size(); ++n) {
        double v = 0.0;
        for (size_t k = 0; k < db.size(); ++k)
            v += std::pow(10.0, db[k] / 20.0) * std::sin(kTwoPi * f0 * (double) (k + 1) * (double) n / fs + 0.7 * (double) k);
        x[n] = (float) v;
    }
    return x;
}

// Runs x through a fresh engine; returns the frames of the last `seconds` (steady state).
template <class Shape>
std::vector<HarmFrame> frames(const std::vector<float>& x, double fs, Shape shape, std::vector<float>* out = nullptr) {
    Engine e;
    e.prepare(fs);
    std::vector<HarmFrame> fr;
    int hopCount = 0;
    for (size_t n = 0; n < x.size(); ++n) {
        const float y = e.process(x[n], [&](const HarmFrame& in, HarmFrame& t) {
            shape(in, t);
            if (++hopCount > 40) fr.push_back(in); // skip warm-up (2048 window)
        });
        if (out) out->push_back(y);
    }
    return fr;
}
auto identity = [](const HarmFrame&, HarmFrame&) {};

std::vector<float> vowelWithNoise(double fs, double seconds) {
    std::vector<float> db;
    for (int k = 1; k <= 60; ++k) db.push_back(-12.0f - 9.0f * (float) std::log2(k));
    auto x = harmonicTone(fs, seconds, 196.0, db);
    std::mt19937 rng(5);
    std::normal_distribution<float> nz(0.0f, 0.003f);
    for (size_t n = 0; n < x.size() / 2; ++n) x[n] *= (float) n / (float) (x.size() / 2); // onset ramp
    for (auto& v : x) v += nz(rng);
    return x;
}

} // namespace

TEST_CASE("identity: gains = 1 gives out[n] = in[n - window] within -100 dBFS", "[engine]") {
    for (double fs : { 44100.0, 48000.0 }) {
        Engine e;
        e.prepare(fs);
        const int L = e.latency();
        REQUIRE(L == (int) std::lround(1024.0 * fs / 48000.0));
        const auto in = vowelWithNoise(fs, 1.0);
        float worst = 0.0f;
        for (size_t n = 0; n < in.size(); ++n) {
            const float expected = n < (size_t) L ? 0.0f : in[n - (size_t) L];
            worst = std::max(worst, std::abs(e.process(in[n]) - expected));
        }
        INFO("fs " << fs);
        CHECK(20.0f * std::log10(worst + 1e-30f) < -100.0f);
    }
}

TEST_CASE("extraction: a sine of X dB is re-extracted at X +- 0.2 dB, any frequency", "[engine]") {
    for (double fs : { 44100.0, 48000.0 }) {
        for (int i = 0; i < 18; ++i) {
            const double hz = 110.0 * std::pow(1.137, i) + 3.3; // off-bin, up to ~1 kHz (analyzer fMax 1.2 kHz)
            const float db = -6.0f - (float) i;
            float worst = 0.0f;
            for (const auto& f : frames(harmonicTone(fs, 0.4, hz, { db }), fs, identity)) {
                REQUIRE(f.f0 > 0.0f);
                worst = std::max(worst, std::abs(f.h[0] - db));
            }
            INFO("fs " << fs << " hz " << hz);
            CHECK(worst < 0.2f);
        }
    }
}

TEST_CASE("extraction: harmonic stack re-extracted within 0.2 dB, small residual", "[engine]") {
    for (double f0 : { 220.0, 330.0, 523.0 }) {
        std::vector<float> db(30, -24.0f);
        for (size_t k = 0; k < db.size(); ++k) db[k] -= 0.5f * (float) k;
        for (const auto& f : frames(harmonicTone(48000.0, 0.4, f0, db), 48000.0, identity)) {
            REQUIRE(f.nValid >= 30);
            for (int k = 0; k < 30; ++k) {
                INFO("f0 " << f0 << " H" << k + 1);
                REQUIRE(std::abs(f.h[k] - db[(size_t) k]) < 0.2f);
            }
            for (int b = 0; b < 8; ++b) {
                INFO("f0 " << f0 << " band " << b);
                CHECK(f.noise[b] < -54.0f); // pure harmonics (H1 at -24 dB) leave a residual >= 30 dB down
            }
        }
    }
}

TEST_CASE("noise bands carry the energy of unvoiced noise", "[engine]") {
    std::mt19937 rng(9);
    std::normal_distribution<float> nz(0.0f, 0.05f);
    std::vector<float> x(48000);
    for (auto& v : x) v = nz(rng);
    for (const auto& f : frames(x, 48000.0, identity)) {
        REQUIRE(f.nValid == 0);
        double ms = 0.0;
        for (float nb : f.noise) ms += std::pow(10.0, nb / 10.0);
        CHECK(std::abs(10.0 * std::log10(ms) - 20.0 * std::log10(0.05)) < 1.0);
    }
}

TEST_CASE("mask: -6 dB on H3 lands at -6 dB, neighbours untouched", "[engine]") {
    const double fs = 48000.0;
    const std::vector<float> db(10, -20.0f);
    std::vector<float> out;
    frames(harmonicTone(fs, 0.6, 220.0, db), fs, [](const HarmFrame& in, HarmFrame& t) { t.h[2] = in.h[2] - 6.0f; }, &out);
    for (const auto& f : frames(out, fs, identity)) {
        CHECK(std::abs(f.h[2] - -26.0f) < 0.3f);
        CHECK(std::abs(f.h[1] - -20.0f) < 0.2f);
        CHECK(std::abs(f.h[3] - -20.0f) < 0.2f);
    }
}

TEST_CASE("additive: absent H15 injected at -30 dB is re-extracted at -30 +- 0.2 dB", "[engine]") {
    const double fs = 48000.0;
    const std::vector<float> db(10, -20.0f);
    std::vector<float> out;
    frames(harmonicTone(fs, 0.6, 220.0, db), fs, [](const HarmFrame&, HarmFrame& t) { t.h[14] = -30.0f; }, &out);
    for (const auto& f : frames(out, fs, identity)) {
        CHECK(std::abs(f.h[14] - -30.0f) < 0.2f);
        CHECK(std::abs(f.h[0] - -20.0f) < 0.1f); // existing harmonics pass through the identity mask
    }
}
