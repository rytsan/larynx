#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include "core/blend.h"

using larynx::Controls;
using larynx::HarmFrame;

namespace {

// Voiced frame, f0 = 200 Hz, flat −30 dB envelope with a bump at H5 (1 kHz) and 60 valid harmonics.
HarmFrame frame() {
    HarmFrame f {};
    f.f0 = 200.0f;
    f.flags = larynx::kVoiced;
    f.nValid = 60;
    for (int i = 0; i < larynx::kNumHarmonics; ++i) f.h[i] = i < 60 ? -40.0f + 10.0f * std::exp(-std::pow((float) (i - 4), 2.0f) / 2.0f) : larynx::kFloorDb;
    for (auto& v : f.noise) v = -70.0f;
    return f;
}

Controls zero() {
    Controls c;
    c.prior = 0.0f;
    return c;
}

} // namespace

TEST_CASE("blend: all controls at 0 is bit-identical (P4 regression of P3)", "[blend]") {
    const auto in = frame();
    HarmFrame t;
    for (float prior : { 0.0f, 0.7f, 1.0f }) {
        auto c = zero();
        c.prior = prior; // no neural delta yet: Prior alone changes nothing
        larynx::blend(in, t, c, 48000.0);
        REQUIRE(std::memcmp(in.h, t.h, sizeof in.h) == 0);
        REQUIRE(std::memcmp(in.noise, t.noise, sizeof in.noise) == 0);
    }
}

TEST_CASE("blend: Prior = 0 with Brightness +6 still shapes (M4 order)", "[blend]") {
    const auto in = frame();
    HarmFrame t;
    auto c = zero();
    c.bright = 6.0f;
    larynx::blend(in, t, c, 48000.0);
    REQUIRE(t.h[0] == in.h[0]);                          // H1 is the pivot
    REQUIRE(std::abs(t.h[7] - in.h[7] - 18.0f) < 1e-4f); // H8: 3 octaves × 6 dB
    c.bright = -6.0f;
    larynx::blend(in, t, c, 48000.0);
    REQUIRE(t.h[7] < in.h[7] - 17.9f);                   // cuts below the input
    REQUIRE(t.h[60] == larynx::kFloorDb);                // absent harmonics stay absent
}

TEST_CASE("blend: even/odd, air, mix", "[blend]") {
    const auto in = frame();
    HarmFrame t;
    auto c = zero();
    c.evenodd = 1.0f;
    larynx::blend(in, t, c, 48000.0);
    REQUIRE(t.h[0] == in.h[0]);
    REQUIRE(std::abs(t.h[1] - in.h[1] - larynx::kEvenOddDb) < 1e-4f);
    REQUIRE(std::abs(t.h[2] - in.h[2] + larynx::kEvenOddDb) < 1e-4f);

    c = zero();
    c.air = 6.0f;
    larynx::blend(in, t, c, 48000.0);
    REQUIRE(std::abs(t.h[9] - in.h[9]) < 1e-4f);         // 2 kHz: below the shelf
    REQUIRE(std::abs(t.h[59] - in.h[59] - 6.0f) < 1e-4f); // 12 kHz: full shelf
    REQUIRE(t.noise[7] > in.noise[7] + 5.0f);            // top ERB band follows
    REQUIRE(std::abs(t.noise[0] - in.noise[0]) < 1e-4f);

    c.mix = 0.5f;
    larynx::blend(in, t, c, 48000.0);
    REQUIRE(std::abs(t.h[59] - in.h[59] - 3.0f) < 1e-4f);
    c.mix = 0.0f;
    larynx::blend(in, t, c, 48000.0);
    REQUIRE(std::memcmp(in.h, t.h, sizeof in.h) == 0);
}

TEST_CASE("blend: formant +12 st moves the envelope peak one octave up, f0 untouched", "[blend]") {
    const auto in = frame();
    HarmFrame t;
    auto c = zero();
    c.formant = 12.0f;
    larynx::blend(in, t, c, 48000.0);
    REQUIRE(t.f0 == in.f0);
    int peak = 0;
    for (int i = 1; i < 40; ++i)
        if (t.h[i] > t.h[peak]) peak = i;
    REQUIRE(peak == 9); // H5 → H10
}

TEST_CASE("vowel reader: nearest of a/e/i/o/u", "[blend]") {
    // f0 = 100 Hz, peaks at F1/F2 of each reference vowel.
    const float ref[][2] = { { 750, 1300 }, { 450, 1950 }, { 300, 2300 }, { 480, 900 }, { 330, 800 } };
    const char expect[] = { 'a', 'e', 'i', 'o', 'u' };
    for (int v = 0; v < 5; ++v) {
        HarmFrame f {};
        f.f0 = 100.0f;
        f.flags = larynx::kVoiced;
        f.nValid = 80;
        for (int i = 0; i < 80; ++i) {
            const float hz = 100.0f * (float) (i + 1);
            const auto bump = [&](float c) { return -60.0f + 40.0f * std::exp(-std::pow((hz - c) / 60.0f, 2.0f)); };
            f.h[i] = std::max(bump(ref[v][0]), bump(ref[v][1]));
        }
        CHECK(larynx::vowelOf(f) == expect[v]);
    }
}
