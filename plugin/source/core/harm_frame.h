#pragma once
// Internal contract between modules (§2.2). One frame per hop (256 @48k).
// f0, h and noise refer to the same instant (end of the analysis windows, §2.3).
#include <cstdint>

namespace larynx {

constexpr int kNumHarmonics = 80;
constexpr int kNumNoiseBands = 8;
constexpr float kFloorDb = -100.0f;

enum HarmFlags : std::uint8_t {
    kVoiced = 1 << 0,
    kSibilant = 1 << 1,
    kTransient = 1 << 2,
};

struct HarmFrame {
    float f0 = 0.0f;         // Hz (0 = unvoiced)
    float confidence = 0.0f; // 0..1 (NSDF clarity)
    std::uint8_t flags = 0;  // HarmFlags
    std::uint8_t nValid = 0; // harmonics with n·f0 < 0.95·fs/2
    float h[kNumHarmonics];  // dBFS sine amplitude, [-100, 0]
    float hz[kNumHarmonics]; // measured peak frequency per harmonic
    float noise[kNumNoiseBands]; // dBFS, ERB bands of the residual
};

} // namespace larynx
