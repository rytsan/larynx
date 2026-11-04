#pragma once
// M6: restoration resynthesis (§7, §2.3). Gain mask on the original STFT (original phase) for existing
// harmonics and ERB noise bands, ISTFT/OLA; additive oscillators (phase 2πn∫f0dt) only for absent harmonics.
// All gains = 1 → identity, out[n] = in[n − win]. No JUCE (core/ rule). Allocates only in prepare().
#include <array>
#include <vector>
#include "delay_line.h"
#include "skeleton.h"

namespace larynx {

class Synth {
public:
    void prepare(const Skeleton& sk, double sampleRate);
    void reset();

    // Once per hop, right after sk.analyze(): reshapes sk.spectrum() from `in` towards `target`
    // (in place), overlap-adds the frame and queues the next hop of additive output.
    void render(Skeleton& sk, const HarmFrame& in, const HarmFrame& target);

    // M5 Sub (§12): f0/2 and f0/4 oscillators at gain × the mean amplitude of H1..H4 (f0/4 at half that),
    // phase-locked to the harmonic bank. 0 = off. Read once per render().
    void setSub(float gain) { subGain_ = gain; }

    // "Duplo virtual" (§12, light phase detune): a second copy of H1..kDoubleHarmonics at +kDoubleCents,
    // gain × the target amplitudes, on its own phase — it beats slowly against the voice like a second
    // take. 0 = off. Read once per render().
    // ponytail: fixed detune and no timing offset; add a slow detune LFO if it sounds static.
    static constexpr int kDoubleHarmonics = 16;
    static constexpr double kDoubleCents = 9.0;
    void setDouble(float gain) { dblGain_ = gain; }

    // One output sample. Called once per input sample, before that sample's render().
    float next() {
        const float y = out_[(size_t) readPos_] + addDelay_.process(add_[(size_t) readPos_]);
        ++readPos_;
        return y;
    }

private:
    double fs_ = 48000.0;
    int win_ = 1024, hop_ = 256, readPos_ = 0;
    float mixStep_ = 0.0f;
    std::vector<float> gain_, ola_, norm_, out_, add_;
    std::array<bool, kNumHarmonics> present_ {};
    std::array<float, kNumHarmonics> mix_ {};     // 1 = mask path, 0 = additive path (20 ms crossfade)
    std::array<float, kNumHarmonics> ampPrev_ {}, ampCur_ {};
    float f0Prev_ = 0.0f, f0Cur_ = 0.0f;
    float subGain_ = 0.0f, subPrev_ = 0.0f, subCur_ = 0.0f;
    double phase_ = 0.0, phaseDbl_ = 0.0;
    float dblGain_ = 0.0f;
    std::array<float, kDoubleHarmonics> dblPrev_ {}, dblCur_ {};
    DelayLine addDelay_; // aligns additive (frame-centre) output with the OLA output
};

} // namespace larynx
