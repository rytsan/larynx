#pragma once
// Per-sample wiring of M1 → M2 → (shape) → M6 (§2.1). `shape(in, target)` is called once per hop
// with target pre-filled as a copy of in; leaving it untouched gives the identity path.
// Latency = STFT window (1024 @48k). No JUCE (core/ rule). Allocates only in prepare().
#include <algorithm>
#include <vector>
#include "analyzer.h"
#include "skeleton.h"
#include "synth.h"

namespace larynx {

class Engine {
public:
    void prepare(double sampleRate) {
        analyzer.prepare(sampleRate);
        skeleton.prepare(sampleRate);
        synth.prepare(skeleton, sampleRate);
        ring_.assign((size_t) skeleton.win(), 0.0f);
        lin_.assign(ring_.size(), 0.0f);
        pos_ = 0;
    }
    void reset() {
        analyzer.reset();
        synth.reset();
        std::fill(ring_.begin(), ring_.end(), 0.0f);
        pos_ = 0;
    }
    int latency() const { return skeleton.win(); }
    const HarmFrame& frame() const { return in_; }

    template <class Shape>
    float process(float x, Shape&& shape) {
        const float y = synth.next();
        ring_[(size_t) pos_] = x;
        if (++pos_ == (int) ring_.size()) pos_ = 0;
        // Analyzer and skeleton share the hop, and their windows end on this same sample (§2.3).
        if (analyzer.push(x, in_)) {
            std::copy(ring_.begin() + pos_, ring_.end(), lin_.begin());
            std::copy(ring_.begin(), ring_.begin() + pos_, lin_.end() - pos_);
            skeleton.analyze(lin_.data(), in_);
            target_ = in_;
            shape(static_cast<const HarmFrame&>(in_), target_);
            synth.render(skeleton, in_, target_);
        }
        return y;
    }
    float process(float x) { return process(x, [](const HarmFrame&, HarmFrame&) {}); }

    Analyzer analyzer;
    Skeleton skeleton;
    Synth synth;

private:
    std::vector<float> ring_, lin_;
    int pos_ = 0;
    HarmFrame in_ {}, target_ {};
};

} // namespace larynx
