#pragma once
// M2: STFT analysis (√Hann, window 1024 / hop 256 @48k) and harmonic extraction into HarmFrame (§7, §2.3).
// No JUCE (core/ rule). Allocates only in prepare().
#include <array>
#include <vector>
#include "fft.h"
#include "harm_frame.h"

namespace larynx {

class Skeleton {
public:
    void prepare(double sampleRate);

    // `window`: the last win() samples, oldest first, ending at the same sample as the analyzer's
    // window. Uses f.f0 from the analyzer; fills f.nValid/h/hz/noise and spectrum().
    void analyze(const float* window, HarmFrame& f);

    std::vector<Fft::cf>& spectrum() { return X_; } // full complex spectrum (fftSize bins)
    int peakBin(int i) const { return peakBin_[(size_t) i]; } // harmonic i (0-based), -1 if not extracted
    int bandOf(int bin) const { return band_[(size_t) bin]; }
    const std::vector<float>& window() const { return w_; }
    const Fft& fft() const { return fft_; }
    int win() const { return win_; }
    int hop() const { return hop_; }
    int fftSize() const { return (int) fft_.size(); }

    // Real amplitude response of the sine (√Hann) window, 1 at δ = 0, δ in window bins.
    static float signedLobe(float delta);

private:
    double fs_ = 48000.0;
    int win_ = 1024, hop_ = 256;
    float sumW_ = 0.0f, sumW2_ = 0.0f;
    Fft fft_;
    std::vector<float> w_, pow_;
    std::vector<Fft::cf> X_, res_;
    std::vector<int> band_;
    std::array<int, kNumHarmonics> peakBin_ {};
};

} // namespace larynx
