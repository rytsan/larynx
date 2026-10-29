#pragma once
// M1: causal F0 (NSDF/MPM), voicing with hysteresis, sibilant/transient flags (§6, §2.3).
// No JUCE (core/ rule). Allocates only in prepare(); push() is real-time safe.
#include <array>
#include <vector>
#include "fft.h"
#include "harm_frame.h"

namespace larynx {

class Analyzer {
public:
    // Calibration knobs. Defaults from the plan; flag thresholds still need tuning on the golden set.
    struct Tuning {
        float fMin = 60.0f, fMax = 1200.0f;     // Hz search range
        float voiceOn = 0.75f, voiceOff = 0.60f; // clarity hysteresis
        float peakK = 0.9f;                     // MPM: first key max >= k * highest
        float floorDb = -70.0f;                 // window RMS below this = silence
        float flatness = 0.3f;                  // sibilant: spectral flatness 2-16 kHz
        float fluxDb = 9.0f;                    // transient: rise of high-passed hop energy
        float centroidJump = 1.5f;              // transient: RMS frequency vs running average
    } tuning;

    void prepare(double sampleRate);
    void reset();

    // Feeds one sample. Every hop, fills f0/confidence/flags of `f` and returns true.
    // The frame refers to the window ending at this sample.
    bool push(float x, HarmFrame& f);

    int hop() const { return hop_; }
    int window() const { return win_; }

private:
    struct Key { float lag, val; };
    void analyze(HarmFrame& f);
    int pickKeys(); // fills keys_ from nsdf_, returns count

    double fs_ = 48000.0;
    int hop_ = 256, win_ = 2048, specWin_ = 1024, minLag_ = 0, maxLag_ = 0;
    float floorPow_ = 0.0f, centroidCoef_ = 0.0f;

    Fft acFft_, specFft_;
    std::vector<float> ring_, lin_, nsdf_, hann_;
    std::vector<Fft::cf> acBuf_, specBuf_;
    int pos_ = 0, count_ = 0;

    static constexpr int kMaxKeys = 32;
    std::array<Key, kMaxKeys> keys_ {};

    bool voiced_ = false;
    std::array<float, 5> hist_ {}; // raw F0 of recent voiced frames (octave guard only)
    int histLen_ = 0, histPos_ = 0;

    float prevX_ = 0.0f, eHop_ = 0.0f, edHop_ = 0.0f;
    std::array<float, 8> edHist_ {};
    int edPos_ = 0, transientHold_ = 0;
    float centroidAvg_ = 1000.0f;
};

} // namespace larynx
