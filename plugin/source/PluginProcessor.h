#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <array>
#include "core/blend.h"
#include "core/delay_line.h"
#include "core/engine.h"
#include "core/neuralcore.h"

namespace ParamID {
inline constexpr auto mode = "mode", prior = "prior", mix = "mix", bright = "bright",
    air = "air", evenodd = "evenodd", formant = "formant", sub = "sub", breath = "breath",
    in = "in", out = "out", bypass = "bypass";
}

class LarynxProcessor : public juce::AudioProcessor,
                        private juce::AudioProcessorValueTreeState::Listener,
                        private juce::AsyncUpdater {
public:
    LarynxProcessor();

    ~LarynxProcessor() override;

    // Algorithmic latency (§2.5): 1024 samples @48k, same 21.3 ms at any rate (§2.7). Studio adds D hops.
    static int latencyFor(double sampleRate) { return juce::roundToInt(1024.0 * sampleRate / 48000.0); }

    enum Mode { DSP = 0, Live = 1, Studio = 2 };
    // Network weights from exported JSON (ml/train/model.py). The plugin loads the embedded
    // model_live/model_studio.json at construction; a mode without weights runs the DSP path.
    static bool parseModel(const juce::String& json, larynx::TcnWeights& out);
    bool loadModel(Mode m, const larynx::TcnWeights& w); // call before prepareToPlay (or the mode switch)
    bool hasModel(Mode m) const { return m != DSP && nets[m - 1].loaded(); }
    Mode runningMode() const { return (Mode) runMode.load(); } // effective mode (DSP when weights missing)

    void prepareToPlay(double sampleRate, int maxBlockSize) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout&) const override;
    using AudioProcessor::processBlock;
    using AudioProcessor::processBlockBypassed;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    void processBlockBypassed(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    juce::AudioProcessorParameter* getBypassParameter() const override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "LARYNX"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;

    juce::AudioProcessorValueTreeState apvts;

    // Meters for the UI (written by audio thread, read by UI timer).
    std::atomic<float> inPeak { 0.0f }, outPeak { 0.0f };
    std::atomic<float> inSq { 0.0f }, outSq { 0.0f }, sqCount { 0.0f }; // sums of squares for the VU (rms)
    juce::AudioProcessLoadMeasurer load;

    // Latest analysed/shaped frame for the graph: audio thread fills slot[1 - published] and flips.
    // ponytail: a reader slower than one hop (5 ms) can see a torn frame — harmless for a display.
    struct GraphFrame {
        std::array<float, 40> in {}, out {};
        float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f;
        char vowel = 0;
    };
    GraphFrame graphSlot[2];
    std::atomic<int> graphPublished { 0 };

private:
    void process(juce::AudioBuffer<float>&, bool forceBypass);
    void configure(double sampleRate); // latency, delays, engines, network state for the current mode
    void parameterChanged(const juce::String& id, float) override;
    void handleAsyncUpdate() override;

    std::atomic<float>* inDb = nullptr;
    std::atomic<float>* outDb = nullptr;
    std::atomic<float>* bypassOn = nullptr;
    std::atomic<float>* priorPct = nullptr;
    std::atomic<float>* mixPct = nullptr;
    std::atomic<float>* brightDb = nullptr;
    std::atomic<float>* airDb = nullptr;
    std::atomic<float>* evenOdd = nullptr;
    std::atomic<float>* formantSt = nullptr;
    std::atomic<float>* subPct = nullptr;
    std::atomic<float>* breathPct = nullptr;

    larynx::DelayLine delay;
    larynx::Engine engine; // analyzer → skeleton → blend (M4) → synth
    // M4 controls, smoothed per sample and read once per hop (§8: automation without zipper).
    juce::SmoothedValue<float> sPrior, sMix, sBright, sAir, sEvenOdd, sFormant, sSub, sBreath;
    double fs = 0.0;

    // M3 (§10/§11). Live: network on the current frame. Studio: the network sees frame t and predicts
    // t-D, so a second engine runs D hops behind on delayed input and resynthesises with that output.
    larynx::NeuralCore nets[2]; // [Live, Studio]
    larynx::Engine studioEngine;
    larynx::DelayLine studioIn;
    std::atomic<int> runMode { DSP };
    juce::SmoothedValue<float> netRamp; // 0→1 over 300 ms after a mode switch (RF warm-up)
    std::array<float, larynx::kNetInputs> netX {};
    std::array<float, larynx::kNetOutputs> netY {};
    std::array<float, larynx::kNetOutputs> netDelta {};
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Multiplicative> inGain { 1.0f }, outGain { 1.0f };
    juce::SmoothedValue<float> bypassFade { 0.0f }; // 0 = processed, 1 = bypassed

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LarynxProcessor)
};
