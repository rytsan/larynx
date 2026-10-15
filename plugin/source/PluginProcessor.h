#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "core/delay_line.h"

namespace ParamID {
inline constexpr auto mode = "mode", prior = "prior", mix = "mix", bright = "bright",
    air = "air", evenodd = "evenodd", formant = "formant", sub = "sub", breath = "breath",
    in = "in", out = "out", bypass = "bypass";
}

class LarynxProcessor : public juce::AudioProcessor {
public:
    LarynxProcessor();

    // Algorithmic latency (§2.5): 1024 samples @48k, same 21.3 ms at any rate (§2.7).
    static int latencyFor(double sampleRate) { return juce::roundToInt(1024.0 * sampleRate / 48000.0); }

    void prepareToPlay(double sampleRate, int maxBlockSize) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout&) const override;
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
    juce::AudioProcessLoadMeasurer load;

private:
    void process(juce::AudioBuffer<float>&, bool forceBypass);

    std::atomic<float>* inDb = nullptr;
    std::atomic<float>* outDb = nullptr;
    std::atomic<float>* bypassOn = nullptr;

    larynx::DelayLine delay;
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Multiplicative> gain { 1.0f };
    juce::SmoothedValue<float> bypassFade { 0.0f }; // 0 = processed, 1 = bypassed

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LarynxProcessor)
};
