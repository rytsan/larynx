#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <array>
#include "core/blend.h"
#include "core/delay_line.h"
#include "core/engine.h"
#include "core/morph.h"
#include "core/neuralcore.h"

namespace ParamID {
inline constexpr auto mode = "mode", prior = "prior", mix = "mix", bright = "bright",
    air = "air", evenodd = "evenodd", formant = "formant", sub = "sub", dbl = "double", breath = "breath",
    in = "in", out = "out", bypass = "bypass", morph = "morph", morphVoice = "morph_voice";
}

class LarynxProcessor : public juce::AudioProcessor,
                        private juce::AudioProcessorValueTreeState::Listener,
                        private juce::AsyncUpdater,
                        private juce::ValueTree::Listener {
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
    void flushModeSwitch() { handleUpdateNowIfNeeded(); } // offline renders/tests: run the pending reconfigure now
    // §2.7: only 44.1/48 kHz are processed; other rates pass the input through the latency delay.
    bool rateSupported() const { return supported.load(); }

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

    // Factory presets (§12); selecting one sets the parameters, the editor curve is left alone.
    int getNumPrograms() override;
    int getCurrentProgram() override { return program; }
    void setCurrentProgram(int) override;
    const juce::String getProgramName(int) override;
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;

    // Morph profiles (§12), embedded profiles.data from ml/data/build_morph.py; empty without it.
    static const larynx::MorphProfiles& profiles();

    juce::UndoManager undo;
    juce::AudioProcessorValueTreeState apvts;

    // Envelope editor (§12): per-harmonic dB offsets H1..H40 drawn on the graph, applied with the manual
    // operators. Stored in the state tree (property "editor", undoable); mirrored for the audio thread.
    static constexpr int kEditBars = 40;
    void setEditorCurve(const std::array<float, kEditBars>& db, bool undoable = true);
    std::array<float, kEditBars> editorCurve() const;

    // Meters for the UI (written by audio thread, read by UI timer).
    std::atomic<float> inPeak { 0.0f }, outPeak { 0.0f };
    std::atomic<float> inSq { 0.0f }, outSq { 0.0f }, sqCount { 0.0f }; // sums of squares for the VU (rms)
    juce::AudioProcessLoadMeasurer load;

    // Latest analysed/shaped frame for the graph: audio thread fills slot[1 - published] and flips.
    // ponytail: a reader slower than one hop (5 ms) can see a torn frame — harmless for a display.
    struct GraphFrame {
        std::array<float, 40> in {}, out {}, net {}; // net: input + Prior·network delta (floor = none)
        float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f;
        char vowel = 0;
    };
    GraphFrame graphSlot[2];
    std::atomic<int> graphPublished { 0 };

private:
    void process(juce::AudioBuffer<float>&, bool forceBypass);
    void configure(double sampleRate, bool switching = false); // latency, delays, engines, network state for the current mode
    void parameterChanged(const juce::String& id, float) override;
    void handleAsyncUpdate() override;
    void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier& id) override {
        if (id == juce::Identifier("editor")) syncEditorFromState(); // undo/redo land here
    }

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
    std::atomic<float>* doublePct = nullptr;
    std::atomic<float>* breathPct = nullptr;
    std::atomic<float>* morphPct = nullptr;
    std::atomic<float>* morphVoiceIdx = nullptr;
    juce::SmoothedValue<float> sMorph;

    larynx::DelayLine delay;
    larynx::Engine engine; // analyzer → skeleton → blend (M4) → synth
    // M4 controls, smoothed per sample and read once per hop (§8: automation without zipper).
    juce::SmoothedValue<float> sPrior, sMix, sBright, sAir, sEvenOdd, sFormant, sSub, sDouble, sBreath;
    double fs = 0.0;
    int program = 0;
    std::array<std::atomic<float>, kEditBars> editDb {};
    void syncEditorFromState();

    // M3 (§10/§11). Live: network on the current frame. Studio: the network sees frame t and predicts
    // t-D, so a second engine runs D hops behind on delayed input and resynthesises with that output.
    larynx::NeuralCore nets[2]; // [Live, Studio]
    larynx::Engine studioEngine;
    larynx::DelayLine studioIn;
    std::atomic<int> runMode { DSP };
    std::atomic<bool> supported { true };
    juce::SmoothedValue<float> netRamp; // 0→1 over 300 ms after a mode switch (RF warm-up)
    // Mode switch without pops: the audio thread fades the wet path out (10 ms), the message thread
    // reconfigures, then the wet path stays muted for the new latency and fades back in (10 ms).
    juce::SmoothedValue<float> switchGain { 1.0f };
    std::atomic<bool> switchPending { false }, switchReady { false };
    int switchMute = 0;
    std::array<float, larynx::kNetInputs> netX {};
    std::array<float, larynx::kNetOutputs> netY {};
    std::array<float, larynx::kNetOutputs> netDelta {};
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Multiplicative> inGain { 1.0f }, outGain { 1.0f };
    juce::SmoothedValue<float> bypassFade { 0.0f }; // 0 = processed, 1 = bypassed

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LarynxProcessor)
};
