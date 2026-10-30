#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace {

juce::AudioProcessorValueTreeState::ParameterLayout createLayout() {
    using namespace juce;
    // IDs are an automation contract (§5): never rename, only append.
    auto id = [](const char* s) { return ParameterID { s, 1 }; };
    auto pct = [&](const char* s, const char* name, float def) {
        return std::make_unique<AudioParameterFloat>(id(s), name, NormalisableRange<float> { 0.0f, 100.0f, 0.1f }, def,
                                                     AudioParameterFloatAttributes().withLabel("%"));
    };
    auto range = [&](const char* s, const char* name, float lo, float hi, const char* unit) {
        return std::make_unique<AudioParameterFloat>(id(s), name, NormalisableRange<float> { lo, hi, 0.01f }, 0.0f,
                                                     AudioParameterFloatAttributes().withLabel(unit));
    };

    AudioProcessorValueTreeState::ParameterLayout l;
    // Mode changes latency/PDC: not automatable (RF10).
    l.add(std::make_unique<AudioParameterChoice>(id(ParamID::mode), "Mode", StringArray { "DSP", "Live", "Studio" }, 0,
                                                 AudioParameterChoiceAttributes().withAutomatable(false)));
    l.add(pct(ParamID::prior, "Prior", 70.0f));
    l.add(pct(ParamID::mix, "Mix", 100.0f));
    l.add(range(ParamID::bright, "Brightness", -12.0f, 12.0f, "dB/oct"));
    l.add(range(ParamID::air, "Air", -12.0f, 12.0f, "dB"));
    l.add(range(ParamID::evenodd, "Even/Odd", -1.0f, 1.0f, ""));
    l.add(range(ParamID::formant, "Formant", -6.0f, 6.0f, "st"));
    l.add(pct(ParamID::sub, "Sub", 0.0f));
    l.add(pct(ParamID::breath, "Breath", 0.0f));
    l.add(range(ParamID::in, "In Trim", -24.0f, 24.0f, "dB"));
    l.add(range(ParamID::out, "Out Trim", -24.0f, 24.0f, "dB"));
    l.add(std::make_unique<AudioParameterBool>(id(ParamID::bypass), "Bypass", false));
    return l;
}

} // namespace

LarynxProcessor::LarynxProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::mono(), true)
                         .withOutput("Output", juce::AudioChannelSet::mono(), true)),
      apvts(*this, nullptr, "LARYNX", createLayout()) {
    inDb = apvts.getRawParameterValue(ParamID::in);
    outDb = apvts.getRawParameterValue(ParamID::out);
    bypassOn = apvts.getRawParameterValue(ParamID::bypass);
}

bool LarynxProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    return layouts.getMainInputChannelSet() == juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() == juce::AudioChannelSet::mono();
}

void LarynxProcessor::prepareToPlay(double sampleRate, int maxBlockSize) {
    load.reset(sampleRate, maxBlockSize);
    const int latency = latencyFor(sampleRate);
    setLatencySamples(latency);
    delay.resize((size_t) latency);
    engine.prepare(sampleRate);
    jassert(engine.latency() == latency);

    inGain.reset(sampleRate, 0.03); // §5: 30 ms smoothing
    outGain.reset(sampleRate, 0.03);
    inGain.setCurrentAndTargetValue(juce::Decibels::decibelsToGain(inDb->load()));
    outGain.setCurrentAndTargetValue(juce::Decibels::decibelsToGain(outDb->load()));
    bypassFade.reset(sampleRate, 0.01); // NFR4: 10 ms bypass crossfade
    bypassFade.setCurrentAndTargetValue(bypassOn->load() > 0.5f ? 1.0f : 0.0f);
}

void LarynxProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) { process(buffer, false); }

void LarynxProcessor::processBlockBypassed(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) { process(buffer, true); }

void LarynxProcessor::process(juce::AudioBuffer<float>& buffer, bool forceBypass) {
    juce::ScopedNoDenormals noDenormals;
    juce::AudioProcessLoadMeasurer::ScopedTimer timer(load, buffer.getNumSamples());

    inGain.setTargetValue(juce::Decibels::decibelsToGain(inDb->load()));
    outGain.setTargetValue(juce::Decibels::decibelsToGain(outDb->load()));
    bypassFade.setTargetValue(forceBypass || bypassOn->load() > 0.5f ? 1.0f : 0.0f);

    auto* x = buffer.getWritePointer(0);
    float pIn = 0.0f, pOut = 0.0f;

    // Processed path: in trim → engine (latency = window) → out trim. Bypass is the input through a
    // delay of the same length, so toggling never moves audio in time (NFR4). The engine keeps
    // running while bypassed, so un-bypassing needs no warm-up.
    // ponytail: Live/Studio run the DSP path until the neural core exists (P6/P7).
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        pIn = std::max(pIn, std::abs(x[i]));
        const float d = delay.process(x[i]);
        const bool fading = bypassFade.isSmoothing();
        const float b = bypassFade.getNextValue();
        const float wet = engine.process(x[i] * inGain.getNextValue()) * outGain.getNextValue();
        // Outside the fade, copy exactly: d + (wet - d) * 0 is not guaranteed bit-exact.
        x[i] = fading ? wet + (d - wet) * b : (b > 0.5f ? d : wet);
        pOut = std::max(pOut, std::abs(x[i]));
    }

    for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
        buffer.clear(ch, 0, buffer.getNumSamples());

    // UI takes these with exchange(0); keep the max until it does.
    inPeak.store(std::max(inPeak.load(), pIn));
    outPeak.store(std::max(outPeak.load(), pOut));
}

juce::AudioProcessorParameter* LarynxProcessor::getBypassParameter() const {
    return apvts.getParameter(ParamID::bypass);
}

juce::AudioProcessorEditor* LarynxProcessor::createEditor() { return new LarynxEditor(*this); }

void LarynxProcessor::getStateInformation(juce::MemoryBlock& dest) {
    if (auto xml = apvts.copyState().createXml())
        copyXmlToBinary(*xml, dest);
}

void LarynxProcessor::setStateInformation(const void* data, int size) {
    if (auto xml = getXmlFromBinary(data, size); xml && xml->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new LarynxProcessor(); }
