#include "PluginProcessor.h"
#include "PluginEditor.h"
#if LARYNX_MODELS
#include <BinaryData.h>
#endif

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
    // Appended in P8 (IDs are a contract: only append).
    l.add(pct(ParamID::morph, "Morph", 0.0f));
    StringArray voices;
    for (int i = 0; i < LarynxProcessor::profiles().singers(); ++i) voices.add(LarynxProcessor::profiles().name(i));
    if (voices.isEmpty()) voices.add("-");
    l.add(std::make_unique<AudioParameterChoice>(id(ParamID::morphVoice), "Morph Voice", voices, 0));
    return l;
}

} // namespace

LarynxProcessor::LarynxProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::mono(), true)
                         .withOutput("Output", juce::AudioChannelSet::mono(), true)),
      apvts(*this, &undo, "LARYNX", createLayout()) {
    inDb = apvts.getRawParameterValue(ParamID::in);
    outDb = apvts.getRawParameterValue(ParamID::out);
    bypassOn = apvts.getRawParameterValue(ParamID::bypass);
    priorPct = apvts.getRawParameterValue(ParamID::prior);
    mixPct = apvts.getRawParameterValue(ParamID::mix);
    brightDb = apvts.getRawParameterValue(ParamID::bright);
    airDb = apvts.getRawParameterValue(ParamID::air);
    evenOdd = apvts.getRawParameterValue(ParamID::evenodd);
    formantSt = apvts.getRawParameterValue(ParamID::formant);
    subPct = apvts.getRawParameterValue(ParamID::sub);
    breathPct = apvts.getRawParameterValue(ParamID::breath);
    morphPct = apvts.getRawParameterValue(ParamID::morph);
    morphVoiceIdx = apvts.getRawParameterValue(ParamID::morphVoice);

#if LARYNX_MODELS
    for (auto [m, name] : { std::pair { Live, "model_live_json" }, { Studio, "model_studio_json" } }) {
        int size = 0;
        larynx::TcnWeights w;
        if (const char* data = BinaryData::getNamedResource(name, size);
            data && parseModel(juce::String::fromUTF8(data, size), w))
            loadModel(m, w);
    }
#endif
    apvts.addParameterListener(ParamID::mode, this);
    apvts.state.addListener(this);
}

LarynxProcessor::~LarynxProcessor() {
    apvts.state.removeListener(this);
    apvts.removeParameterListener(ParamID::mode, this);
    cancelPendingUpdate();
}

namespace {

struct Preset {
    const char* name;
    std::initializer_list<std::pair<const char*, float>> values; // unlisted parameters go to default
};

const Preset presets[] = {
    { "Pass Through", { { ParamID::prior, 0.0f } } },
    { "Studio Restore", { { ParamID::mode, 2.0f }, { ParamID::prior, 80.0f } } },
    { "Telefone → Estúdio", { { ParamID::mode, 1.0f }, { ParamID::prior, 100.0f }, { ParamID::air, 4.0f }, { ParamID::bright, 1.5f } } },
    { "Growl suave", { { ParamID::evenodd, -0.4f }, { ParamID::sub, 25.0f }, { ParamID::bright, -1.0f } } },
    { "Sopro etéreo", { { ParamID::prior, 50.0f }, { ParamID::breath, 70.0f }, { ParamID::air, 6.0f }, { ParamID::bright, 1.0f } } },
    // ponytail: no phase detune yet ("detune leve de fase" of §12): sub + slight parity/formant shift only.
    { "Duplo virtual", { { ParamID::sub, 15.0f }, { ParamID::evenodd, 0.15f }, { ParamID::formant, 0.5f } } },
};

} // namespace

const larynx::MorphProfiles& LarynxProcessor::profiles() {
    static const larynx::MorphProfiles p = [] {
        larynx::MorphProfiles m;
#if LARYNX_MODELS
        int size = 0;
        if (const char* data = BinaryData::getNamedResource("profiles_data", size)) m.load(data, (size_t) size);
#endif
        return m;
    }();
    return p;
}

int LarynxProcessor::getNumPrograms() { return (int) std::size(presets); }

const juce::String LarynxProcessor::getProgramName(int i) {
    return juce::isPositiveAndBelow(i, getNumPrograms()) ? juce::String::fromUTF8(presets[i].name) : juce::String();
}

void LarynxProcessor::setCurrentProgram(int i) {
    if (!juce::isPositiveAndBelow(i, getNumPrograms())) return;
    program = i;
    for (auto* p : getParameters())
        if (auto* r = dynamic_cast<juce::RangedAudioParameter*>(p); r && r->getParameterID() != ParamID::bypass) {
            float v = r->convertFrom0to1(r->getDefaultValue());
            for (auto& [id, value] : presets[i].values)
                if (r->getParameterID() == id) v = value;
            r->setValueNotifyingHost(r->convertTo0to1(v));
        }
}

void LarynxProcessor::setEditorCurve(const std::array<float, kEditBars>& db, bool undoable) {
    juce::StringArray parts;
    for (float v : db) parts.add(juce::String(v, 2));
    apvts.state.setProperty("editor", parts.joinIntoString(" "), undoable ? &undo : nullptr);
    syncEditorFromState();
}

std::array<float, LarynxProcessor::kEditBars> LarynxProcessor::editorCurve() const {
    std::array<float, kEditBars> out {};
    for (size_t i = 0; i < out.size(); ++i) out[i] = editDb[i].load();
    return out;
}

void LarynxProcessor::syncEditorFromState() {
    const auto parts = juce::StringArray::fromTokens(apvts.state.getProperty("editor").toString(), " ", "");
    for (int i = 0; i < kEditBars; ++i) editDb[(size_t) i] = i < parts.size() ? parts[i].getFloatValue() : 0.0f;
}

bool LarynxProcessor::parseModel(const juce::String& json, larynx::TcnWeights& w) {
    const auto v = juce::JSON::parse(json);
    if (!v.isObject() || (int) v["contract"] != larynx::kNetContract || (int) v["inputs"] != larynx::kNetInputs
        || (int) v["outputs"] != larynx::kNetOutputs)
        return false; // other I/O contract: refuse, DSP fallback (§9.6)
    std::function<void(const juce::var&, std::vector<float>&)> flat = [&](const juce::var& a, std::vector<float>& out) {
        if (auto* arr = a.getArray())
            for (auto& e : *arr) flat(e, out);
        else
            out.push_back((float) (double) a);
    };
    w = {};
    w.channels = v["n_channels"];
    w.kernel = v["kernel"];
    w.delay = v["D"];
    if (auto* d = v["dilations"].getArray())
        for (auto& e : *d) w.dilations.push_back((int) e);
    if (auto* layers = v["layers"].getArray())
        for (auto& l : *layers) {
            std::vector<float> W, B;
            flat(l["w"], W);
            flat(l["b"], B);
            const auto name = l["name"].toString();
            if (name == "in") w.inW = std::move(W), w.inB = std::move(B);
            else if (name == "out") w.outW = std::move(W), w.outB = std::move(B);
            else w.convW.push_back(std::move(W)), w.convB.push_back(std::move(B));
        }
    return larynx::NeuralCore().load(w);
}

bool LarynxProcessor::loadModel(Mode m, const larynx::TcnWeights& w) {
    jassert(m != DSP);
    return nets[m - 1].load(w);
}

// The audio thread fades out and sets switchReady, which triggers the reconfigure below.
// ponytail: the switch waits for processBlock calls; a host that stops calling it before the fade ends
// applies the new mode at the next prepareToPlay instead.
void LarynxProcessor::parameterChanged(const juce::String&, float) { switchPending = true; }

void LarynxProcessor::handleAsyncUpdate() {
    if (fs <= 0.0 || !switchReady) return; // not prepared yet: prepareToPlay picks the mode up
    // Mode changes latency (PDC): reconfigure with the audio callback held, then tell the host
    // (hosts only re-apply PDC on the next stop/start, §2.6).
    suspendProcessing(true);
    configure(fs, true);
    suspendProcessing(false);
    updateHostDisplay(ChangeDetails().withLatencyChanged(true));
}

void LarynxProcessor::configure(double sampleRate, bool switching) {
    fs = sampleRate;
    supported = sampleRate == 44100.0 || sampleRate == 48000.0;
    const int m = juce::roundToInt(apvts.getRawParameterValue(ParamID::mode)->load());
    runMode = m == DSP || !nets[m - 1].loaded() ? DSP : m;
    engine.prepare(sampleRate);
    studioEngine.prepare(sampleRate);
    const int extra = runMode == Studio ? nets[1].delay() * engine.skeleton.hop() : 0;
    studioIn.resize((size_t) extra);
    const int latency = latencyFor(sampleRate) + extra;
    setLatencySamples(latency);
    delay.resize((size_t) latency);
    jassert(engine.latency() == latencyFor(sampleRate));
    for (auto& n : nets)
        if (n.loaded()) n.reset();
    netRamp.reset(sampleRate, 0.3);
    netRamp.setCurrentAndTargetValue(0.0f);
    netRamp.setTargetValue(1.0f);
    switchPending = switchReady = false;
    switchGain.reset(sampleRate, 0.01);
    switchGain.setCurrentAndTargetValue(switching ? 0.0f : 1.0f);
    switchMute = switching ? latency : 0;
}

bool LarynxProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    return layouts.getMainInputChannelSet() == juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() == juce::AudioChannelSet::mono();
}

void LarynxProcessor::prepareToPlay(double sampleRate, int maxBlockSize) {
    load.reset(sampleRate, maxBlockSize);
    configure(sampleRate);

    inGain.reset(sampleRate, 0.03); // §5: 30 ms smoothing
    outGain.reset(sampleRate, 0.03);
    inGain.setCurrentAndTargetValue(juce::Decibels::decibelsToGain(inDb->load()));
    outGain.setCurrentAndTargetValue(juce::Decibels::decibelsToGain(outDb->load()));
    bypassFade.reset(sampleRate, 0.01); // NFR4: 10 ms bypass crossfade
    bypassFade.setCurrentAndTargetValue(bypassOn->load() > 0.5f ? 1.0f : 0.0f);

    for (auto [sv, p, scale] : { std::tuple { &sPrior, priorPct, 0.01f }, { &sMix, mixPct, 0.01f }, { &sBright, brightDb, 1.0f },
                                 { &sAir, airDb, 1.0f }, { &sEvenOdd, evenOdd, 1.0f }, { &sFormant, formantSt, 1.0f },
                                 { &sSub, subPct, 0.01f }, { &sBreath, breathPct, 0.01f }, { &sMorph, morphPct, 0.01f } }) {
        sv->reset(sampleRate, 0.03);
        sv->setCurrentAndTargetValue(p->load() * scale);
    }
}

void LarynxProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) { process(buffer, false); }

void LarynxProcessor::processBlockBypassed(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) { process(buffer, true); }

void LarynxProcessor::process(juce::AudioBuffer<float>& buffer, bool forceBypass) {
    juce::ScopedNoDenormals noDenormals;
    juce::AudioProcessLoadMeasurer::ScopedTimer timer(load, buffer.getNumSamples());

    inGain.setTargetValue(juce::Decibels::decibelsToGain(inDb->load()));
    outGain.setTargetValue(juce::Decibels::decibelsToGain(outDb->load()));
    bypassFade.setTargetValue(forceBypass || bypassOn->load() > 0.5f ? 1.0f : 0.0f);
    sPrior.setTargetValue(priorPct->load() * 0.01f);
    sMix.setTargetValue(mixPct->load() * 0.01f);
    sBright.setTargetValue(brightDb->load());
    sAir.setTargetValue(airDb->load());
    sEvenOdd.setTargetValue(evenOdd->load());
    sFormant.setTargetValue(formantSt->load());
    sSub.setTargetValue(subPct->load() * 0.01f);
    sBreath.setTargetValue(breathPct->load() * 0.01f);
    sMorph.setTargetValue(morphPct->load() * 0.01f);
    const int voice = juce::roundToInt(morphVoiceIdx->load());
    float morphAmt = 0.0f;
    // ponytail: morph acts after the M4 blend (so Mix does not scale it); it has its own amount knob.
    const auto& prof = profiles();

    larynx::Controls c;
    std::array<float, kEditBars> user;
    for (size_t i = 0; i < user.size(); ++i) user[i] = editDb[i].load(std::memory_order_relaxed);
    const int mode = runMode.load();
    auto publish = [&](const larynx::HarmFrame& in, const larynx::HarmFrame& t, const float* net = nullptr, float prior = 0.0f) {
        auto& g = graphSlot[1 - graphPublished.load(std::memory_order_relaxed)];
        std::copy_n(in.h, g.in.size(), g.in.begin());
        for (size_t i = 0; i < g.net.size(); ++i) // ghost: what the network alone asks for
            g.net[i] = net && (int) i < in.nValid ? std::clamp(in.h[i] + prior * net[i], larynx::kFloorDb, 0.0f) : larynx::kFloorDb;
        std::copy_n(t.h, g.out.size(), g.out.begin());
        g.f0 = (in.flags & larynx::kVoiced) ? in.f0 : 0.0f;
        g.vowel = larynx::vowelOf(t, &g.f1, &g.f2);
        graphPublished.store(1 - graphPublished.load(std::memory_order_relaxed), std::memory_order_release);
    };
    // Main engine: DSP and Live resynthesise here; in Studio it only feeds the network.
    auto shape = [&](const larynx::HarmFrame& in, larynx::HarmFrame& t) {
        if (mode == DSP) {
            larynx::blend(in, t, c, fs);
            prof.apply(t, voice, morphAmt);
            publish(in, t);
            return;
        }
        larynx::netInput(in, netX.data());
        nets[mode - 1].step(netX.data(), netY.data());
        if (mode == Live) {
            larynx::NeuralCore::toDeltas(in, netY.data(), netDelta.data());
            larynx::blend(in, t, c, fs, netDelta.data());
            prof.apply(t, voice, morphAmt);
            publish(in, t, netDelta.data(), c.prior);
        }
    };
    // Studio engine: input delayed by D hops, so its frame now is the frame netY describes.
    auto shapeStudio = [&](const larynx::HarmFrame& in, larynx::HarmFrame& t) {
        larynx::NeuralCore::toDeltas(in, netY.data(), netDelta.data());
        larynx::blend(in, t, c, fs, netDelta.data());
        prof.apply(t, voice, morphAmt);
        publish(in, t, netDelta.data(), c.prior);
    };

    auto* x = buffer.getWritePointer(0);
    float pIn = 0.0f, pOut = 0.0f, sIn = 0.0f, sOut = 0.0f;
    auto switchFade = [&] {
        if (switchPending && !switchReady) {
            switchGain.setTargetValue(0.0f);
            if (switchGain.getCurrentValue() <= 0.0f) {
                switchReady = true;
                triggerAsyncUpdate();
            }
        }
        if (switchMute > 0) {
            if (--switchMute == 0) switchGain.setTargetValue(1.0f);
            return 0.0f;
        }
        return switchGain.getNextValue();
    };

    // Processed path: in trim → engine (latency = window) → out trim. Bypass is the input through a
    // delay of the same length, so toggling never moves audio in time (NFR4). The engine keeps
    // running while bypassed, so un-bypassing needs no warm-up.
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        pIn = std::max(pIn, std::abs(x[i]));
        sIn += x[i] * x[i];
        const float d = delay.process(x[i]);
        const bool fading = bypassFade.isSmoothing();
        const float b = bypassFade.getNextValue();
        c = { sPrior.getNextValue() * netRamp.getNextValue(), sMix.getNextValue(), sBright.getNextValue(),
              sAir.getNextValue(), sEvenOdd.getNextValue(), sFormant.getNextValue(), sSub.getNextValue(),
              sBreath.getNextValue(), user.data() };
        morphAmt = sMorph.getNextValue();
        engine.synth.setSub(c.sub);
        studioEngine.synth.setSub(c.sub);
        const float xin = x[i] * inGain.getNextValue();
        if (!supported) { // NFR4 path at any rate: delayed input, no crash, no DSP (UI warns)
            switchFade();
            x[i] = d;
            outGain.getNextValue();
            pOut = std::max(pOut, std::abs(x[i]));
            sOut += x[i] * x[i];
            continue;
        }
        float wet = engine.process(xin, shape);
        if (mode == Studio) wet = studioEngine.process(studioIn.process(xin), shapeStudio);
        wet *= outGain.getNextValue() * switchFade();
        // Outside the fade, copy exactly: d + (wet - d) * 0 is not guaranteed bit-exact.
        x[i] = fading ? wet + (d - wet) * b : (b > 0.5f ? d : wet);
        pOut = std::max(pOut, std::abs(x[i]));
        sOut += x[i] * x[i];
    }

    for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
        buffer.clear(ch, 0, buffer.getNumSamples());

    // UI takes these with exchange(0); keep the max until it does.
    inPeak.store(std::max(inPeak.load(), pIn));
    outPeak.store(std::max(outPeak.load(), pOut));
    inSq.store(inSq.load() + sIn);
    outSq.store(outSq.load() + sOut);
    sqCount.store(sqCount.load() + (float) buffer.getNumSamples());
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
    if (auto xml = getXmlFromBinary(data, size); xml && xml->hasTagName(apvts.state.getType())) {
        apvts.state.removeListener(this);
        apvts.replaceState(juce::ValueTree::fromXml(*xml));
        apvts.state.addListener(this);
    }
    syncEditorFromState();
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new LarynxProcessor(); }
