#include <catch2/catch_test_macros.hpp>
#include <juce_events/juce_events.h>
#include <random>
#include "PluginProcessor.h"

namespace {

void setParam(LarynxProcessor& p, const char* id, float value) {
    auto* param = p.apvts.getParameter(id);
    param->setValueNotifyingHost(param->convertTo0to1(value));
}

// Runs `in` through the processor in uneven block sizes, returns the output.
std::vector<float> run(LarynxProcessor& p, const std::vector<float>& in) {
    std::vector<float> out(in.size());
    juce::MidiBuffer midi;
    const int sizes[] = { 512, 17, 256, 1, 333 };
    size_t pos = 0;
    for (int b = 0; pos < in.size(); ++b) {
        const int n = (int) std::min<size_t>((size_t) sizes[b % 5], in.size() - pos);
        juce::AudioBuffer<float> buf(1, n);
        std::copy_n(in.data() + pos, n, buf.getWritePointer(0));
        p.processBlock(buf, midi);
        std::copy_n(buf.getReadPointer(0), n, out.data() + pos);
        pos += (size_t) n;
    }
    return out;
}

std::vector<float> noise(size_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// out[n] must equal in[n - latency] bit-for-bit, from sample `from` on.
bool delayedCopy(const std::vector<float>& in, const std::vector<float>& out, int latency, size_t from = 0) {
    for (size_t n = from; n < out.size(); ++n) {
        const float expected = n < (size_t) latency ? 0.0f : in[n - (size_t) latency];
        if (out[n] != expected) return false;
    }
    return true;
}

} // namespace

TEST_CASE("bypass is bit-perfect delayed copy (NFR4)") {
    juce::ScopedJuceInitialiser_GUI init;
    for (double sr : { 44100.0, 48000.0 }) {
        LarynxProcessor p;
        setParam(p, ParamID::bypass, 1.0f);
        setParam(p, ParamID::in, 12.0f); // trims must not leak into bypass
        p.prepareToPlay(sr, 512);
        REQUIRE(p.getLatencySamples() == LarynxProcessor::latencyFor(sr));
        const auto in = noise(20000);
        CHECK(delayedCopy(in, run(p, in), p.getLatencySamples()));
    }
}

TEST_CASE("host bypass (processBlockBypassed) is the same delayed copy") {
    juce::ScopedJuceInitialiser_GUI init;
    LarynxProcessor p;
    p.prepareToPlay(48000.0, 512);
    // Starting un-bypassed, the first 10 ms are a crossfade; exact after that.
    const auto in = noise(8192);
    juce::AudioBuffer<float> buf(1, (int) in.size());
    std::copy(in.begin(), in.end(), buf.getWritePointer(0));
    juce::MidiBuffer midi;
    p.processBlockBypassed(buf, midi);
    std::vector<float> out(buf.getReadPointer(0), buf.getReadPointer(0) + in.size());
    CHECK(delayedCopy(in, out, p.getLatencySamples(), 480));
}

TEST_CASE("processed path with 0 dB trims is the delayed input within -100 dBFS (P3 identity)") {
    juce::ScopedJuceInitialiser_GUI init;
    for (double sr : { 44100.0, 48000.0 }) {
        LarynxProcessor p;
        p.prepareToPlay(sr, 512);
        const auto in = noise(20000);
        const auto out = run(p, in);
        const size_t L = (size_t) p.getLatencySamples();
        float worst = 0.0f;
        for (size_t n = 0; n < out.size(); ++n)
            worst = std::max(worst, std::abs(out[n] - (n < L ? 0.0f : in[n - L])));
        CHECK(worst < 1e-5f);
    }
}

TEST_CASE("bypass toggle returns to bit-perfect after the 10 ms crossfade") {
    juce::ScopedJuceInitialiser_GUI init;
    LarynxProcessor p;
    setParam(p, ParamID::out, -6.0f);
    p.prepareToPlay(48000.0, 512);
    const auto in = noise(48000);
    run(p, std::vector<float>(in.begin(), in.begin() + 4800)); // settle trims
    setParam(p, ParamID::bypass, 1.0f);
    const auto out = run(p, in);
    // Delay line still holds pre-toggle samples; compare after latency + fade (480 samples).
    const int L = p.getLatencySamples();
    for (size_t n = (size_t) L + 1024; n < out.size(); ++n)
        REQUIRE(out[n] == in[n - (size_t) L]);
}

TEST_CASE("state roundtrip restores every parameter") {
    juce::ScopedJuceInitialiser_GUI init;
    LarynxProcessor a;
    setParam(a, ParamID::mode, 2.0f);
    setParam(a, ParamID::prior, 33.3f);
    setParam(a, ParamID::bright, -4.5f);
    setParam(a, ParamID::formant, 2.25f);
    setParam(a, ParamID::out, -12.0f);
    setParam(a, ParamID::bypass, 1.0f);

    juce::MemoryBlock state;
    a.getStateInformation(state);
    LarynxProcessor b;
    b.setStateInformation(state.getData(), (int) state.getSize());

    for (auto* param : a.getParameters())
        CHECK(param->getValue() == b.getParameters()[param->getParameterIndex()]->getValue());
}

TEST_CASE("mode is not automatable") {
    juce::ScopedJuceInitialiser_GUI init;
    LarynxProcessor p;
    CHECK_FALSE(p.apvts.getParameter(ParamID::mode)->isAutomatable());
    CHECK(p.apvts.getParameter(ParamID::prior)->isAutomatable());
}

TEST_CASE("Prior = 0 with Brightness +6 changes a voiced tone (M4 order, P4 gate)") {
    juce::ScopedJuceInitialiser_GUI init;
    std::vector<float> tone(48000);
    for (size_t n = 0; n < tone.size(); ++n)
        for (int k = 1; k <= 20; ++k)
            tone[n] += 0.3f / (float) k * std::sin(2.0f * 3.14159265f * 220.0f * (float) k * (float) n / 48000.0f);
    LarynxProcessor p;
    setParam(p, ParamID::prior, 0.0f);
    p.prepareToPlay(48000.0, 512);
    const auto flat = run(p, tone);
    LarynxProcessor q;
    setParam(q, ParamID::prior, 0.0f);
    setParam(q, ParamID::bright, 6.0f);
    q.prepareToPlay(48000.0, 512);
    const auto bright = run(q, tone);
    double diff = 0.0, ref = 0.0;
    for (size_t n = 24000; n < tone.size(); ++n) {
        diff += std::pow(bright[n] - flat[n], 2.0);
        ref += std::pow(flat[n], 2.0);
    }
    CHECK(10.0 * std::log10(diff / ref) > -20.0); // clearly audible change
    const size_t L = (size_t) p.getLatencySamples();
    float worst = 0.0f;
    for (size_t n = 24000; n < tone.size(); ++n) worst = std::max(worst, std::abs(flat[n] - tone[n - L]));
    CHECK(worst < 1e-5f); // and Prior = 0 with timbre at 0 is still the identity
}

TEST_CASE("factory presets set parameters, unlisted ones back to default") {
    juce::ScopedJuceInitialiser_GUI init;
    LarynxProcessor p;
    REQUIRE(p.getNumPrograms() == 6);
    setParam(p, ParamID::formant, 3.0f);
    p.setCurrentProgram(4); // Sopro etéreo
    REQUIRE(p.getProgramName(4) == "Sopro etéreo");
    CHECK(p.apvts.getRawParameterValue(ParamID::breath)->load() == 70.0f);
    CHECK(p.apvts.getRawParameterValue(ParamID::formant)->load() == 0.0f);
    p.setCurrentProgram(0);
    CHECK(p.apvts.getRawParameterValue(ParamID::prior)->load() == 0.0f);
    CHECK(p.apvts.getRawParameterValue(ParamID::breath)->load() == 0.0f);
}

TEST_CASE("envelope editor: curve survives state roundtrip, undo restores it, and it shapes the sound") {
    juce::ScopedJuceInitialiser_GUI init;
    LarynxProcessor p;
    std::array<float, LarynxProcessor::kEditBars> c {};
    c[2] = 6.0f;
    p.undo.beginNewTransaction();
    p.setEditorCurve(c);
    juce::MemoryBlock mb;
    p.getStateInformation(mb);
    LarynxProcessor q;
    q.setStateInformation(mb.getData(), (int) mb.getSize());
    CHECK(q.editorCurve()[2] == 6.0f);
    p.undo.undo();
    CHECK(p.editorCurve()[2] == 0.0f);
    p.undo.redo();
    CHECK(p.editorCurve()[2] == 6.0f);

    std::vector<float> tone(48000);
    for (size_t n = 0; n < tone.size(); ++n)
        for (int k = 1; k <= 8; ++k) tone[n] += 0.2f / (float) k * std::sin(2.0f * 3.14159265f * 220.0f * (float) k * (float) n / 48000.0f);
    p.prepareToPlay(48000.0, 512);
    const auto out = run(p, tone);
    const size_t L = (size_t) p.getLatencySamples();
    float worst = 0.0f;
    for (size_t n = 24000; n < tone.size(); ++n) worst = std::max(worst, std::abs(out[n] - tone[n - L]));
    CHECK(worst > 0.01f);
}

TEST_CASE("morph: profiles embedded, 0% is identity, 100% reshapes a voiced tone") {
    juce::ScopedJuceInitialiser_GUI init;
    REQUIRE(LarynxProcessor::profiles().singers() == 12);
    std::vector<float> tone(48000);
    for (size_t n = 0; n < tone.size(); ++n) // /a/-like: H3-H4 strong around 700 Hz at f0 = 220 Hz
        for (int k = 1; k <= 30; ++k)
            tone[n] += (k == 3 || k == 4 ? 0.25f : 0.08f / (float) k) * std::sin(2.0f * 3.14159265f * 220.0f * (float) k * (float) n / 48000.0f);
    LarynxProcessor p;
    p.prepareToPlay(48000.0, 512);
    const size_t L = (size_t) p.getLatencySamples();
    auto worst = [&](const std::vector<float>& out) {
        float w = 0.0f;
        for (size_t n = 24000; n < tone.size(); ++n) w = std::max(w, std::abs(out[n] - tone[n - L]));
        return w;
    };
    CHECK(worst(run(p, tone)) < 1e-5f);
    LarynxProcessor q;
    setParam(q, ParamID::morph, 100.0f);
    setParam(q, ParamID::morphVoice, 5.0f);
    q.prepareToPlay(48000.0, 512);
    CHECK(worst(run(q, tone)) > 0.01f);
}

TEST_CASE("edge cases: unsupported rates bypass, DC, clipping, silence and noise stay finite") {
    juce::ScopedJuceInitialiser_GUI init;
    for (double sr : { 88200.0, 96000.0 }) {
        LarynxProcessor p;
        p.prepareToPlay(sr, 512);
        CHECK_FALSE(p.rateSupported());
        const auto in = noise(30000);
        CHECK(delayedCopy(in, run(p, in), p.getLatencySamples()));
    }
    LarynxProcessor p;
    setParam(p, ParamID::bright, 12.0f);
    setParam(p, ParamID::sub, 100.0f);
    setParam(p, ParamID::breath, 100.0f);
    p.prepareToPlay(48000.0, 512);
    std::vector<std::vector<float>> inputs = { std::vector<float>(48000, 0.0f), std::vector<float>(48000, 0.5f), noise(48000) };
    std::vector<float> clipped(48000);
    for (size_t n = 0; n < clipped.size(); ++n) clipped[n] = std::clamp(4.0f * std::sin(2.0f * 3.14159265f * 150.0f * (float) n / 48000.0f), -1.0f, 1.0f);
    inputs.push_back(clipped);
    for (auto& in : inputs) {
        const auto out = run(p, in);
        bool finite = true;
        float peak = 0.0f;
        for (float v : out) finite &= std::isfinite(v), peak = std::max(peak, std::abs(v));
        CHECK(finite);
        CHECK(peak < 100.0f); // +12 dB/oct on a clipped square reaches ~+30 dBFS (each harmonic capped at 0 dBFS)
    }
}
