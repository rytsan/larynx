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
    CHECK(delayedCopy(in, out, p.getLatencySamples()));
}

TEST_CASE("processed path with 0 dB trims is also a bit-perfect delayed copy") {
    juce::ScopedJuceInitialiser_GUI init;
    LarynxProcessor p;
    p.prepareToPlay(48000.0, 512);
    const auto in = noise(20000);
    CHECK(delayedCopy(in, run(p, in), p.getLatencySamples()));
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
