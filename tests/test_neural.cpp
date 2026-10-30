#include <catch2/catch_test_macros.hpp>
#include <juce_events/juce_events.h>
#include <cmath>
#include <random>
#include "PluginProcessor.h"

namespace {

// Random TCN with delay D, as JSON in the export format of model.py (default: small, C = 8, 3 layers).
juce::String modelJson(int D, unsigned seed, int C = 8, juce::Array<juce::var> dil = { 1, 2, 4 }) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.0f, 0.3f / std::sqrt((float) C / 8.0f));
    const int k = 3;
    auto mat = [&](int rows, int cols, int depth) {
        juce::Array<juce::var> r;
        for (int i = 0; i < rows; ++i) {
            juce::Array<juce::var> c;
            for (int j = 0; j < cols; ++j) {
                if (depth == 0) {
                    c.add(n(rng));
                    continue;
                }
                juce::Array<juce::var> d;
                for (int q = 0; q < depth; ++q) d.add(n(rng));
                c.add(d);
            }
            r.add(c);
        }
        return juce::var(r);
    };
    auto vec = [&](int len) {
        juce::Array<juce::var> v;
        for (int i = 0; i < len; ++i) v.add(n(rng));
        return juce::var(v);
    };
    auto layer = [&](const char* name, juce::var w, juce::var b) {
        auto* o = new juce::DynamicObject();
        o->setProperty("name", name);
        o->setProperty("w", w);
        o->setProperty("b", b);
        return juce::var(o);
    };
    juce::Array<juce::var> layers { layer("in", mat(C, 51, 0), vec(C)) };
    for (int l = 0; l < dil.size(); ++l) layers.add(layer("conv", mat(2 * C, C, k), vec(2 * C)));
    layers.add(layer("out", mat(88, C, 0), vec(88)));
    auto* o = new juce::DynamicObject();
    o->setProperty("contract", larynx::kNetContract);
    o->setProperty("inputs", 51);
    o->setProperty("outputs", 88);
    o->setProperty("D", D);
    o->setProperty("n_channels", C);
    o->setProperty("kernel", k);
    o->setProperty("dilations", dil);
    o->setProperty("layers", layers);
    return juce::JSON::toString(juce::var(o));
}

void setParam(LarynxProcessor& p, const char* id, float value) {
    auto* param = p.apvts.getParameter(id);
    param->setValueNotifyingHost(param->convertTo0to1(value));
}

std::vector<float> tone(size_t len) {
    std::vector<float> x(len);
    for (size_t i = 0; i < len; ++i)
        for (int k = 1; k <= 12; ++k) x[i] += 0.2f / (float) k * std::sin(2.0f * 3.14159265f * 196.0f * (float) k * (float) i / 48000.0f);
    return x;
}

std::vector<float> run(LarynxProcessor& p, const std::vector<float>& in) {
    std::vector<float> out(in.size());
    juce::MidiBuffer midi;
    for (size_t pos = 0; pos < in.size(); pos += 480) {
        const int n = (int) std::min<size_t>(480, in.size() - pos);
        juce::AudioBuffer<float> buf(1, n);
        std::copy_n(in.data() + pos, n, buf.getWritePointer(0));
        p.processBlock(buf, midi);
        std::copy_n(buf.getReadPointer(0), n, out.data() + pos);
    }
    return out;
}

} // namespace

TEST_CASE("model JSON: parse, contract check", "[neural]") {
    larynx::TcnWeights w;
    REQUIRE(LarynxProcessor::parseModel(modelJson(0, 1), w));
    REQUIRE(w.channels == 8);
    REQUIRE(w.convW.size() == 3);
    REQUIRE_FALSE(LarynxProcessor::parseModel(modelJson(0, 1).replace("\"contract\": 1", "\"contract\": 99"), w));
    REQUIRE_FALSE(LarynxProcessor::parseModel("{}", w));
}

TEST_CASE("mode without weights runs DSP", "[neural]") {
    juce::ScopedJuceInitialiser_GUI init;
    LarynxProcessor p;
    REQUIRE_FALSE(p.loadModel(LarynxProcessor::Live, {})); // invalid weights unload (embedded model, if any)
    setParam(p, ParamID::mode, 1.0f);
    p.prepareToPlay(48000.0, 480);
    REQUIRE(p.runningMode() == LarynxProcessor::DSP);
    REQUIRE(p.getLatencySamples() == LarynxProcessor::latencyFor(48000.0));
}

TEST_CASE("Live: network shapes the voice, latency unchanged; Prior = 0 is the DSP identity", "[neural]") {
    juce::ScopedJuceInitialiser_GUI init;
    larynx::TcnWeights w;
    REQUIRE(LarynxProcessor::parseModel(modelJson(0, 2), w));
    const auto x = tone(48000);
    LarynxProcessor p;
    REQUIRE(p.loadModel(LarynxProcessor::Live, w));
    setParam(p, ParamID::mode, 1.0f);
    setParam(p, ParamID::prior, 100.0f);
    p.prepareToPlay(48000.0, 480);
    REQUIRE(p.runningMode() == LarynxProcessor::Live);
    REQUIRE(p.getLatencySamples() == LarynxProcessor::latencyFor(48000.0));
    const auto y = run(p, x);
    const size_t L = (size_t) p.getLatencySamples();
    double diff = 0.0;
    bool finite = true;
    for (size_t i = 24000; i < x.size(); ++i) {
        finite &= std::isfinite(y[i]);
        diff = std::max(diff, (double) std::abs(y[i] - x[i - L]));
    }
    REQUIRE(finite);
    REQUIRE(diff > 1e-3);

    LarynxProcessor q;
    q.loadModel(LarynxProcessor::Live, w);
    setParam(q, ParamID::mode, 1.0f);
    setParam(q, ParamID::prior, 0.0f);
    q.prepareToPlay(48000.0, 480);
    const auto z = run(q, x);
    float worst = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) worst = std::max(worst, std::abs(z[i] - (i < L ? 0.0f : x[i - L])));
    REQUIRE(worst < 1e-5f);
}

TEST_CASE("Studio: latency = window + D hops, output aligned with the bypass delay", "[neural]") {
    juce::ScopedJuceInitialiser_GUI init;
    larynx::TcnWeights w;
    REQUIRE(LarynxProcessor::parseModel(modelJson(16, 3), w));
    for (double sr : { 44100.0, 48000.0 }) {
        LarynxProcessor p;
        p.loadModel(LarynxProcessor::Studio, w);
        setParam(p, ParamID::mode, 2.0f);
        setParam(p, ParamID::prior, 0.0f); // network off: the studio path must be the delayed identity
        p.prepareToPlay(sr, 480);
        REQUIRE(p.runningMode() == LarynxProcessor::Studio);
        const int hop = sr == 48000.0 ? 256 : 235;
        REQUIRE(p.getLatencySamples() == LarynxProcessor::latencyFor(sr) + 16 * hop);
        const auto x = tone(48000);
        const auto y = run(p, x);
        const size_t L = (size_t) p.getLatencySamples();
        float worst = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) worst = std::max(worst, std::abs(y[i] - (i < L ? 0.0f : x[i - L])));
        REQUIRE(worst < 1e-5f);
    }
}

TEST_CASE("RTF of the full chain with the §2.3 TCN (96 ch, 7 layers) < 0.2", "[neural][rtf]") {
    juce::ScopedJuceInitialiser_GUI init;
    larynx::TcnWeights w;
    REQUIRE(LarynxProcessor::parseModel(modelJson(16, 4, 96, { 1, 2, 4, 8, 16, 32, 64 }), w));
    const auto x = tone(48000 * 5);
    for (int mode : { 1, 2 }) {
        LarynxProcessor p;
        p.loadModel((LarynxProcessor::Mode) mode, w);
        setParam(p, ParamID::mode, (float) mode);
        p.prepareToPlay(48000.0, 480);
        const auto t0 = juce::Time::getMillisecondCounterHiRes();
        run(p, x);
        const double rtf = (juce::Time::getMillisecondCounterHiRes() - t0) / 1000.0 / 5.0;
        WARN((mode == 1 ? "Live" : "Studio") << " RTF " << rtf);
        CHECK(rtf < 0.2);
    }
}
