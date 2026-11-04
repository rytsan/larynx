// Headless render through the real plugin processor (neural modes, presets, every parameter):
//   larynx_render_plugin in.wav out.wav [--<paramID> value ...]   e.g. --mode 1 --prior 100
// Values in parameter units (mode: 0 DSP, 1 Live, 2 Studio). Latency removed so out lines up with in.
#include <cstdio>
#include <cstring>
#include <juce_events/juce_events.h>
#include "PluginProcessor.h"
#include "wav.h"

int main(int argc, char** argv) {
    if (argc < 3 || argc % 2 == 0) {
        std::fprintf(stderr, "usage: %s in.wav out.wav [--<paramID> value ...]\n", argv[0]);
        return 2;
    }
    juce::ScopedJuceInitialiser_GUI juce;
    std::vector<float> x;
    double fs = 0.0;
    if (!readWav(argv[1], x, fs)) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    LarynxProcessor p;
    for (int i = 3; i + 1 < argc; i += 2) {
        auto* param = std::strncmp(argv[i], "--", 2) ? nullptr : p.apvts.getParameter(argv[i] + 2);
        if (!param) {
            std::fprintf(stderr, "unknown parameter %s\n", argv[i]);
            return 2;
        }
        param->setValueNotifyingHost(param->convertTo0to1(std::strtof(argv[i + 1], nullptr)));
    }
    constexpr int kBlock = 512;
    p.prepareToPlay(fs, kBlock);
    const auto L = (size_t) p.getLatencySamples();
    x.resize(x.size() + L, 0.0f);
    juce::MidiBuffer midi;
    juce::AudioBuffer<float> buf(1, kBlock);
    std::vector<float> y;
    y.reserve(x.size());
    for (size_t pos = 0; pos < x.size(); pos += kBlock) {
        const int n = (int) std::min<size_t>(kBlock, x.size() - pos);
        buf.setSize(1, n, false, false, true);
        std::copy_n(x.data() + pos, n, buf.getWritePointer(0));
        p.processBlock(buf, midi);
        y.insert(y.end(), buf.getReadPointer(0), buf.getReadPointer(0) + n);
    }
    y.erase(y.begin(), y.begin() + (long) L);
    if (!writeWav(argv[2], y, fs)) {
        std::fprintf(stderr, "cannot write %s\n", argv[2]);
        return 1;
    }
    std::fprintf(stderr, "latency %zu samples\n", L);
}
