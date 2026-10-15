#pragma once
#include "RackLookAndFeel.h"

namespace larynx::ui {

// Harmonic bar graph. P1: 40 dummy bars animated at 60 fps; P4 swaps in HarmFrame data.
class HarmGraph : public juce::Component, private juce::Timer {
public:
    HarmGraph() { startTimerHz(60); }

    void paint(juce::Graphics& g) override {
        auto r = getLocalBounds().toFloat();
        g.setColour(colour::panel);
        g.fillRoundedRectangle(r, 6.0f);

        auto area = r.reduced(12.0f, 14.0f);
        const float slot = area.getWidth() / (float) numBars;
        const auto accent = static_cast<RackLookAndFeel&>(getLookAndFeel()).getAccent();

        for (int n = 0; n < numBars; ++n) {
            // Dummy envelope: ~ -6 dB/oct tilt with a slow wobble, input dim / processed lit.
            const float tilt = 1.0f - std::log2((float) n + 1.0f) / 6.0f;
            const float wobble = 0.08f * std::sin(phase + (float) n * 0.4f);
            const float processed = juce::jlimit(0.02f, 1.0f, tilt + wobble);
            const float input = juce::jlimit(0.02f, 1.0f, processed - 0.12f);
            const float x = area.getX() + (float) n * slot + 1.0f;
            const float w = slot - 2.0f;

            g.setColour(accent);
            g.fillRect(x, area.getBottom() - processed * area.getHeight(), w, processed * area.getHeight());
            g.setColour(colour::dim);
            g.fillRect(x, area.getBottom() - input * area.getHeight(), w, input * area.getHeight());
        }
    }

private:
    void timerCallback() override {
        phase += 0.05f;
        repaint();
    }

    static constexpr int numBars = 40;
    float phase = 0.0f;
};

} // namespace larynx::ui
