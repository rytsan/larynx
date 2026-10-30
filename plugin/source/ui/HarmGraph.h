#pragma once
#include <array>
#include <functional>
#include "RackLookAndFeel.h"

namespace larynx::ui {

// Harmonic bar graph at 60 fps (§8): H1..H40 of the input (dim) behind the processed envelope (lit),
// F0 and the vowel reader in the corner. `poll` fills Data from the processor; false = no new frame.
class HarmGraph : public juce::Component, private juce::Timer {
public:
    struct Data {
        std::array<float, 40> in {}, out {}; // dBFS
        float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f;
        char vowel = 0;
    };
    std::function<bool(Data&)> poll;

    HarmGraph() {
        shown.in.fill(kBottomDb);
        shown.out.fill(kBottomDb);
        startTimerHz(60);
    }

    void paint(juce::Graphics& g) override {
        auto r = getLocalBounds().toFloat();
        g.setColour(colour::panel);
        g.fillRoundedRectangle(r, 6.0f);

        auto area = r.reduced(12.0f, 14.0f);
        const float slot = area.getWidth() / (float) shown.in.size();
        const auto accent = static_cast<RackLookAndFeel&>(getLookAndFeel()).getAccent();
        auto height = [&](float db) { return juce::jmap(juce::jlimit(kBottomDb, 0.0f, db), kBottomDb, 0.0f, 0.0f, area.getHeight()); };

        for (size_t n = 0; n < shown.in.size(); ++n) {
            const float x = area.getX() + (float) n * slot + 1.0f, w = slot - 2.0f;
            const float ho = height(shown.out[n]), hi = height(shown.in[n]);
            g.setColour(accent);
            g.fillRect(x, area.getBottom() - ho, w, ho);
            g.setColour(colour::dim);
            g.fillRect(x, area.getBottom() - std::min(hi, ho), w, std::min(hi, ho));
            if (hi > ho) { // cut below the input: outline what was removed
                g.drawRect(x, area.getBottom() - hi, w, hi - ho, 1.0f);
            }
        }

        g.setColour(colour::text);
        g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
        juce::String label = shown.f0 > 0.0f ? juce::String(shown.f0, 1) + " Hz" : juce::String("—");
        if (shown.vowel) label << "   /" << juce::String::charToString((juce::juce_wchar) shown.vowel) << "/  F1 "
                               << juce::roundToInt(shown.f1) << "  F2 " << juce::roundToInt(shown.f2);
        g.drawText(label, area.toNearestInt().removeFromTop(18).withTrimmedLeft(6), juce::Justification::centredLeft);
    }

private:
    void timerCallback() override {
        Data d;
        if (poll && poll(d)) {
            // Bars: instant rise, ~150 ms fall so the display is readable at 60 fps.
            for (size_t n = 0; n < d.in.size(); ++n) {
                shown.in[n] = std::max(d.in[n], shown.in[n] - 2.0f);
                shown.out[n] = std::max(d.out[n], shown.out[n] - 2.0f);
            }
            shown.f0 = d.f0, shown.f1 = d.f1, shown.f2 = d.f2, shown.vowel = d.vowel;
        }
        repaint();
    }

    static constexpr float kBottomDb = -90.0f;
    Data shown;
};

} // namespace larynx::ui
