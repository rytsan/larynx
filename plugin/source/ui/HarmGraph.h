#pragma once
#include <array>
#include <functional>
#include "RackLookAndFeel.h"

namespace larynx::ui {

// Harmonic bar graph at 60 fps (§8, §12): H1..H40 of the input (dim) behind the processed envelope (lit),
// the network's request as a ghost outline, F0 and the vowel reader in the corner.
// Envelope editor: drag a bar up/down to set its dB offset (±kEditRange), drag across to paint;
// double-click resets; the curve is drawn as a line around the centre. `poll` fills Data from the
// processor (false = no new frame); `getCurve/setCurve/beginEdit` reach the editor state.
class HarmGraph : public juce::Component, private juce::Timer {
public:
    static constexpr int kBars = 40;
    static constexpr float kEditRange = 24.0f;
    struct Data {
        std::array<float, kBars> in {}, out {}, net {}; // dBFS
        float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f;
        char vowel = 0;
    };
    using Curve = std::array<float, kBars>;
    std::function<bool(Data&)> poll;
    std::function<Curve()> getCurve;
    std::function<void(const Curve&)> setCurve;
    std::function<void()> beginEdit; // one undo transaction per gesture

    HarmGraph() {
        shown.in.fill(kBottomDb);
        shown.out.fill(kBottomDb);
        shown.net.fill(kBottomDb);
        startTimerHz(60);
    }

    void paint(juce::Graphics& g) override {
        auto r = getLocalBounds().toFloat();
        g.setColour(colour::panel);
        g.fillRoundedRectangle(r, 6.0f);

        const auto area = plot();
        const float slot = area.getWidth() / (float) kBars;
        const auto accent = static_cast<RackLookAndFeel&>(getLookAndFeel()).getAccent();
        auto height = [&](float db) { return juce::jmap(juce::jlimit(kBottomDb, 0.0f, db), kBottomDb, 0.0f, 0.0f, area.getHeight()); };

        for (int n = 0; n < kBars; ++n) {
            const float x = area.getX() + (float) n * slot + 1.0f, w = slot - 2.0f;
            const float ho = height(shown.out[(size_t) n]), hi = height(shown.in[(size_t) n]);
            g.setColour(accent);
            g.fillRect(x, area.getBottom() - ho, w, ho);
            g.setColour(colour::dim);
            g.fillRect(x, area.getBottom() - std::min(hi, ho), w, std::min(hi, ho));
            if (hi > ho) g.drawRect(x, area.getBottom() - hi, w, hi - ho, 1.0f); // cut below the input
            if (shown.net[(size_t) n] > kBottomDb) {                               // network ghost
                const float hn = height(shown.net[(size_t) n]);
                g.setColour(colour::text.withAlpha(0.35f));
                g.drawHorizontalLine(juce::roundToInt(area.getBottom() - hn), x, x + w);
            }
        }

        if (getCurve) { // editor curve: centre = 0 dB, top/bottom = ±kEditRange
            const auto c = getCurve();
            if (std::any_of(c.begin(), c.end(), [](float v) { return std::abs(v) > 0.01f; })) {
                juce::Path p;
                for (int n = 0; n < kBars; ++n) {
                    const float x = area.getX() + ((float) n + 0.5f) * slot, y = yOf(c[(size_t) n]);
                    n == 0 ? p.startNewSubPath(x, y) : p.lineTo(x, y);
                }
                g.setColour(colour::text.withAlpha(0.8f));
                g.strokePath(p, juce::PathStrokeType(1.5f));
            }
        }

        g.setColour(colour::text);
        g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
        juce::String label = shown.f0 > 0.0f ? juce::String(shown.f0, 1) + " Hz" : juce::String("—");
        if (shown.vowel) label << "   /" << juce::String::charToString((juce::juce_wchar) shown.vowel) << "/  F1 "
                               << juce::roundToInt(shown.f1) << "  F2 " << juce::roundToInt(shown.f2);
        g.drawText(label, area.toNearestInt().removeFromTop(18).withTrimmedLeft(6), juce::Justification::centredLeft);
    }

    void mouseDown(const juce::MouseEvent& e) override {
        if (!setCurve || !getCurve) return;
        if (beginEdit) beginEdit();
        last = -1;
        edit(e);
    }
    void mouseDrag(const juce::MouseEvent& e) override { edit(e); }
    void mouseDoubleClick(const juce::MouseEvent&) override {
        if (!setCurve) return;
        if (beginEdit) beginEdit();
        setCurve({});
    }

private:
    juce::Rectangle<float> plot() const { return getLocalBounds().toFloat().reduced(12.0f, 14.0f); }
    float yOf(float db) const {
        const auto a = plot();
        return a.getCentreY() - juce::jlimit(-kEditRange, kEditRange, db) / kEditRange * a.getHeight() * 0.5f;
    }

    void edit(const juce::MouseEvent& e) {
        const auto a = plot();
        const int n = juce::jlimit(0, kBars - 1, (int) ((e.position.x - a.getX()) / a.getWidth() * (float) kBars));
        const float db = juce::jlimit(-kEditRange, kEditRange, (a.getCentreY() - e.position.y) / (a.getHeight() * 0.5f) * kEditRange);
        auto c = getCurve();
        // Paint every bar between the last and this one, so fast drags leave no gaps.
        const int from = last < 0 ? n : last;
        for (int i = std::min(from, n); i <= std::max(from, n); ++i) c[(size_t) i] = std::round(db * 2.0f) / 2.0f;
        last = n;
        setCurve(c);
        repaint();
    }

    void timerCallback() override {
        Data d;
        if (poll && poll(d)) {
            // Bars: instant rise, ~150 ms fall so the display is readable at 60 fps.
            for (size_t n = 0; n < (size_t) kBars; ++n) {
                shown.in[n] = std::max(d.in[n], shown.in[n] - 2.0f);
                shown.out[n] = std::max(d.out[n], shown.out[n] - 2.0f);
                shown.net[n] = d.net[n];
            }
            shown.f0 = d.f0, shown.f1 = d.f1, shown.f2 = d.f2, shown.vowel = d.vowel;
        }
        repaint();
    }

    static constexpr float kBottomDb = -90.0f;
    Data shown;
    int last = -1;
};

} // namespace larynx::ui
