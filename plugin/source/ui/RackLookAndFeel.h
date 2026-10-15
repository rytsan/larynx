#pragma once
#include <juce_gui_basics/juce_gui_basics.h>

namespace larynx::ui {

// §14 palette.
namespace colour {
inline const juce::Colour bg { 0xff14161a }, panel { 0xff1d2025 }, amber { 0xffe8a33d }, cyan { 0xff4dc9d6 },
    text { 0xffc9ccd1 }, dim { 0xff3a3f47 };
}

class RackLookAndFeel : public juce::LookAndFeel_V4 {
public:
    RackLookAndFeel() {
        setColour(juce::Slider::textBoxTextColourId, colour::text);
        setColour(juce::Label::textColourId, colour::text);
        setColour(juce::TextButton::buttonColourId, colour::panel);
        setColour(juce::TextButton::buttonOnColourId, colour::panel);
        setColour(juce::TextButton::textColourOffId, colour::text.withAlpha(0.5f));
        setColour(juce::TextButton::textColourOnId, colour::amber);
        setColour(juce::BubbleComponent::backgroundColourId, colour::panel);
        setColour(juce::BubbleComponent::outlineColourId, colour::dim);
        setColour(juce::TooltipWindow::textColourId, colour::text);
    }

    void setAccent(juce::Colour c) {
        accent = c;
        setColour(juce::TextButton::textColourOnId, c);
    }
    juce::Colour getAccent() const { return accent; }

    // Matte knob + LED arc + value marker, no gradients (§5).
    void drawRotarySlider(juce::Graphics& g, int x, int y, int w, int h, float pos, float a0, float a1,
                          juce::Slider& s) override {
        auto r = juce::Rectangle<int>(x, y, w, h).toFloat().reduced(4.0f);
        const float size = std::min(r.getWidth(), r.getHeight());
        r = r.withSizeKeepingCentre(size, size);
        const auto c = r.getCentre();
        const float arcR = size * 0.5f - 3.0f;
        const float angle = a0 + pos * (a1 - a0);
        const auto on = s.isEnabled() ? accent : colour::dim;

        juce::Path track, value;
        track.addCentredArc(c.x, c.y, arcR, arcR, 0.0f, a0, a1, true);
        g.setColour(colour::dim);
        g.strokePath(track, juce::PathStrokeType(3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        // Bipolar params light from centre, unipolar from the start.
        const bool bipolar = s.getMinimum() < 0.0 && s.getMaximum() > 0.0;
        const float from = bipolar ? (a0 + a1) * 0.5f : a0;
        value.addCentredArc(c.x, c.y, arcR, arcR, 0.0f, std::min(from, angle), std::max(from, angle), true);
        g.setColour(on);
        g.strokePath(value, juce::PathStrokeType(3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        const float bodyR = arcR - 7.0f;
        g.setColour(colour::panel.brighter(0.15f));
        g.fillEllipse(c.x - bodyR, c.y - bodyR, bodyR * 2, bodyR * 2);
        g.setColour(colour::bg);
        g.drawEllipse(c.x - bodyR, c.y - bodyR, bodyR * 2, bodyR * 2, 1.5f);

        const auto tip = c.getPointOnCircumference(bodyR - 4.0f, angle);
        const auto base = c.getPointOnCircumference(bodyR * 0.45f, angle);
        g.setColour(colour::text);
        g.drawLine({ base, tip }, 2.5f);
    }

private:
    juce::Colour accent = colour::amber;
};

} // namespace larynx::ui
