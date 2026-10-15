#pragma once
#include <juce_opengl/juce_opengl.h>
#include "PluginProcessor.h"
#include "ui/HarmGraph.h"

class LarynxEditor : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit LarynxEditor(LarynxProcessor&);
    ~LarynxEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    struct Knob {
        juce::Slider slider { juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::NoTextBox };
        juce::Label label;
        std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;
    };

    void timerCallback() override;
    Knob& addKnob(const char* paramId, const char* text);
    void setMode(int index);

    LarynxProcessor& proc;
    larynx::ui::RackLookAndFeel lnf;
    juce::OpenGLContext gl;
    juce::TooltipWindow tooltips { this };

    std::vector<std::unique_ptr<Knob>> knobs;
    Knob *prior, *mix, *bright, *air, *evenodd, *formant, *sub, *breath, *inTrim, *outTrim;
    juce::Slider morph { juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::NoTextBox }; // P8 placeholder
    juce::Label morphLabel;

    juce::TextButton modeButtons[3] { juce::TextButton { "DSP" }, juce::TextButton { "LIVE" }, juce::TextButton { "STUDIO" } };
    std::unique_ptr<juce::ParameterAttachment> modeAttachment;

    juce::TextButton bypassPedal { "BYPASS" };
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;

    larynx::ui::HarmGraph graph;
    juce::String status;
    float inLevel = 0.0f, outLevel = 0.0f; // meter display, linear peak with decay

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LarynxEditor)
};
