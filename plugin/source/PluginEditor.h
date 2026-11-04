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
    bool keyPressed(const juce::KeyPress&) override; // Ctrl+Z / Ctrl+Shift+Z: editor curve and parameters

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
    Knob *prior, *mix, *bright, *air, *evenodd, *formant, *sub, *dbl, *breath, *morph, *inTrim, *outTrim;
    juce::ComboBox morphVoice;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> morphVoiceAttachment;

    juce::TextButton modeButtons[3] { juce::TextButton { "DSP" }, juce::TextButton { "LIVE" }, juce::TextButton { "STUDIO" } };
    std::unique_ptr<juce::ParameterAttachment> modeAttachment;

    juce::TextButton bypassPedal { "BYPASS" };
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;

    larynx::ui::HarmGraph graph;
    int lastGraph = 0;
    bool seenGraph = false;
    juce::String status;
    float inLevel = 0.0f, outLevel = 0.0f; // meter display, linear peak with decay
    float inRms = 0.0f, outRms = 0.0f;     // VU (rms) display

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LarynxEditor)
};
