#include "PluginEditor.h"

using namespace larynx::ui;

namespace {
constexpr int W = 900, H = 540;
const juce::Rectangle<int> header { 0, 0, W, 48 }, graphArea { 12, 56, W - 24, 222 },
    restore { 12, 288, 228, 160 }, timbre { 248, 288, 404, 160 }, create { 660, 288, 228, 160 },
    footer { 12, 456, W - 24, 72 };
} // namespace

LarynxEditor::LarynxEditor(LarynxProcessor& p) : AudioProcessorEditor(p), proc(p) {
    setLookAndFeel(&lnf);
    gl.attachTo(*this); // whole editor rendered through OpenGL (§5 graph at 60 fps)

    prior = &addKnob(ParamID::prior, "PRIOR");
    mix = &addKnob(ParamID::mix, "MIX");
    bright = &addKnob(ParamID::bright, "BRIGHT");
    air = &addKnob(ParamID::air, "AIR");
    evenodd = &addKnob(ParamID::evenodd, "EVEN/ODD");
    formant = &addKnob(ParamID::formant, "FORMANT");
    sub = &addKnob(ParamID::sub, "SUB");
    breath = &addKnob(ParamID::breath, "BREATH");
    inTrim = &addKnob(ParamID::in, "IN");
    outTrim = &addKnob(ParamID::out, "OUT");

    morph.setEnabled(false);
    morph.setTooltip("Morph (P8)");
    addAndMakeVisible(morph);
    morphLabel.setText("MORPH", juce::dontSendNotification);
    morphLabel.setJustificationType(juce::Justification::centred);
    morphLabel.setFont(juce::FontOptions(11.0f));
    addAndMakeVisible(morphLabel);

    auto* modeParam = proc.apvts.getParameter(ParamID::mode);
    modeAttachment = std::make_unique<juce::ParameterAttachment>(*modeParam, [this](float v) {
        const int idx = juce::roundToInt(v);
        for (int i = 0; i < 3; ++i)
            modeButtons[i].setToggleState(i == idx, juce::dontSendNotification);
        lnf.setAccent(idx == 2 ? colour::cyan : colour::amber); // Live amber / Studio cyan (§11)
        repaint();
    });
    for (int i = 0; i < 3; ++i) {
        modeButtons[i].setClickingTogglesState(false);
        modeButtons[i].onClick = [this, i] { setMode(i); };
        addAndMakeVisible(modeButtons[i]);
    }
    modeAttachment->sendInitialUpdate();

    bypassPedal.setClickingTogglesState(true);
    bypassAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(proc.apvts, ParamID::bypass,
                                                                                              bypassPedal);
    addAndMakeVisible(bypassPedal);

    addAndMakeVisible(graph);
    setSize(W, H);
    startTimerHz(30);
}

LarynxEditor::~LarynxEditor() {
    gl.detach();
    setLookAndFeel(nullptr);
}

LarynxEditor::Knob& LarynxEditor::addKnob(const char* paramId, const char* text) {
    auto& k = *knobs.emplace_back(std::make_unique<Knob>());
    k.slider.setPopupDisplayEnabled(true, true, this); // exact value bubble while dragging
    k.slider.setScrollWheelEnabled(true);
    k.attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(proc.apvts, paramId, k.slider);
    k.label.setText(text, juce::dontSendNotification);
    k.label.setJustificationType(juce::Justification::centred);
    k.label.setFont(juce::FontOptions(11.0f));
    addAndMakeVisible(k.slider);
    addAndMakeVisible(k.label);
    return k;
}

void LarynxEditor::setMode(int index) {
    modeAttachment->setValueAsCompleteGesture((float) index);
}

void LarynxEditor::timerCallback() {
    // Peak meters: instant attack, ~300 ms release.
    constexpr float release = 0.89f; // per 33 ms tick
    inLevel = std::max(proc.inPeak.exchange(0.0f), inLevel * release);
    outLevel = std::max(proc.outPeak.exchange(0.0f), outLevel * release);

    const double sr = proc.getSampleRate();
    const double latencyMs = sr > 0 ? 1000.0 * proc.getLatencySamples() / sr : 0.0;
    status = juce::String(latencyMs, 1) + " ms   CPU " + juce::String(proc.load.getLoadAsPercentage(), 1) + "%";
    repaint(header.withTrimmedLeft(W / 2));
    repaint(footer);
}

void LarynxEditor::paint(juce::Graphics& g) {
    g.fillAll(colour::bg);

    g.setColour(colour::text);
    g.setFont(juce::FontOptions(20.0f, juce::Font::bold));
    g.drawText("LARYNX", header.withTrimmedLeft(16), juce::Justification::centredLeft);

    // Latency LED + CPU.
    g.setColour(lnf.getAccent());
    g.fillEllipse(W - 196.0f, 20.0f, 8.0f, 8.0f);
    g.setColour(colour::text);
    g.setFont(juce::FontOptions(12.0f));
    g.drawText(status, juce::Rectangle<int>(W - 180, 0, 168, 48), juce::Justification::centredLeft);

    for (auto [r, title] : { std::pair { restore, "RESTAURAR" }, { timbre, "TIMBRE" }, { create, "CRIAR" } }) {
        g.setColour(colour::panel);
        g.fillRoundedRectangle(r.toFloat(), 6.0f);
        g.setColour(colour::text.withAlpha(0.6f));
        g.setFont(juce::FontOptions(11.0f, juce::Font::bold));
        g.drawText(title, r.withHeight(24).withTrimmedLeft(10), juce::Justification::centredLeft);
    }

    g.setColour(colour::panel);
    g.fillRoundedRectangle(footer.toFloat(), 6.0f);

    auto meter = [&](juce::Rectangle<int> r, float level) {
        g.setColour(colour::bg);
        g.fillRect(r);
        const float db = juce::Decibels::gainToDecibels(level, -60.0f);
        const float frac = juce::jmap(db, -60.0f, 0.0f, 0.0f, 1.0f);
        g.setColour(db > -1.0f ? juce::Colours::red : lnf.getAccent());
        g.fillRect(r.toFloat().withTrimmedTop(r.getHeight() * (1.0f - frac)));
    };
    meter({ footer.getX() + 84, footer.getY() + 10, 10, footer.getHeight() - 20 }, inLevel);
    meter({ footer.getRight() - 94, footer.getY() + 10, 10, footer.getHeight() - 20 }, outLevel);
}

void LarynxEditor::resized() {
    auto row = [](Knob* k, juce::Rectangle<int> r) {
        k->label.setBounds(r.removeFromBottom(18));
        k->slider.setBounds(r.reduced(6));
    };
    auto layout = [&](juce::Rectangle<int> panel, std::initializer_list<juce::Component*> items) {
        auto r = panel.withTrimmedTop(24).reduced(6, 4);
        const int w = r.getWidth() / (int) items.size();
        for (auto* c : items) {
            auto cell = r.removeFromLeft(w);
            if (c == &morph) {
                morphLabel.setBounds(cell.removeFromBottom(18));
                morph.setBounds(cell.reduced(6));
                continue;
            }
            for (auto& k : knobs)
                if (&k->slider == c)
                    row(k.get(), cell);
        }
    };

    auto hdr = header.reduced(0, 10);
    for (int i = 0; i < 3; ++i)
        modeButtons[i].setBounds(W / 2 - 120 + i * 80, hdr.getY(), 76, hdr.getHeight());

    graph.setBounds(graphArea);
    layout(restore, { &prior->slider, &mix->slider });
    layout(timbre, { &bright->slider, &air->slider, &evenodd->slider, &formant->slider });
    layout(create, { &sub->slider, &breath->slider, &morph });

    row(inTrim, { footer.getX() + 8, footer.getY() + 2, 70, footer.getHeight() - 4 });
    row(outTrim, { footer.getRight() - 78, footer.getY() + 2, 70, footer.getHeight() - 4 });
    bypassPedal.setBounds(footer.withSizeKeepingCentre(120, 44));
}
