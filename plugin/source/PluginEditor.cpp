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
    dbl = &addKnob(ParamID::dbl, "DOUBLE");
    breath = &addKnob(ParamID::breath, "BREATH");
    inTrim = &addKnob(ParamID::in, "IN");
    outTrim = &addKnob(ParamID::out, "OUT");

    morph = &addKnob(ParamID::morph, "MORPH");
    morph->slider.setEnabled(proc.profiles().singers() > 0);
    if (auto* voices = dynamic_cast<juce::AudioParameterChoice*>(proc.apvts.getParameter(ParamID::morphVoice)))
        morphVoice.addItemList(voices->choices, 1);
    morphVoice.setEnabled(proc.profiles().singers() > 0);
    morphVoiceAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(proc.apvts, ParamID::morphVoice,
                                                                                                    morphVoice);
    addAndMakeVisible(morphVoice);

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

    graph.poll = [this](larynx::ui::HarmGraph::Data& d) {
        const int idx = proc.graphPublished.load(std::memory_order_acquire);
        if (idx == lastGraph && seenGraph) return false;
        const auto& f = proc.graphSlot[idx];
        d.in = f.in, d.out = f.out, d.net = f.net, d.f0 = f.f0, d.f1 = f.f1, d.f2 = f.f2, d.vowel = f.vowel;
        lastGraph = idx, seenGraph = true;
        return true;
    };
    graph.getCurve = [this] { return proc.editorCurve(); };
    graph.setCurve = [this](const larynx::ui::HarmGraph::Curve& c) { proc.setEditorCurve(c); };
    graph.beginEdit = [this] { proc.undo.beginNewTransaction(); };
    addAndMakeVisible(graph);
    setWantsKeyboardFocus(true);
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

bool LarynxEditor::keyPressed(const juce::KeyPress& k) {
    const auto mods = juce::ModifierKeys::commandModifier;
    if (k == juce::KeyPress('z', mods, 0)) return proc.undo.undo();
    if (k == juce::KeyPress('z', mods | juce::ModifierKeys::shiftModifier, 0) || k == juce::KeyPress('y', mods, 0))
        return proc.undo.redo();
    return false;
}

void LarynxEditor::setMode(int index) {
    modeAttachment->setValueAsCompleteGesture((float) index);
}

void LarynxEditor::timerCallback() {
    // Peak: instant attack, ~300 ms release. RMS: VU ballistics (~300 ms integration).
    constexpr float release = 0.89f, vu = 0.105f; // per 33 ms tick
    inLevel = std::max(proc.inPeak.exchange(0.0f), inLevel * release);
    outLevel = std::max(proc.outPeak.exchange(0.0f), outLevel * release);
    // ponytail: exchange of three separate atomics can mix two blocks — off by one block of samples.
    if (const float n = proc.sqCount.exchange(0.0f); n > 0.0f) {
        inRms += vu * (std::sqrt(proc.inSq.exchange(0.0f) / n) - inRms);
        outRms += vu * (std::sqrt(proc.outSq.exchange(0.0f) / n) - outRms);
    }

    const double sr = proc.getSampleRate();
    const double latencyMs = sr > 0 ? 1000.0 * proc.getLatencySamples() / sr : 0.0;
    status = proc.rateSupported()
                 ? juce::String(latencyMs, 1) + " ms   CPU " + juce::String(proc.load.getLoadAsPercentage(), 1) + "%"
                 : juce::String(sr / 1000.0, 1) + " kHz: bypass (44.1/48k)";
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

    auto meter = [&](juce::Rectangle<int> r, float peak, float rms) {
        g.setColour(colour::bg);
        g.fillRect(r);
        auto frac = [](float v) { return juce::jmap(juce::Decibels::gainToDecibels(v, -60.0f), -60.0f, 0.0f, 0.0f, 1.0f); };
        const auto rf = r.toFloat();
        g.setColour(lnf.getAccent());
        g.fillRect(rf.withTrimmedTop(rf.getHeight() * (1.0f - frac(rms))));           // VU body
        g.setColour(juce::Decibels::gainToDecibels(peak) > -1.0f ? juce::Colours::red : colour::text);
        g.fillRect(rf.getX(), rf.getBottom() - rf.getHeight() * frac(peak) - 1.0f, rf.getWidth(), 2.0f); // peak line
    };
    meter({ footer.getX() + 84, footer.getY() + 10, 10, footer.getHeight() - 20 }, inLevel, inRms);
    meter({ footer.getRight() - 94, footer.getY() + 10, 10, footer.getHeight() - 20 }, outLevel, outRms);
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
    layout(create, { &sub->slider, &dbl->slider, &breath->slider, &morph->slider });
    morphVoice.setBounds(morph->label.getBounds().expanded(4, 0)); // voice picker replaces the MORPH label
    morph->label.setVisible(false);

    row(inTrim, { footer.getX() + 8, footer.getY() + 2, 70, footer.getHeight() - 4 });
    row(outTrim, { footer.getRight() - 78, footer.getY() + 2, 70, footer.getHeight() - 4 });
    bypassPedal.setBounds(footer.withSizeKeepingCentre(120, 44));
}
