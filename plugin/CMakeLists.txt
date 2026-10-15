juce_add_plugin(Larynx
    COMPANY_NAME "Ricardo"
    PLUGIN_MANUFACTURER_CODE Rcdo
    PLUGIN_CODE Lrnx
    FORMATS VST3 Standalone
    PRODUCT_NAME "LARYNX"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
    VST3_CATEGORIES Fx)

clap_juce_extensions_plugin(TARGET Larynx
    CLAP_ID "com.ricardo.larynx"
    CLAP_FEATURES audio-effect mono)

target_sources(Larynx PRIVATE
    source/PluginProcessor.cpp
    source/PluginEditor.cpp)

target_include_directories(Larynx PUBLIC source)

target_compile_definitions(Larynx PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(Larynx
    PRIVATE
        juce::juce_audio_utils
        juce::juce_opengl
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)
