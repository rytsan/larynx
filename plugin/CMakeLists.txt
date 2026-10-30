# core/: pure DSP, no JUCE. Shared by the plugin, tests and tools/bench_plugin.
add_library(larynx_core STATIC source/core/analyzer.cpp source/core/skeleton.cpp source/core/synth.cpp source/core/neuralcore.cpp)
target_include_directories(larynx_core PUBLIC source)
target_compile_features(larynx_core PUBLIC cxx_std_20)
set_target_properties(larynx_core PROPERTIES POSITION_INDEPENDENT_CODE ON) # linked into ml/pycore
target_link_libraries(larynx_core PRIVATE juce::juce_recommended_warning_flags juce::juce_recommended_config_flags)

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

# Network weights (ml/train/model.py) and morph profiles (ml/data/build_morph.py). Without them Live/Studio
# run DSP and Morph does nothing.
file(GLOB LARYNX_MODEL_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/neural/model_*.json
    ${CMAKE_CURRENT_SOURCE_DIR}/source/neural/profiles.data)
if(LARYNX_MODEL_FILES)
    juce_add_binary_data(LarynxModels SOURCES ${LARYNX_MODEL_FILES})
    set_target_properties(LarynxModels PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(Larynx PRIVATE LarynxModels)
    target_compile_definitions(Larynx PRIVATE LARYNX_MODELS=1)
endif()

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
        larynx_core
        juce::juce_audio_utils
        juce::juce_opengl
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)
