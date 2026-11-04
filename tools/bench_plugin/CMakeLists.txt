foreach(tool analyze render)
    add_executable(larynx_${tool} ${tool}.cpp)
    target_link_libraries(larynx_${tool} PRIVATE larynx_core juce::juce_recommended_config_flags)
endforeach()

# Render through the real processor (neural modes, presets): reuses the plugin's shared-code lib like the tests.
add_executable(larynx_render_plugin render_plugin.cpp)
target_include_directories(larynx_render_plugin PRIVATE $<TARGET_PROPERTY:Larynx,INCLUDE_DIRECTORIES>)
target_compile_definitions(larynx_render_plugin PRIVATE $<TARGET_PROPERTY:Larynx,COMPILE_DEFINITIONS>)
target_link_libraries(larynx_render_plugin PRIVATE Larynx larynx_core)
