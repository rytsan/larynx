foreach(tool analyze render)
    add_executable(larynx_${tool} ${tool}.cpp)
    target_link_libraries(larynx_${tool} PRIVATE larynx_core juce::juce_recommended_config_flags)
endforeach()
