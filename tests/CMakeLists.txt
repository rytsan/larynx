FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.16.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Catch2)

add_executable(larynx_tests test_processor.cpp test_analyzer.cpp test_engine.cpp test_blend.cpp test_neural.cpp)
set_source_files_properties(test_blend.cpp test_neural.cpp PROPERTIES COMPILE_OPTIONS -Wno-float-equal)

# Reuse the plugin's shared-code static lib (JUCE modules already compiled into it).
target_include_directories(larynx_tests PRIVATE $<TARGET_PROPERTY:Larynx,INCLUDE_DIRECTORIES>)
target_compile_definitions(larynx_tests PRIVATE $<TARGET_PROPERTY:Larynx,COMPILE_DEFINITIONS>)
# Bit-exact comparisons are the point of these tests.
set_source_files_properties(test_processor.cpp PROPERTIES COMPILE_OPTIONS -Wno-float-equal)
target_link_libraries(larynx_tests PRIVATE Larynx larynx_core Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(larynx_tests)
