cmake_minimum_required(VERSION 3.24)
project(Larynx VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Pinned by commit: JUCE 8.0.16-security, clap-juce-extensions main @ 2026-10-06.
FetchContent_Declare(juce
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG ea77444750de45e036e0279bb0291144bf8e3688
    GIT_SHALLOW FALSE)
FetchContent_Declare(clap_juce_extensions
    GIT_REPOSITORY https://github.com/free-audio/clap-juce-extensions.git
    GIT_TAG 7adee3a1bd4684d4caa5601100e364abccff4b4d
    GIT_SUBMODULES_RECURSE TRUE)
FetchContent_MakeAvailable(juce clap_juce_extensions)

add_subdirectory(plugin)
add_subdirectory(tools/bench_plugin)

option(LARYNX_PYCORE "Build the pybind11 core module for ml/" ON)
if(LARYNX_PYCORE)
    add_subdirectory(ml/pycore)
endif()

option(LARYNX_TESTS "Build tests" ON)
if(LARYNX_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
