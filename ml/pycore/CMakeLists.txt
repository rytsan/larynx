# Python module over core/ (§9.3). Built into build/ml/pycore; use with PYTHONPATH=build/ml/pycore.
find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(pycore pycore.cpp)
target_link_libraries(pycore PRIVATE larynx_core)
