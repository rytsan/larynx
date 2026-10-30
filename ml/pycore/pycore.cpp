// pybind11 bridge over plugin/source/core — the only source of training features (§9.3).
//   import pycore; f = pycore.analyze(x_float32, fs)
// Runs the plugin's Engine sample by sample (identity shape) and returns one row per hop:
// f0, confidence, flags, n_valid (n,), h, hz (n, 80), noise (n, 8). Same frames as larynx_render.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <memory>
#include <vector>
#include "core/engine.h"

namespace py = pybind11;

namespace {

template <class T>
py::array_t<T> toArray(std::vector<T>& v, std::vector<py::ssize_t> shape) {
    auto* p = new std::vector<T>(std::move(v));
    py::capsule owner(p, [](void* q) { delete static_cast<std::vector<T>*>(q); });
    return py::array_t<T>(shape, p->data(), owner);
}

py::dict analyze(py::array_t<float, py::array::c_style | py::array::forcecast> x, double fs) {
    if (x.ndim() != 1) throw std::invalid_argument("x must be 1-D (mono)");
    if (fs != 44100.0 && fs != 48000.0) throw std::invalid_argument("fs must be 44100 or 48000 (§2.7)");
    auto e = std::make_unique<larynx::Engine>();
    e->prepare(fs);
    std::vector<float> f0, conf, h, hz, noise;
    std::vector<std::uint8_t> flags, nValid;
    const float* p = x.data();
    {
        py::gil_scoped_release nogil;
        for (py::ssize_t n = 0; n < x.shape(0); ++n)
            e->process(p[n], [&](const larynx::HarmFrame& f, larynx::HarmFrame&) {
                f0.push_back(f.f0);
                conf.push_back(f.confidence);
                flags.push_back(f.flags);
                nValid.push_back(f.nValid);
                h.insert(h.end(), f.h, f.h + larynx::kNumHarmonics);
                hz.insert(hz.end(), f.hz, f.hz + larynx::kNumHarmonics);
                noise.insert(noise.end(), f.noise, f.noise + larynx::kNumNoiseBands);
            });
    }
    const py::ssize_t n = (py::ssize_t) f0.size();
    py::dict d;
    d["f0"] = toArray(f0, {n});
    d["confidence"] = toArray(conf, {n});
    d["flags"] = toArray(flags, {n});
    d["n_valid"] = toArray(nValid, {n});
    d["h"] = toArray(h, {n, larynx::kNumHarmonics});
    d["hz"] = toArray(hz, {n, larynx::kNumHarmonics});
    d["noise"] = toArray(noise, {n, larynx::kNumNoiseBands});
    d["hop"] = e->skeleton.hop();
    return d;
}

} // namespace

PYBIND11_MODULE(pycore, m) {
    m.doc() = "LARYNX core (analyzer + skeleton) for the ML pipeline";
    m.def("analyze", &analyze, py::arg("x"), py::arg("fs"));
    m.attr("VOICED") = (int) larynx::kVoiced;
    m.attr("SIBILANT") = (int) larynx::kSibilant;
    m.attr("TRANSIENT") = (int) larynx::kTransient;
    m.attr("FLOOR_DB") = larynx::kFloorDb;
}
