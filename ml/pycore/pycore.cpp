// pybind11 bridge over plugin/source/core — the only source of training features (§9.3).
//   import pycore; f = pycore.analyze(x_float32, fs); x = pycore.net_input(f)
// analyze: runs the plugin's Engine sample by sample (identity shape) and returns one row per hop:
// f0, confidence, flags, n_valid (n,), h, hz (n, 80), noise (n, 8). Same frames as larynx_render.
// net_input: the 51-channel network input of core/features.h for each row of such a dict
// (masks edit h/noise/flags in Python first).
// tcn_run: core/neuralcore over a (T, 51) input with weights as exported by ml/train/model.py
// (the json.load()ed dict) → (T, 88). Export equivalence test (§9.6).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <vector>
#include "core/engine.h"
#include "core/blend.h"
#include "core/features.h"
#include "core/neuralcore.h"

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

template <class T>
py::array_t<T, py::array::c_style | py::array::forcecast> col(const py::dict& d, const char* k, py::ssize_t n, py::ssize_t w) {
    auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(d[k]);
    if (!a || a.shape(0) != n || (w ? a.ndim() != 2 || a.shape(1) != w : a.ndim() != 1))
        throw std::invalid_argument(std::string("bad shape for ") + k);
    return a;
}

py::array_t<float> netInput(const py::dict& d) {
    const py::ssize_t n = py::array::ensure(d["f0"]).shape(0);
    auto f0 = col<float>(d, "f0", n, 0);
    auto flags = col<std::uint8_t>(d, "flags", n, 0);
    auto nValid = col<std::uint8_t>(d, "n_valid", n, 0);
    auto h = col<float>(d, "h", n, larynx::kNumHarmonics);
    auto noise = col<float>(d, "noise", n, larynx::kNumNoiseBands);
    py::array_t<float> out({n, (py::ssize_t) larynx::kNetInputs});
    float* o = out.mutable_data();
    larynx::HarmFrame f {};
    for (py::ssize_t i = 0; i < n; ++i) {
        f.f0 = f0.data()[i];
        f.flags = flags.data()[i];
        f.nValid = nValid.data()[i];
        std::copy_n(h.data() + i * larynx::kNumHarmonics, larynx::kNumHarmonics, f.h);
        std::copy_n(noise.data() + i * larynx::kNumNoiseBands, larynx::kNumNoiseBands, f.noise);
        larynx::netInput(f, o + i * larynx::kNetInputs);
    }
    return out;
}

std::vector<float> flat(const py::handle& o) {
    std::vector<float> v;
    if (py::isinstance<py::list>(o)) {
        for (auto e : o.cast<py::list>()) {
            if (py::isinstance<py::list>(e)) {
                auto sub = flat(e);
                v.insert(v.end(), sub.begin(), sub.end());
            } else {
                v.push_back(e.cast<float>());
            }
        }
    } else {
        v.push_back(o.cast<float>());
    }
    return v;
}

larynx::TcnWeights weightsFrom(const py::dict& j) {
    larynx::TcnWeights w;
    w.channels = j["n_channels"].cast<int>();
    w.kernel = j["kernel"].cast<int>();
    w.delay = j["D"].cast<int>();
    w.dilations = j["dilations"].cast<std::vector<int>>();
    for (auto l : j["layers"].cast<py::list>()) {
        auto d = l.cast<py::dict>();
        const auto name = d["name"].cast<std::string>();
        auto W = flat(d["w"]), B = flat(d["b"]);
        if (name == "in") w.inW = W, w.inB = B;
        else if (name == "out") w.outW = W, w.outB = B;
        else w.convW.push_back(W), w.convB.push_back(B);
    }
    return w;
}

py::array_t<float> tcnRun(const py::dict& j, py::array_t<float, py::array::c_style | py::array::forcecast> x) {
    if (j["contract"].cast<int>() != larynx::kNetContract) throw std::invalid_argument("network contract mismatch");
    if (x.ndim() != 2 || x.shape(1) != larynx::kNetInputs) throw std::invalid_argument("x must be (T, 51)");
    larynx::NeuralCore nc;
    if (!nc.load(weightsFrom(j))) throw std::invalid_argument("inconsistent weight shapes");
    const py::ssize_t T = x.shape(0);
    py::array_t<float> y({T, (py::ssize_t) larynx::kNetOutputs});
    for (py::ssize_t t = 0; t < T; ++t) nc.step(x.data() + t * larynx::kNetInputs, y.mutable_data() + t * larynx::kNetOutputs);
    return y;
}

py::tuple vowelOf(const py::dict& d) {
    // core/blend.h vowel reader per row: (vowel char code or 0, F1, F2)
    const py::ssize_t n = py::array::ensure(d["f0"]).shape(0);
    auto f0 = col<float>(d, "f0", n, 0);
    auto flags = col<std::uint8_t>(d, "flags", n, 0);
    auto nValid = col<std::uint8_t>(d, "n_valid", n, 0);
    auto h = col<float>(d, "h", n, larynx::kNumHarmonics);
    py::array_t<std::uint8_t> v(n);
    py::array_t<float> f1(n), f2(n);
    larynx::HarmFrame f {};
    for (py::ssize_t i = 0; i < n; ++i) {
        f.f0 = f0.data()[i];
        f.flags = flags.data()[i];
        f.nValid = nValid.data()[i];
        std::copy_n(h.data() + i * larynx::kNumHarmonics, larynx::kNumHarmonics, f.h);
        float a = 0.0f, b = 0.0f;
        v.mutable_data()[i] = (std::uint8_t) larynx::vowelOf(f, &a, &b);
        f1.mutable_data()[i] = a;
        f2.mutable_data()[i] = b;
    }
    return py::make_tuple(v, f1, f2);
}

} // namespace

PYBIND11_MODULE(pycore, m) {
    m.doc() = "LARYNX core (analyzer + skeleton) for the ML pipeline";
    m.def("analyze", &analyze, py::arg("x"), py::arg("fs"));
    m.def("net_input", &netInput, py::arg("frames"));
    m.def("tcn_run", &tcnRun, py::arg("weights"), py::arg("x"));
    m.def("vowel_of", &vowelOf, py::arg("frames"));
    m.attr("NET_CONTRACT") = larynx::kNetContract;
    m.attr("NET_INPUTS") = larynx::kNetInputs;
    m.attr("NET_OUTPUTS") = larynx::kNetOutputs;
    m.attr("NET_HARMONICS") = larynx::kNetHarmonics;
    m.attr("VOICED") = (int) larynx::kVoiced;
    m.attr("SIBILANT") = (int) larynx::kSibilant;
    m.attr("TRANSIENT") = (int) larynx::kTransient;
    m.attr("FLOOR_DB") = larynx::kFloorDb;
}
