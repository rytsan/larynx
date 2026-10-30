// Headless DSP-mode render: in.wav → out.wav, latency removed so out lines up with in (for ABX).
//   larynx_render in.wav out.wav [--bright dB/oct] [--air dB] [--evenodd -1..1] [--formant st] [--mix 0..1]
//                [--sub 0..1] [--breath 0..1]
// Same M4 blend as the plugin (core/blend.h), parameters in plugin units, Prior irrelevant (no network).
// Without options it is the P3 identity path: prints max error vs. input and LSD H1–H40.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "core/blend.h"
#include "core/engine.h"
#include "wav.h"

namespace {

std::vector<larynx::HarmFrame> analyzeAll(const std::vector<float>& x, double fs) {
    larynx::Engine e;
    e.prepare(fs);
    std::vector<larynx::HarmFrame> fr;
    for (float v : x) e.process(v, [&](const larynx::HarmFrame& in, larynx::HarmFrame&) { fr.push_back(in); });
    return fr;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s in.wav out.wav [--bright dB/oct] [--air dB] [--evenodd -1..1] [--formant st] [--mix 0..1] [--sub 0..1] [--breath 0..1]\n", argv[0]);
        return 2;
    }
    larynx::Controls c;
    c.prior = 0.0f;
    for (int i = 3; i + 1 < argc; i += 2) {
        const float v = std::strtof(argv[i + 1], nullptr);
        if (!std::strcmp(argv[i], "--bright")) c.bright = v;
        else if (!std::strcmp(argv[i], "--air")) c.air = v;
        else if (!std::strcmp(argv[i], "--evenodd")) c.evenodd = v;
        else if (!std::strcmp(argv[i], "--formant")) c.formant = v;
        else if (!std::strcmp(argv[i], "--mix")) c.mix = v;
        else if (!std::strcmp(argv[i], "--sub")) c.sub = v;
        else if (!std::strcmp(argv[i], "--breath")) c.breath = v;
        else { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    std::vector<float> x;
    double fs = 0.0;
    if (!readWav(argv[1], x, fs)) {
        std::fprintf(stderr, "cannot read %s (PCM 16/24/32 or float32 WAV)\n", argv[1]);
        return 1;
    }

    larynx::Engine e;
    e.prepare(fs);
    e.synth.setSub(c.sub);
    const size_t L = (size_t) e.latency();
    auto shape = [&](const larynx::HarmFrame& in, larynx::HarmFrame& t) { larynx::blend(in, t, c, fs); };
    std::vector<float> y;
    y.reserve(x.size() + L);
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t n = 0; n < x.size() + L; ++n) y.push_back(e.process(n < x.size() ? x[n] : 0.0f, shape));
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    y.erase(y.begin(), y.begin() + (long) L);

    if (!writeWav(argv[2], y, fs)) {
        std::fprintf(stderr, "cannot write %s\n", argv[2]);
        return 1;
    }
    std::fprintf(stderr, "audio %.2f s, engine %.3f s, RTF %.4f\n", (double) x.size() / fs, secs, secs * fs / (double) x.size());

    float worst = 0.0f;
    for (size_t n = 0; n < x.size(); ++n) worst = std::max(worst, std::abs(y[n] - x[n]));
    // LSD vs. the requested envelope (input reshaped by the same ops; = input at identity), H1–H40,
    // voiced frames. "all": harmonics present (> −90 dB) in input and target. "strong": both within
    // 40 dB of the frame's loudest harmonic — a mask cannot push a harmonic under the noise in its bins.
    const auto a = analyzeAll(x, fs), b = analyzeAll(y, fs);
    double sum[2] = {}, cnt[2] = {};
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (!(a[i].flags & larynx::kVoiced) || std::abs(a[i].f0 - b[i].f0) > 0.01f * a[i].f0) continue;
        auto t = a[i];
        shape(a[i], t);
        const int nk = std::min<int>(40, a[i].nValid);
        const float loudest = *std::max_element(a[i].h, a[i].h + nk);
        for (int k = 0; k < nk; ++k) {
            if (a[i].h[k] <= -90.0f || t.h[k] <= -90.0f) continue;
            const double d2 = (double) (t.h[k] - b[i].h[k]) * (t.h[k] - b[i].h[k]);
            const bool strong = a[i].h[k] > loudest - 40.0f && t.h[k] > loudest - 40.0f;
            for (int c = 0; c < (strong ? 2 : 1); ++c) {
                sum[c] += d2;
                cnt[c] += 1.0;
            }
        }
    }
    const auto lsd = [&](int c) { return cnt[c] > 0.0 ? std::sqrt(sum[c] / cnt[c]) : 0.0; };
    std::fprintf(stderr, "max |out - in| %.1f dBFS, LSD(H1-H40) vs target: all %.3f dB, strong %.3f dB\n",
                 20.0 * std::log10(worst + 1e-30), lsd(0), lsd(1));
}
