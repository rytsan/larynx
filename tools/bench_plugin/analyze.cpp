// Headless analyzer run over a WAV: per-frame CSV on stdout, RTF on stderr.
//   larynx_analyze in.wav [--harm] > frames.csv
// Used for the P2 gates on the golden set (F0 stability, flag annotation) and RTF(analyzer).
// --harm runs the whole Engine (identity) and appends h1..h80 (dB) — reference for the ml/pycore
// equivalence gate (§9); RTF is then the engine's.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include "core/engine.h"
#include "wav.h"

int main(int argc, char** argv) {
    const bool harm = argc == 3 && !std::strcmp(argv[2], "--harm");
    if (argc != 2 && !harm) {
        std::fprintf(stderr, "usage: %s in.wav [--harm] > frames.csv\n", argv[0]);
        return 2;
    }
    std::vector<float> x;
    double fs = 0.0;
    if (!readWav(argv[1], x, fs)) {
        std::fprintf(stderr, "cannot read %s (PCM 16/24/32 or float32 WAV)\n", argv[1]);
        return 1;
    }

    larynx::Engine e;
    e.prepare(fs);
    auto& a = e.analyzer;
    larynx::HarmFrame f;
    std::vector<larynx::HarmFrame> frames;
    frames.reserve(x.size() / (size_t) a.hop() + 1);

    const auto t0 = std::chrono::steady_clock::now();
    if (harm)
        for (float v : x) e.process(v, [&](const larynx::HarmFrame& in, larynx::HarmFrame&) { frames.push_back(in); });
    else
        for (float v : x)
            if (a.push(v, f)) frames.push_back(f);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("time,f0,confidence,voiced,sibilant,transient");
    if (harm)
        for (int k = 1; k <= larynx::kNumHarmonics; ++k) std::printf(",h%d", k);
    std::printf("\n");
    for (size_t i = 0; i < frames.size(); ++i) {
        const auto& fr = frames[i];
        // Window end of frame i is sample (i+1)*hop - 1.
        std::printf("%.4f,%.2f,%.3f,%d,%d,%d", (double) ((i + 1) * (size_t) a.hop()) / fs, fr.f0, fr.confidence,
                    !!(fr.flags & larynx::kVoiced), !!(fr.flags & larynx::kSibilant), !!(fr.flags & larynx::kTransient));
        if (harm)
            for (float v : fr.h) std::printf(",%.4f", v);
        std::printf("\n");
    }
    std::fprintf(stderr, "%zu frames, audio %.2f s, analyzer %.3f s, RTF %.4f\n", frames.size(),
                 (double) x.size() / fs, secs, secs * fs / (double) x.size());
}
