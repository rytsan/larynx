// Headless analyzer run over a WAV: per-frame CSV on stdout, RTF on stderr.
//   larynx_analyze in.wav > frames.csv
// Used for the P2 gates on the golden set (F0 stability, flag annotation) and RTF(analyzer).
#include <chrono>
#include <cstdio>
#include <vector>
#include "core/analyzer.h"
#include "wav.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s in.wav > frames.csv\n", argv[0]);
        return 2;
    }
    std::vector<float> x;
    double fs = 0.0;
    if (!readWav(argv[1], x, fs)) {
        std::fprintf(stderr, "cannot read %s (PCM 16/24/32 or float32 WAV)\n", argv[1]);
        return 1;
    }

    larynx::Analyzer a;
    a.prepare(fs);
    larynx::HarmFrame f;
    std::vector<larynx::HarmFrame> frames;
    frames.reserve(x.size() / (size_t) a.hop() + 1);

    const auto t0 = std::chrono::steady_clock::now();
    for (float v : x)
        if (a.push(v, f)) frames.push_back(f);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("time,f0,confidence,voiced,sibilant,transient\n");
    for (size_t i = 0; i < frames.size(); ++i) {
        const auto& fr = frames[i];
        // Window end of frame i is sample (i+1)*hop - 1.
        std::printf("%.4f,%.2f,%.3f,%d,%d,%d\n", (double) ((i + 1) * (size_t) a.hop()) / fs, fr.f0, fr.confidence,
                    !!(fr.flags & larynx::kVoiced), !!(fr.flags & larynx::kSibilant), !!(fr.flags & larynx::kTransient));
    }
    std::fprintf(stderr, "%zu frames, audio %.2f s, analyzer %.3f s, RTF %.4f\n", frames.size(),
                 (double) x.size() / fs, secs, secs * fs / (double) x.size());
}
