// Headless analyzer run over a WAV: per-frame CSV on stdout, RTF on stderr.
//   larynx_analyze in.wav > frames.csv
// Used for the P2 gates on the golden set (F0 stability, flag annotation) and RTF(analyzer).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include "core/analyzer.h"

namespace {

// Minimal RIFF reader: mono/multi PCM 16/24/32 or float32; returns channel 0.
bool readWav(const char* path, std::vector<float>& out, double& fs) {
    std::ifstream f(path, std::ios::binary);
    char id[4];
    std::uint32_t size = 0;
    if (!f.read(id, 4) || std::memcmp(id, "RIFF", 4) != 0) return false;
    f.ignore(4);
    if (!f.read(id, 4) || std::memcmp(id, "WAVE", 4) != 0) return false;
    std::uint16_t fmt = 0, channels = 0, bits = 0;
    std::uint32_t rate = 0;
    while (f.read(id, 4) && f.read(reinterpret_cast<char*>(&size), 4)) {
        if (std::memcmp(id, "fmt ", 4) == 0) {
            std::vector<char> c(size);
            f.read(c.data(), size);
            std::memcpy(&fmt, &c[0], 2);
            std::memcpy(&channels, &c[2], 2);
            std::memcpy(&rate, &c[4], 4);
            std::memcpy(&bits, &c[14], 2);
            if (fmt == 0xFFFE && size >= 26) std::memcpy(&fmt, &c[24], 2); // WAVE_FORMAT_EXTENSIBLE
        } else if (std::memcmp(id, "data", 4) == 0) {
            if (!channels || !bits) return false;
            std::vector<unsigned char> d(size);
            f.read(reinterpret_cast<char*>(d.data()), size);
            const size_t bps = bits / 8, frames = size / (bps * channels);
            out.resize(frames);
            for (size_t i = 0; i < frames; ++i) {
                const unsigned char* p = &d[i * bps * channels];
                if (fmt == 3 && bits == 32) std::memcpy(&out[i], p, 4);
                else if (bits == 16) out[i] = (float) (std::int16_t) (p[0] | p[1] << 8) / 32768.0f;
                else if (bits == 24) out[i] = (float) ((std::int32_t) ((std::uint32_t) p[0] << 8 | (std::uint32_t) p[1] << 16 | (std::uint32_t) p[2] << 24) >> 8) / 8388608.0f;
                else if (bits == 32) out[i] = (float) ((double) (std::int32_t) (p[0] | p[1] << 8 | p[2] << 16 | (std::uint32_t) p[3] << 24) / 2147483648.0);
                else return false;
            }
            fs = rate;
            return true;
        } else {
            f.ignore(size + (size & 1));
        }
    }
    return false;
}

} // namespace

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
