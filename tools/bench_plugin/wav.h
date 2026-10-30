#pragma once
// Minimal WAV I/O for the headless tools (no JUCE: tools build against core/ only).
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

// Minimal RIFF reader: mono/multi PCM 16/24/32 or float32; returns channel 0.
inline bool readWav(const char* path, std::vector<float>& out, double& fs) {
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


// Mono float32 WAV.
inline bool writeWav(const char* path, const std::vector<float>& x, double fs) {
    std::ofstream f(path, std::ios::binary);
    const auto u32 = [&](std::uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    const auto u16 = [&](std::uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    const auto bytes = (std::uint32_t) (x.size() * 4);
    f.write("RIFF", 4); u32(36 + bytes); f.write("WAVE", 4);
    f.write("fmt ", 4); u32(16); u16(3); u16(1); u32((std::uint32_t) fs); u32((std::uint32_t) fs * 4); u16(4); u16(32);
    f.write("data", 4); u32(bytes);
    f.write(reinterpret_cast<const char*>(x.data()), bytes);
    return (bool) f;
}
