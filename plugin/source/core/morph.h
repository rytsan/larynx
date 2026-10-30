#pragma once
// M5 Morph v2 (§12): pulls the harmonic envelope towards a singer profile built by
// ml/data/build_morph.py (mean dB envelope per vowel × 2-semitone pitch bin, PCA space).
// The frame's envelope (H1..H40 − rms_db) is projected on the PCA components, slerped towards the
// profile's coefficients for the frame's vowel and pitch (nearest valid bin, "any vowel" fallback),
// and rebuilt keeping the part of the envelope outside the PCA subspace. No JUCE (core/ rule).
// load() allocates; apply() is real-time safe.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "blend.h"
#include "features.h"

namespace larynx {

class MorphProfiles {
public:
    bool load(const void* data, std::size_t size) {
        nSingers_ = 0;
        const auto* p = static_cast<const std::uint8_t*>(data);
        const auto* end = p + size;
        auto take = [&](void* dst, std::size_t n) {
            if ((std::size_t) (end - p) < n) return false;
            std::memcpy(dst, p, n);
            p += n;
            return true;
        };
        char magic[4];
        std::uint32_t hdr[6];
        float bins[2];
        if (!take(magic, 4) || std::memcmp(magic, "LXMP", 4) != 0 || !take(hdr, sizeof hdr) || !take(bins, sizeof bins)) return false;
        if (hdr[0] != 1 || hdr[1] != (std::uint32_t) kNetHarmonics || hdr[2] == 0 || hdr[2] > 64 || hdr[3] != 6 || hdr[4] == 0
            || hdr[4] > 64 || hdr[5] == 0 || hdr[5] > 256)
            return false;
        nComp_ = (int) hdr[2], nVowels_ = (int) hdr[3], nBins_ = (int) hdr[4];
        const int ns = (int) hdr[5];
        bin0Hz_ = bins[0], binSt_ = bins[1];
        mean_.resize(kNetHarmonics);
        comp_.resize((std::size_t) (nComp_ * kNetHarmonics));
        names_.assign((std::size_t) ns, {});
        coef_.resize((std::size_t) (ns * nVowels_ * nBins_ * nComp_));
        valid_.resize((std::size_t) (ns * nVowels_ * nBins_));
        if (!take(mean_.data(), mean_.size() * 4) || !take(comp_.data(), comp_.size() * 4)) return false;
        for (auto& n : names_) {
            char buf[32];
            if (!take(buf, 32)) return false;
            buf[31] = 0;
            n = buf;
        }
        if (!take(coef_.data(), coef_.size() * 4) || !take(valid_.data(), valid_.size())) return false;
        z_.resize((std::size_t) nComp_);
        zp_.resize((std::size_t) nComp_);
        nSingers_ = ns;
        return true;
    }

    int singers() const { return nSingers_; }
    const std::string& name(int i) const { return names_[(std::size_t) i]; }

    // Morphs t (in place) towards singer `s` by `amount` 0..1. Only harmonics present in t are touched.
    void apply(HarmFrame& t, int s, float amount) const {
        if (nSingers_ == 0 || amount <= 0.0f || s < 0 || s >= nSingers_ || !(t.flags & kVoiced) || t.f0 <= 0.0f) return;
        static constexpr char kVowels[] = "aeiou";
        const char v = vowelOf(t);
        int vi = nVowels_ - 1;
        for (int i = 0; i < 5; ++i)
            if (kVowels[i] == v) vi = i;
        const int b = (int) std::lround(12.0f * std::log2(t.f0 / bin0Hz_) / binSt_);
        const float* zp = find(s, vi, b);
        if (!zp && vi != nVowels_ - 1) zp = find(s, nVowels_ - 1, b);
        if (!zp) return;

        const float rms = netRmsDb(t);
        const int nh = std::min<int>(kNetHarmonics, t.nValid);
        float e[kNetHarmonics];
        for (int i = 0; i < kNetHarmonics; ++i) e[i] = i < nh && t.h[i] > kFloorDb ? std::max(t.h[i] - rms, -80.0f) : -80.0f;
        for (int c = 0; c < nComp_; ++c) {
            float acc = 0.0f;
            for (int i = 0; i < kNetHarmonics; ++i) acc += comp_[(std::size_t) (c * kNetHarmonics + i)] * (e[i] - mean_[(std::size_t) i]);
            z_[(std::size_t) c] = acc;
        }
        slerp(z_.data(), zp, amount, zp_.data());
        for (int i = 0; i < nh; ++i) {
            if (t.h[i] <= kFloorDb) continue;
            float d = 0.0f;
            for (int c = 0; c < nComp_; ++c) d += comp_[(std::size_t) (c * kNetHarmonics + i)] * (zp_[(std::size_t) c] - z_[(std::size_t) c]);
            t.h[i] = std::clamp(t.h[i] + d, kFloorDb, 0.0f);
        }
    }

private:
    // Nearest valid pitch bin of (singer, vowel), searching outwards up to ±3 bins (±6 st).
    const float* find(int s, int v, int b) const {
        for (int d = 0; d <= 3; ++d)
            for (int bb : { b - d, b + d })
                if (bb >= 0 && bb < nBins_ && valid_[(std::size_t) ((s * nVowels_ + v) * nBins_ + bb)])
                    return &coef_[(std::size_t) (((s * nVowels_ + v) * nBins_ + bb) * nComp_)];
        return nullptr;
    }

    // Spherical interpolation of direction, linear of length (the two norms differ).
    void slerp(const float* a, const float* b, float t, float* out) const {
        float na = 0.0f, nb = 0.0f, dot = 0.0f;
        for (int c = 0; c < nComp_; ++c) na += a[c] * a[c], nb += b[c] * b[c], dot += a[c] * b[c];
        na = std::sqrt(na), nb = std::sqrt(nb);
        const float len = na + t * (nb - na);
        const float cosw = na > 1e-6f && nb > 1e-6f ? std::clamp(dot / (na * nb), -1.0f, 1.0f) : 1.0f;
        const float w = std::acos(cosw);
        if (w < 1e-3f || na < 1e-6f || nb < 1e-6f) {
            for (int c = 0; c < nComp_; ++c) out[c] = a[c] + t * (b[c] - a[c]);
            return;
        }
        const float sa = std::sin((1.0f - t) * w) / std::sin(w), sb = std::sin(t * w) / std::sin(w);
        float n = 0.0f;
        for (int c = 0; c < nComp_; ++c) out[c] = sa * a[c] / na + sb * b[c] / nb, n += out[c] * out[c];
        n = std::sqrt(std::max(n, 1e-12f));
        for (int c = 0; c < nComp_; ++c) out[c] *= len / n;
    }

    int nSingers_ = 0, nComp_ = 0, nVowels_ = 0, nBins_ = 0;
    float bin0Hz_ = 55.0f, binSt_ = 2.0f;
    std::vector<float> mean_, comp_, coef_;
    mutable std::vector<float> z_, zp_; // scratch (one audio thread)
    std::vector<std::uint8_t> valid_;
    std::vector<std::string> names_;
};

} // namespace larynx
