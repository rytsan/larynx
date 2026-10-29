#pragma once
// In-place radix-2 complex FFT. No JUCE (core/ rule). Allocates only in prepare().
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <utility>
#include <vector>

namespace larynx {

class Fft {
public:
    using cf = std::complex<float>;

    void prepare(std::size_t n) { // n must be a power of two
        n_ = n;
        rev_.resize(n);
        for (std::size_t i = 0, bits = (std::size_t) std::countr_zero(n); i < n; ++i) {
            std::size_t r = 0;
            for (std::size_t b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
            rev_[i] = r;
        }
        tw_.resize(n / 2);
        for (std::size_t k = 0; k < n / 2; ++k)
            tw_[k] = std::polar(1.0f, (float) (-2.0 * std::numbers::pi * (double) k / (double) n));
    }
    std::size_t size() const { return n_; }

    // Forward transform (e^{-i...}), unscaled.
    void forward(cf* x) const {
        for (std::size_t i = 0; i < n_; ++i)
            if (i < rev_[i]) std::swap(x[i], x[rev_[i]]);
        for (std::size_t len = 2; len <= n_; len <<= 1) {
            const std::size_t half = len / 2, step = n_ / len;
            for (std::size_t i = 0; i < n_; i += len)
                for (std::size_t k = 0; k < half; ++k) {
                    const cf t = x[i + k + half] * tw_[k * step];
                    x[i + k + half] = x[i + k] - t;
                    x[i + k] += t;
                }
        }
    }

private:
    std::size_t n_ = 0;
    std::vector<std::size_t> rev_;
    std::vector<cf> tw_;
};

} // namespace larynx
