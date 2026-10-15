#pragma once
// Fixed integer delay. No JUCE (core/ rule). Allocates only in resize().
#include <algorithm>
#include <cstddef>
#include <vector>

namespace larynx {

class DelayLine {
public:
    void resize(std::size_t delaySamples) {
        buf_.assign(delaySamples, 0.0f);
        pos_ = 0;
    }
    void reset() { std::fill(buf_.begin(), buf_.end(), 0.0f); pos_ = 0; }
    std::size_t delay() const { return buf_.size(); }

    float process(float x) {
        if (buf_.empty()) return x;
        float y = buf_[pos_];
        buf_[pos_] = x;
        if (++pos_ == buf_.size()) pos_ = 0;
        return y;
    }

private:
    std::vector<float> buf_;
    std::size_t pos_ = 0;
};

} // namespace larynx
