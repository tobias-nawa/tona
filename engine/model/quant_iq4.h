// IQ4_NL quantizer (ggml type 20) for host code that builds a truncated copy of a weight, e.g. the MTP head's rows.
// Header-only on purpose: it takes the floating-point contraction flags of the translation unit that includes it, as
// the copy it replaced did. No backend dependency.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace tona::iq4 {

inline constexpr int8_t kIq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline int nearest_iq4(float v) {
    int best = 0;
    for (int k = 1; k < 16; k++)
        if (std::fabs(v - kIq4[k]) < std::fabs(v - kIq4[best])) best = k;
    return best;
}

// One candidate scale for a block of 32: indices q and the least-squares refit of the scale; returns the squared error
inline float iq4_candidate(const float* v, float id, uint8_t* q, float& dd) {
    double xl = 0, ll = 0;
    for (int j = 0; j < 32; j++) {
        q[j] = (uint8_t)nearest_iq4(v[j] * id);
        xl += v[j] * kIq4[q[j]], ll += kIq4[q[j]] * kIq4[q[j]];
    }
    dd = (float)(_Float16)(ll > 0 ? xl / ll : 0);
    float e = 0;
    for (int j = 0; j < 32; j++) {
        const float r = v[j] - kIq4[q[j]] * dd;
        e += r * r;
    }
    return e;
}

inline void quantize_iq4_block(const float* v, uint8_t* o) {
    float amax = 0, mx = 0;
    for (int j = 0; j < 32; j++)
        if (std::fabs(v[j]) > amax) amax = std::fabs(v[j]), mx = v[j];
    uint8_t q[32] = {}, bq[32] = {};
    float best_d = 0, best_e = INFINITY;
    for (int k = -7; k <= 7 && amax > 0; k++) {
        float dd;
        const float e = iq4_candidate(v, (k + kIq4[0]) / mx, q, dd);
        if (e < best_e) best_e = e, best_d = dd, std::memcpy(bq, q, 32);
    }
    const _Float16 h = (_Float16)best_d;
    std::memcpy(o, &h, 2);
    for (int j = 0; j < 16; j++) o[2 + j] = (uint8_t)(bq[j] | bq[j + 16] << 4);
}

// IQ4_NL (type 20, 32 values: f16 d, 16 bytes; element j in the low nibble, j + 16 in the high one, value kv[q] * d):
// scale like ggml (candidates id = (k + kv[0]) / max with the sign of max, each with a least-squares refit)
inline void quantize_iq4_nl(const float* x, uint8_t* out, int64_t n) {
    for (int64_t b = 0; b < n / 32; b++) quantize_iq4_block(x + b * 32, out + b * 18);
}

}  // namespace tona::iq4
