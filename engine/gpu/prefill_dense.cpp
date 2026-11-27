// PrefillRun: matmuls on the dense weights (unpack to f16, then GEMM; optionally FP8) and the gated residual.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <map>
#include <sstream>
#include <string>

#include "backend/env.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_run.h"

namespace tona::gpu {

namespace {

// Diagnostics, both off by default. TONA_PF_FP8_SKIP=rows:in[:type][,...] keeps the dense weights of these shapes on
// the f16 GEMM; TONA_PF_FP8_STATS=1 prints, per shape, how well the FP8 copy of the input (factor per token) represents
// the f16 input: relative error and share of values flushed to zero over a sample of tokens, and the largest absolute
// value
struct SkipShape {
    int64_t rows, in, type;  // type -1: any
};

bool fp8_skipped(const PfW& w) {
    static const std::vector<SkipShape> skip = [] {
        std::vector<SkipShape> v;
        std::stringstream list(tona::env::text("TONA_PF_FP8_SKIP") ? tona::env::text("TONA_PF_FP8_SKIP") : "");
        for (std::string item; std::getline(list, item, ',');) {
            const size_t c = item.find(':'), c2 = item.find(':', c + 1);
            if (c == std::string::npos) continue;
            v.push_back({std::stoll(item.substr(0, c)), std::stoll(item.substr(c + 1, c2 - c - 1)),
                         c2 == std::string::npos ? -1 : std::stoll(item.substr(c2 + 1))});
        }
        return v;
    }();
    return std::any_of(skip.begin(), skip.end(), [&](const SkipShape& s) {
        return s.rows == (int64_t)w.rows && s.in == (int64_t)w.in && (s.type < 0 || s.type == (int64_t)w.type);
    });
}

// E4M3 (OCP, bias 7, no infinity) to float
float e4m3_value(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? std::ldexp(m / 8.0f, -6) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return b & 0x80 ? -v : v;
}

struct Fp8Stat {
    int64_t calls = 0, rows = 0;
    double err_sum = 0, err_max = 0, flush_sum = 0, absmax = 0, crest_max = 0;
};

struct Fp8In {
    const _Float16* x;
    int ld, rows;
    const uint8_t* x8;
    const float* sx;
    hipStream_t st;
};

// Error, flush and crest of one converted row, added to the table entry
void add_row_stats(Fp8Stat& s, const std::vector<_Float16>& xr, const std::vector<uint8_t>& qr, float scale) {
    const int64_t n = (int64_t)xr.size();
    double e2 = 0, n2 = 0, mx = 0;
    int64_t flushed = 0;
    for (int64_t i = 0; i < n; i++) {
        const double a = (float)xr[i], d = a - scale * e4m3_value(qr[i]);
        e2 += d * d, n2 += a * a, mx = std::max(mx, std::fabs(a));
        flushed += a != 0 && qr[i] % 128 == 0;
    }
    const double err = n2 > 0 ? std::sqrt(e2 / n2) : 0;
    s.rows++, s.err_sum += err, s.err_max = std::max(s.err_max, err), s.flush_sum += (double)flushed / (double)n;
    s.absmax = std::max(s.absmax, mx);
    s.crest_max = std::max(s.crest_max, n2 > 0 ? mx / std::sqrt(n2 / (double)n) : 0.0);
}

void fp8_stats(const PfW& w, const Fp8In& in) {
    const _Float16* x = in.x;
    const int ldx = in.ld, rows_t = in.rows;
    const uint8_t* x8 = in.x8;
    const float* sx = in.sx;
    hipStream_t st = in.st;
    static std::map<std::pair<int64_t, int64_t>, Fp8Stat> table;
    static int64_t total = 0;
    HIPCHECK(hipStreamSynchronize(st));
    const int sample = std::min(rows_t, 24);
    std::vector<_Float16> xr(w.in);
    std::vector<uint8_t> qr(w.in);
    Fp8Stat& s = table[{w.rows, w.in}];
    for (int k = 0; k < sample; k++) {
        const int t = (int)((int64_t)k * rows_t / sample);
        float scale = 0;
        HIPCHECK(hipMemcpy(xr.data(), x + (int64_t)t * ldx, w.in * sizeof(_Float16), hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(qr.data(), x8 + (int64_t)t * w.in, w.in, hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(&scale, sx + t, sizeof scale, hipMemcpyDeviceToHost));
        add_row_stats(s, xr, qr, scale);
    }
    s.calls++;
    if (++total % 400 != 0) return;
    for (const auto& [shape, v] : table)
        std::fprintf(
            stderr,
            "fp8 stats %lldx%lld: %lld calls, %lld rows, rel err mean %.4f max %.4f, flushed %.4f, absmax %.1f, "
            "crest max %.1f\n",
            (long long)shape.first, (long long)shape.second, (long long)v.calls, (long long)v.rows, v.err_sum / v.rows,
            v.err_max, v.flush_sum / v.rows, v.absmax, v.crest_max);
}

}  // namespace

// FP8 path: the input is converted once per call; the weights are unpacked to f16 in slices, then to E4M3 into the
// back part of w16 (f16 2 bytes + fp8 1 byte per value)
void PrefillRun::gemmq_fp8(const PfW& w, const _Float16* x, int ldx, float* y, int ldy, int rows_t) {
    const int64_t rb = row_bytes_of(w.type, w.in);
    const X8Key key{x, ldx, rows_t, (int)w.in};
    const bool have =
        x8_share_ && x8_key_.x == x && x8_key_.ld == ldx && x8_key_.rows == rows_t && x8_key_.in == key.in;
    if (!have) {
        to_fp8_rows({x, ldx, rows_t, w.in, B.x8, w.in, B.sx}, st);
        if (x8_share_) x8_key_ = key;
        static const bool stats = tona::env::flag("TONA_PF_FP8_STATS");
        if (stats) fp8_stats(w, {x, ldx, rows_t, B.x8, B.sx, st});
    }
    const int64_t per8 = std::max<int64_t>(1, (int64_t)(2 * B.w16_elems / w.in) / 128 * 128);
    uint8_t* w8 = reinterpret_cast<uint8_t*>(B.w16);
    for (int64_t r0 = 0; r0 < w.rows; r0 += per8) {
        const int64_t rs = std::min(per8, w.rows - r0);
        bool fill = true;
        uint8_t* kept = wc_get(w.w, r0, 1, (size_t)(rs * w.in + rs * 4), fill);
        uint8_t* w8s = kept ? kept : w8;
        float* sws = kept ? reinterpret_cast<float*>(kept + rs * w.in) : B.sw + r0;
        if (fill) dequant_fp8({w.type, rs, w.in, w.in}, static_cast<const uint8_t*>(w.w) + r0 * rb, w8s, sws, st);
        const Gemm8Seg s{B.x8,        B.sx,          w8s,           sws,           y + r0, rows_t,
                         (int32_t)rs, (int32_t)w.in, (int32_t)w.in, (int32_t)w.in, ldy};
        gemm_fp8(&s, 1, st);
    }
}

// Matmul with a weight in GGUF format: unpack rows in slices to f16, then GEMM
bool pf_fp8_route(const PfW& w, bool fp8_on, int64_t max_in, int64_t max_rows) {
    return fp8_on && w.rows >= 512 && w.in % 64 == 0 && w.in <= 8192 && w.in <= max_in && w.rows <= max_rows &&
           !fp8_skipped(w);
}

void PrefillRun::gemmq(const PfW& w, const _Float16* x, int ldx, float* y, int ldy, int rows_t) {
    if (pf_fp8_route(w, pf_fp8_on(M.m_), M.pf_max_in(), M.pf_max_rows())) {
        gemmq_fp8(w, x, ldx, y, ldy, rows_t);
        return;
    }
    gemmq_f16(w, x, ldx, y, ldy, rows_t);
}

// Output projection of a layer (ssm_out, attn_output): f16 unless TONA_PF_FP8_OUT. Their FP8 rounding noise goes
// straight into the residual of every layer and carries about 70 % of the FP8 loss; all other dense weights stay FP8
// (research/measurements/2026-11-10-fp8-quality.md)
void PrefillRun::gemmq_out(const PfW& w, const _Float16* x, int ldx, float* y, int ldy, int rows_t) {
    if (flags_.fp8_out)
        gemmq(w, x, ldx, y, ldy, rows_t);
    else
        gemmq_f16(w, x, ldx, y, ldy, rows_t);
}

void PrefillRun::gemmq_f16(const PfW& w, const _Float16* x, int ldx, float* y, int ldy, int rows_t) {
    const int64_t rb = row_bytes_of(w.type, w.in);
    const int64_t per = std::max<int64_t>(1, (int64_t)(B.w16_elems / w.in) / 128 * 128);
    for (int64_t r0 = 0; r0 < w.rows; r0 += per) {
        const int64_t rs = std::min(per, w.rows - r0);
        bool fill = true;
        uint8_t* kept = wc_get(w.w, r0, 0, (size_t)(rs * w.in) * sizeof(_Float16), fill);
        _Float16* wp = kept ? reinterpret_cast<_Float16*>(kept) : B.w16;
        if (fill) dequant_f16({w.type, rs, w.in, w.in}, static_cast<const uint8_t*>(w.w) + r0 * rb, wp, st);
        const GemmSeg s{x, wp, y + r0, rows_t, (int32_t)rs, (int32_t)w.in, ldx, (int32_t)w.in, ldy};
        gemm_f16(&s, 1, st);
    }
}

uint8_t* PrefillRun::wc_get(const void* w, int64_t r0, int tag, size_t bytes, bool& fill) {
    fill = true;
    if (!wc_on_) return nullptr;
    const auto key = std::make_tuple(w, r0, tag);
    if (const auto it = wc_map_.find(key); it != wc_map_.end()) {
        fill = false;
        return it->second;
    }
    const size_t at = (wc_used_ + 255) / 256 * 256;
    if (at + bytes > wc_size_)
        throw std::runtime_error(
            std::format("gpu::Mega: the dense weight cache of {} bytes is too small (needs {})", wc_size_, at + bytes));
    wc_used_ = at + bytes;
    wc_peak_ = std::max(wc_peak_, wc_used_);
    return wc_map_[key] = wc_base_ + at;
}

}  // namespace tona::gpu
