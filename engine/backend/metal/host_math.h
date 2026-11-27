// Host-side numbers of a pass that depend only on the position and the token history, as engine/cpu/ref_qwen4exp.cpp
// computes them: the RoPE table, the n-gram rows of the PLE lookup and the cell counts of the QSA indexer; and the
// buffer windows over the weight files. Plain C++,
// tested without a GPU (tests/test_metal_host.cpp).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "backend/rope_yarn.h"
#include "model/ple.h"
#include "model/qwen4exp.h"

namespace tona::metal {

// [cos 0..n_rot/2 | sin 0..n_rot/2] of position pos, with YaRN and its mscale, computed in double as ref.cpp rope()
inline std::vector<float> rope_table(int64_t n_rot, int64_t pos, float base) {
    const int64_t half = n_rot / 2;
    const tona::rope::YarnTable& yarn = tona::rope::yarn_active(base);
    std::vector<float> cs(n_rot);
    for (int64_t i = 0; i < half; i++) {
        const auto [c, s] = tona::rope::pair_cos_sin(yarn, n_rot, pos, base, i);
        cs[i] = (float)c;
        cs[half + i] = (float)s;
    }
    return cs;
}

// Rows of the n-gram table for the token tokens[size - 1], one per head (model/ple.h)
inline std::vector<int32_t> ple_rows(const Hparams& h, const int32_t* tokens, size_t size) {
    std::vector<int32_t> rows(h.ple_heads);
    tona::ple::rows_at(h, tokens, (int64_t)size - 1, rows.data());
    return rows;
}

inline std::vector<int32_t> ple_rows(const Hparams& h, const std::vector<int32_t>& tokens) {
    return ple_rows(h, tokens.data(), tokens.size());
}

struct Window {
    const uint8_t* base;
    uint64_t bytes;
};

// Page-aligned windows over one file's tensors (sorted by address), each at most `limit` bytes and holding whole
// tensors: one buffer may not exceed maxBufferLength (20.1 GB on the M1 Max 32 GB, less than a 20.4 GB model;
// research/measurements/2026-11-11-mac-synth.md). Windows may overlap by part of a page. A tensor more than `max_gap`
// bytes after the window's end opens a new one: with the arena the routed experts between the tensors stay out of the
// buffers, which a command buffer would otherwise hold in its working set.
inline std::vector<Window> windows(const std::vector<const TensorInfo*>& ts, uint64_t limit, uint64_t page,
                                   uint64_t max_gap = UINT64_MAX) {
    std::vector<Window> out;
    for (const TensorInfo* t : ts) {
        const uint8_t* end = t->data + t->bytes;
        const bool near = !out.empty() && (uintptr_t)t->data - (uintptr_t)out.back().base <=
                                              out.back().bytes + std::min(max_gap, limit);
        if (near && (uint64_t)(end - out.back().base) <= limit) {
            out.back().bytes = std::max<uint64_t>(out.back().bytes, (end - out.back().base + page - 1) / page * page);
            continue;
        }
        const uint8_t* base = (const uint8_t*)((uintptr_t)t->data / page * page);
        if ((uint64_t)(end - base) > limit)
            throw std::runtime_error("Metal: tensor " + t->name + " exceeds one buffer");
        out.push_back({base, (end - base + page - 1) / page * page});
    }
    return out;
}

// The memory at the arena's start (TONA_METAL_ARENA_GB=auto), in bytes
struct MemNow {
    uint64_t ram = 0;          // physical memory
    uint64_t taken = 0;        // wired, anonymous and compressed pages of all processes, ours included
    uint64_t mapped = 0;       // our weights outside the arena: file pages that must stay resident
    uint64_t keep = 0;         // left for the rest of the process, stashed prompts, the page cache and the user
    uint64_t working_set = 0;  // the GPU's recommended working set
    uint64_t gpu = 0;          // our buffers without the arena, mappings included
    uint64_t slack = 0;        // headroom in the working set
    uint64_t max_buffer = 0;   // the arena is one buffer
};

// The arena takes what neither other programs nor the rest of the engine hold: page cache counts as free (it is
// reclaimed first), anonymous memory of other programs does not (taking it means compressing it)
inline int64_t arena_auto_budget(const MemNow& m) {
    const auto left = [](uint64_t a, uint64_t b) { return a > b ? (int64_t)(a - b) : (int64_t)0; };
    return std::min(
        {left(m.ram, m.taken + m.mapped + m.keep), left(m.working_set, m.gpu + m.slack), (int64_t)m.max_buffer});
}

// With the governor the start budget only sets where planning ends: the fill (backend::fill_step) and the governor's
// growth go on while free RAM stays above the reserve, so the arena is planned up to what the GPU's working set
// leaves. Its blocks (at most 32 MB each) lift the one-buffer limit
inline int64_t arena_ceiling(const MemNow& m) {
    return m.working_set > m.gpu + m.slack ? (int64_t)(m.working_set - m.gpu - m.slack) : 0;
}

}  // namespace tona::metal
