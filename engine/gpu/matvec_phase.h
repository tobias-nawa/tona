// A phase of the matvec chain: the waves of the grid work through the tasks of the phase and pick the unit loop of
// each task by format and path. Part of matvec_dev.h; do not include on its own.
#pragma once

#include "gpu/matvec_loops.h"
#include "gpu/matvec_wide.h"

namespace tona::gpu {
namespace {

// Tasks [a, b) of a task array
struct PhaseRange {
    const DevTask* tasks;
    int a, b;
};

// Wave w of W in the grid, with its lane, its LDS staging buffer (float path) and the grid tables (int8 path)
struct WaveCtx {
    int64_t w, W;
    int lane;
    uint8_t* stage;
    const Tabs& tb;
};

// Which Q6_K layouts a kernel instantiation has loops for (template parameter L of k_mega). The tile cases in the
// switches of the int8 and wide paths cost k_mega 8 % of every pass at 1 token even when no task has that type (SGPR
// spills, research/measurements/2026-11-15-pc-q6k-tile-spills.md), so a launch uses kPlain unless its program has
// tile tasks (Mega::start_kernel).
enum class Q6Layout { kPlain, kBoth };

// Which formats beyond the base set a kernel instantiation has loops for (template parameter X of run_phase, from the
// kernel profile): kNvfp4Iq2xxs adds NVFP4 (type 40) and IQ2_XXS (type 16) to the int8 and the wide path. Same reason
// as Q6Layout: a loop that no task of the program uses must not exist in the kernel. A kernel with kNvfp4Iq2xxs calls
// load_iq2xxs_lds() once per block (matvec_tables.h).
enum class ExtraQuants { kNone, kNvfp4Iq2xxs };

// The formats of ExtraQuants (nothing for kNone: the call is empty and the kernel is the one without them)
template <ExtraQuants X>
__device__ __forceinline__ void dispatch_wide_extra(const DevTask& t, int64_t r, const WaveCtx wc, const Batch& bt) {
    if constexpr (X == ExtraQuants::kNvfp4Iq2xxs) {
        if (t.type == 16) wide_loop<16>(t, r, wc.W, wc.lane, wc.tb, bt);
        if (t.type == 40) wide_loop<40>(t, r, wc.W, wc.lane, wc.tb, bt);
    }
}

template <int NB, bool LX, ExtraQuants X>
__device__ __forceinline__ void dispatch_q8_extra(const DevTask& t, int64_t r, const WaveCtx wc, const Batch& bt) {
    if constexpr (X == ExtraQuants::kNvfp4Iq2xxs) {
        if (t.type == 16) units_loop_q8<16, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
        if (t.type == 40) units_loop_q8<40, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
    }
}

template <Q6Layout L, ExtraQuants X>
__device__ __forceinline__ void dispatch_wide(const DevTask& t, int64_t r, const WaveCtx wc, const Batch& bt) {
    switch (t.type) {
        case 8:
            wide_loop<8>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 12:
            wide_loop<12>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 13:
            wide_loop<13>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 14:
            wide_loop<14>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 23:
            wide_loop<23>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case kQ6KTile:
            if constexpr (L == Q6Layout::kBoth) wide_loop<kQ6KTile>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        default:
            dispatch_wide_extra<X>(t, r, wc, bt);
            break;
    }
}

template <int NB, bool LX, Q6Layout L, ExtraQuants X>
__device__ __forceinline__ void dispatch_q8(const DevTask& t, int64_t r, const WaveCtx wc, const Batch& bt) {
    switch (t.type) {
        case 8:
            units_loop_q8<8, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 12:
            units_loop_q8<12, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 13:
            units_loop_q8<13, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 14:
            units_loop_q8<14, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 20:
            units_loop_q8<20, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 23:
            units_loop_q8<23, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 22:
            units_loop_q8<22, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 18:
            units_loop_q8<18, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 21:
            units_loop_q8<21, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case 42:
            units_loop_q8<42, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        case kQ6KTile:
            if constexpr (L == Q6Layout::kBoth) units_loop_q8<kQ6KTile, NB, LX>(t, r, wc.W, wc.lane, wc.tb, bt);
            break;
        default:
            dispatch_q8_extra<NB, LX, X>(t, r, wc, bt);
            break;
    }
}

// BF16 rows whose length is not a multiple of 8 take the generic row function (type 31)
template <int NB>
__device__ __forceinline__ void rows_loop_bf16(const DevTask& t, int64_t r, const WaveCtx wc, const Batch& bt) {
    if (t.in % 8)
        rows_loop<31, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
    else
        rows_loop<30, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
}

template <int NB>
__device__ __forceinline__ void dispatch_float(const DevTask& t, int64_t r, const WaveCtx wc, const Batch& bt) {
    switch (t.type) {
        case 0:
            rows_loop<0, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
            break;
        case 1:
            rows_loop<1, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
            break;
        case 30:
            rows_loop_bf16<NB>(t, r, wc, bt);
            break;
        case 8:
            rows_loop<8, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
            break;
        case 20:
            rows_loop<20, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
            break;
        case 42:
            rows_loop<42, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
            break;
        case 12:
            rows_loop<12, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
            break;
        case 13:
            rows_loop<13, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
            break;
        case 14:
            rows_loop<14, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
            break;
        case 22:
            rows_loop<22, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
            break;
        case 18:
            rows_loop<18, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
            break;
        case 21:
            rows_loop<21, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
            break;
        case 23:
            rows_loop<23, NB>(t, r, wc.W, wc.lane, wc.stage, bt);
            break;
        default:
            break;
    }
}

// Path of one task, starting at unit r of this wave. tw: the task runs on the wide path of this pass.
template <int NB, Q6Layout L, ExtraQuants X>
__device__ __forceinline__ void run_task(const DevTask& t, bool tw, int64_t r, const WaveCtx wc, const Batch& bt) {
    if (tw) {
        dispatch_wide<L, X>(t, r, wc, bt);
        return;
    }
    if constexpr (NB == 1) {
        // One token with x in LDS only for quantize-on-load (gr.up, Q8_0, see OP_MATVEC i[6])
        if (t.xq && bt.lx && t.type == 8) {
            units_loop_q8<8, 1, true>(t, r, wc.W, wc.lane, wc.tb, bt);
            return;
        }
    } else {
        if (t.xq && bt.lx) {
            dispatch_q8<NB, true, L, X>(t, r, wc, bt);
            return;
        }
    }
    if (t.xq)
        dispatch_q8<NB, false, L, X>(t, r, wc, bt);
    else
        dispatch_float<NB>(t, r, wc, bt);
}

// Wave w of W computes the units g = w (mod W) of the flat unit sequence of the phase.
// With bt.n_u >= 0 the number of units of the expert groups depends on the pass; the position of each task in the
// flat sequence (base) is then summed up here instead of read from the task. The body stays in one function on
// purpose: moving the task preparation into a helper changed the machine code of k_mega.
template <int NB = 1, Q6Layout L = Q6Layout::kBoth, ExtraQuants X = ExtraQuants::kNone>
__device__ __forceinline__ void run_phase(const PhaseRange pr, const WaveCtx wc, const Batch& bt = Batch{}) {
    int64_t base = 0;
    for (int i = pr.a; i < pr.b; i++) {
        DevTask t = pr.tasks[i];
        const bool tw = bt.wide && t.wide;
        if (tw) t.units = (t.rows + 15) / 16;  // unit = 16 rows (wide_loop)
        if (bt.n_u >= 0 || bt.wide) {
            if (bt.n_u >= 0 && t.n_exp > 1) t.units = (int64_t)t.upe * min(t.n_exp, bt.n_u);
            t.base = base;
            base += t.units;
        }
        if (t.slot_e >= 0 && t.n_exp <= 1) t.w += (int64_t)bt.slots[t.slot_e] * t.slot_stride;
        t.y += bt.pos * t.y_pos_stride;
        // First unit of this wave: g = w (mod W) in the flat sequence; 32 bits are enough
        const int64_t r = (int64_t)(((uint32_t)wc.w + (uint32_t)wc.W - (uint32_t)(t.base % wc.W)) % (uint32_t)wc.W);
        run_task<NB, L, X>(t, tw, r, wc, bt);
    }
}

}  // namespace
}  // namespace tona::gpu
