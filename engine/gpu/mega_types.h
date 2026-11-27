// Structs and limits the host shares with the megakernel (gpu/mega_kernel.h): one step of the program, the per-pass
// parameters and the mailbox in host memory.
#pragma once

#include <cstdint>

#include "backend/backend.h"
#include "gpu/matvec.h"
#include "gpu/mega_mailbox_fwd.h"

namespace tona::gpu {

constexpr int MEGA_BMAX = backend::BMAX;  // tokens per pass (k_mega<4> up to 4, k_mega<8> above)
constexpr int MEGA_UMAX = 128;            // experts per layer and pass (union over the tokens), >= n_expert_used * BMAX
constexpr int MEGA_MAX_CELLS = 4096;      // KV cells per token in the QSA attention (top-k blocks and tail)
constexpr int MEGA_SMAX = backend::SMAX;  // sequences with their own state (slots), TONA_SEQS
constexpr int MEGA_RAW_RING = 16;         // raw indexer keys in the ring (the pool needs the last blk, plus one pass)

using MegaStats = backend::Stats;

// One step of the program (kinds in mega_ops.h). a, b: task range of a matvec phase; otherwise numbers and device
// pointers per kind
struct MegaOp {
    int32_t kind = 0, cond = 0, a = 0, b = 0;
    int32_t nobar = 0;  // 1: no barrier after this step (the next one reads nothing of it); 2: dependency instead of
                        // a barrier: the next step waits only for the blocks in its p[13] (masks [BMAX][block] as
                        // uint64: per width and own block, the blocks whose data it reads)
    int32_t seq = -1;   // >= 0: step with state of slot seq in the program of several sequences (rows sb0 ...)
    int32_t i[8] = {};
    float f[4] = {};
    void* p[14] = {};
};

// Matvec task with the parts that change per pass (see DevTask in matvec_dev.h)
struct MegaTask {
    MatvecTask t;
    int32_t slot_e = -1;  // expert e of the selection: w += slot * slot_stride
    int64_t slot_stride = 0, y_pos_stride = 0;
    float act_scale = 0;  // != 0: y = silu(y * act_scale)
    int32_t n_exp = 1;    // > 1: group of experts slot_e ... slot_e + n_exp - 1, x/y offset per expert by x_e/y_e
    int64_t x_e = 0, y_e = 0;
    int64_t xb = 0, yb = 0;  // distance of the token vectors of a pass in x and y
    int32_t wide = 0;        // dense, x valid as float: wide passes run it through wide_loop (DevTask::wide)
};

// Written by the CPU per pass; n_cells and the expert selection are written by the kernel itself
struct MegaParams {
    int64_t pos, seq;            // position of the first token; sequence number for the mailbox
    int32_t nb;                  // tokens in this pass
    int32_t commit;              // so many tokens of the previous pass are committed by DeltaNet
    int32_t parity;              // half of the DeltaNet intermediates this pass writes to
    int32_t n_u, n_miss, n_cpu;  // n_cpu > 0: the CPU computes experts of this layer, result in cpu_y
    int32_t fill_half, pad_;     // CPU mode: moe.copy copies from this half of the refill buffer
    int32_t n_cells[MEGA_BMAX];
    int32_t slot[MEGA_UMAX];
    uint32_t umask[MEGA_UMAX];        // bit b: token b uses expert u
    float ew[MEGA_BMAX * MEGA_UMAX];  // weight of expert u for token b: ew[b * UMAX + u], 0 = not chosen
    int32_t miss_slot[MEGA_UMAX];
    unsigned long long* arr;  // TONA_BAR_PROF: arrival of each block at each barrier ([step * blocks + block])
    uint32_t* aborted;        // mailbox (target model only): seq + 63 after an abort, so the CPU does not wait 10 s
    int32_t nobar_all;        // TONA_MEGA_NOBAR=1: measuring mode without barriers (output is meaningless)
    int32_t wide_min;         // from this many tokens the dense int8 tasks run on WMMA (TONA_WIDE_MIN, 0: never)
    const float* mtp_h;       // MTP: h per token [nb][hc * d] (target or last MTP step), null: 0
    // several sequences in one pass: nseq > 0; per slot first row, tokens, committed count, half, position
    int32_t nseq;
    int32_t sb0[MEGA_SMAX], snb[MEGA_SMAX], scommit[MEGA_SMAX], sparity[MEGA_SMAX];
    int64_t spos[MEGA_SMAX];
    int32_t budget_ms;  // deadline of the whole pass in ms (0: none), also for the waits on the CPU
    // Yield instead of abort: if block 0 waits longer than yield_ms for the CPU, all blocks end and the CPU restarts
    // the kernel at that step once it is done (0: abort after HOST_WAIT_MS)
    int32_t yield_ms;
};

// Mailbox in host memory (coherent, mapped): at the router the GPU reports gpu_seq and waits for cpu_seq; in
// between the CPU reads the router logits and writes the selection. MaxE: experts per layer the router rows hold.
template <int MaxE>
struct MegaMailboxT {
    static constexpr int kMaxExperts = MaxE;
    uint32_t gpu_seq, pad0[15];
    uint32_t cpu_seq, pad1[15];
    int32_t n_u, n_miss, n_cpu, fill_half, pad2[12];
    uint32_t cpu_done, pad3[15];  // the CPU has computed its experts of the layer (sequence number as cpu_seq)
    uint32_t aborted, pad4[15];   // pass aborted (MegaParams::aborted)
    uint32_t yield_seq, yield_op, pad5[14];  // yield (MegaParams::yield_ms): seq + 62, step to continue at
    int32_t slot[MEGA_UMAX];
    uint32_t umask[MEGA_UMAX];
    float ew[MEGA_BMAX * MEGA_UMAX];
    int32_t miss_slot[MEGA_UMAX];
    float router[MEGA_BMAX * MaxE];
    float pred[MEGA_BMAX * MaxE];              // router of the next layer on the same x (prefetching)
    uint32_t ev_req, ev_ack, ev_ms, pad6[13];  // eviction credit: request and answer counter, evicted ms (wait_clock.h)
    // NaN values among the E router logits (router) and the predicted ones (pred) of the last step; the GPU clamped
    // them before the copy (op_host_wait), the CPU reports them
    uint32_t nan_router, nan_pred, pad7[14];
};

// The router rows (and the predicted ones) of a pass: in the mailbox for up to MEGA_MAX_EXPERTS experts per layer, for
// the wide size class in an extension behind the struct, so that every other field keeps its offset in both classes
// and the Coder's kernel code does not change.
#ifdef __HIPCC__
#define TONA_MB_HD __host__ __device__
#else
#define TONA_MB_HD
#endif
template <int MaxE>
TONA_MB_HD inline float* mailbox_router(MegaMailbox* mb) {
    if constexpr (MaxE <= MEGA_MAX_EXPERTS)
        return mb->router;
    else
        return reinterpret_cast<float*>(mb + 1);
}
template <int MaxE>
TONA_MB_HD inline float* mailbox_pred(MegaMailbox* mb) {
    if constexpr (MaxE <= MEGA_MAX_EXPERTS)
        return mb->pred;
    else
        return reinterpret_cast<float*>(mb + 1) + MEGA_BMAX * MaxE;
}
// Bytes of the mailbox with the extension of the size class
constexpr size_t mailbox_bytes(int max_e) {
    return sizeof(MegaMailbox) + (max_e > MEGA_MAX_EXPERTS ? 2 * sizeof(float) * (size_t)MEGA_BMAX * (size_t)max_e : 0);
}

// Router rows as the host reads them: row b of the logits and of the prediction, stride in floats
struct MailboxRows {
    float *router = nullptr, *pred = nullptr;
    int stride = MEGA_MAX_EXPERTS;
    const float* router_row(int b) const { return router + (size_t)b * stride; }
    const float* pred_row(int b) const { return pred + (size_t)b * stride; }
};

}  // namespace tona::gpu
