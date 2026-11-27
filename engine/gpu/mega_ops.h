// Step kinds of the megakernel program and the limits the kernel (gpu/mega_kernel.h) and the host code agree on.
#pragma once

#include <cstdint>

#include "gpu/mega_types.h"
#include "gpu/wait_clock.h"

namespace tona::gpu {
namespace {

// All blocks run one step together, followed by a grid barrier. Every step computes the nb tokens of the pass
// (token b at position pos + b); per-token buffers lie at a fixed distance (given in the comments of the steps).
enum Kind : int32_t {
    OP_MATVEC,        // matvec phase: tasks [a, b)
    OP_BCAST_NORM,    // R = embedding in all streams, xn = norm(R) * w
    OP_COMBINE_NORM,  // R' = R + y * 2 sigma(inject/hc) (without y: R' = R), xn = norm(R') * w (without w: no xn)
    OP_HC_MIX_QUANT,  // mixed = mean over the streams of xn * sigma(gate), plus int8
    OP_DN_CONV,       // DeltaNet: causal conv + SiLU, L2 norm of the q/k heads
    OP_DN_RECUR,      // DeltaNet: delta rule per V head (commit of the previous pass, then nb tokens), raw
    OP_QSA_PREP,      // QSA: split q/gate, norm + RoPE of q, k (into the cache), indexer pool and q
    OP_IDX_SCORE,     // QSA: score per pool block
    OP_TOPK,          // QSA: pick the best pool blocks (block b for token b)
    OP_ATTN,          // QSA: attention per head, * sigma(gate), int8
    OP_PLE_GATE,      // PLE: gate per stream, gated, normed
    OP_PLE_CONV,      // PLE: dilated conv, R' = R + gated + silu(conv)
    OP_ACT_QUANT,     // MoE: silu(g) * u, int8
    OP_MOE_SUM,       // MoE: y = sum of w_e * out_e + sigma(g_shared) * out_shared
    OP_DN_NORM,       // DeltaNet: norm per V head * w * sigma(z), int8
    OP_HOST_WAIT,     // block 0: router logits into the mailbox, wait for the CPU, take over the selection
    OP_EXPERT_COPY,   // copy misses from the host buffer into their slots
    OP_QUANT,         // round x to int8 (one wave per block of 32)
    OP_HC_MEAN,       // mean over the streams of R, for the drafter
    OP_NORM,          // drafter: RMSNorm per token * w, float and int8
    OP_DCONV,         // drafter: dynamic convolution over the block (kernel 2), optionally added to the residual
    OP_DQK,           // drafter: norm + RoPE of q and k, K/V into the cache
    OP_DATTN,         // drafter: attention over context and block (not causal)
    OP_SWIGLU,        // drafter: silu(g) * u, int8
    OP_HOST_Y,        // block 0: wait for the experts of the CPU (only if it computes some)
    OP_ATTN_SPLIT,    // QSA: attention, one wave per (token, head, 256 cells), partials with max and sum
    OP_ATTN_MERGE,    // QSA: merge the partials, * sigma(gate), int8
    OP_DATTN_SPLIT,   // drafter: attention like OP_DATTN, one wave per (token, head, 256 cells)
    OP_DATTN_MERGE,   // drafter: merge the partials, int8
    OP_ATTN_WMMA,     // QSA: one block per (token, KV head, chunk), all heads of the group on WMMA, partials
    OP_DATTN_WMMA,    // drafter: one block per (KV head, chunk), all tokens x heads of the group on WMMA, partials
    OP_MTP_IN,        // MTP: x per (token, stream) = [enorm(e) | hnorm(h_c)], float and int8 (h from P->mtp_h)
    OP_ROUTE,         // MTP: routing on the GPU (softmax, top-k, renormalized, union) into P (block 0)
    OP_MTP_PICK,      // MTP: argmax and sum of exp of the last token's logits (block 0), row/max/sum into p[2]
    OP_MTP_PROJ,      // MTP dynamic rows: p = B^T x of the scorer
    OP_MTP_DYN,       // MTP dynamic rows: scores of all candidates, the best rows per partition, copied from the head
    OP_FATTN_PREP,    // full attention: split q/gate, optional norm and RoPE of q and k (k, v into the cache)
    OP_FATTN_SPLIT,   // full attention: one block per (token, KV head, chunk of positions), partials with max and sum
    OP_FATTN_MERGE,   // full attention: merge the partials, optional * sigma(gate), int8
    OP_RES_NORM,      // one residual stream: R' = R + y, xn = norm(R') * w, int8 (without y: R' = R)
    OP_ROUTE_SIG,     // routing on the GPU with the sigmoid policy (score + bias, scaled weights) into P (block 0)
    OP_MAMBA_CONV,    // Mamba-2: causal conv + bias + SiLU over x, B, C of the in-projection
    OP_MAMBA_SCAN,    // Mamba-2: selective scan per (head, rows), commit of the previous pass, then nb tokens
    OP_MAMBA_NORM,    // Mamba-2: y * silu(z), RMS norm per group * w, int8
    OP_ACT_RELU2,     // latent MoE: relu(u)^2 of the ungated experts and the shared expert, int8
    OP_MOE_SUM_LAT,   // latent MoE: sum of w_e * out_e in the latent space, float and int8
    OP_MOE_ADD,       // y = a + b (projected routed experts plus the shared expert)
    OP_QUANT_LAT,     // latent MoE: the latent input x to int8 (one wave per block of 32)
};
const char* const kind_names[] = {
    "matvec",    "embed",      "combine",    "mix",        "dn.conv",   "dn.recur",   "qsa.prep", "qsa.score",
    "qsa.topk",  "qsa.attn",   "ple.gate",   "ple.conv",   "moe.act",   "moe.sum",    "dn.norm",  "host.wait",
    "moe.copy",  "quant",      "feat",       "d.norm",     "d.conv",    "d.qk",       "d.attn",   "d.swiglu",
    "host.y",    "qsa.attn",   "qsa.merge",  "d.attn",     "d.merge",   "qsa.attn",   "d.attn",   "mtp.in",
    "route",     "mtp.pick",   "mtp.proj",   "mtp.dyn",    "fa.prep",   "fa.attn",    "fa.merge", "res.norm",
    "route.sig", "mamba.conv", "mamba.scan", "mamba.norm", "moe.relu2", "moe.sumlat", "moe.add",  "quant.lat"};

// Waves per block: one block per CU with 16 waves holds as many waves at once as 2 blocks of 8 without a block
// having to wait for the room of a second one (2 blocks of 8 per CU deadlocked with the desktop,
// research/measurements/2026-10-23-mega-v2)
#ifndef TONA_MEGA_WAVES
#define TONA_MEGA_WAVES 16
#endif
constexpr int MWAVES = TONA_MEGA_WAVES;
constexpr int MAX_CELLS = MEGA_MAX_CELLS;
// LDS floats of the kernel. It must not grow: with 61.6 KB not all 32 blocks fitted next to the desktop and the
// barriers aborted; x of a matvec phase in LDS is limited to the same buffer (4 of 4 runs hung with 55 KB)
constexpr int SM_FLOATS = 64 + 256 + MAX_CELLS;
constexpr int XSTAGE_BYTES = SM_FLOATS * 4;
constexpr int BMAX = MEGA_BMAX, UMAX = MEGA_UMAX;
// Width the splits are sized for (attention chunks, partial sums of the HC norm): fixed, not nb and not BMAX, so a
// token's rounding does not depend on the width of the pass and passes up to 4 tokens stay as they were
constexpr int PNB = 4;
// Longest wait of the kernel for the CPU. The kernel occupies the GPU meanwhile and the desktop on the same card
// cannot draw. amdgpu aborts its jobs after 2 s (lockup_timeout) and the session crashes. With a barrier behind the
// wait stay well below.
constexpr int HOST_WAIT_MS = 1000;
// Barrier buffer: [0..3] barrier, [4..7] start times, [8..13] counters (acc), from byte 64 progress per block, from
// byte 64 + 1024 the last finished step + 1 per block (dependencies instead of barriers, MegaOp::nobar = 2), behind
// it 7 words that say what the block saw that counted an abort (note_abort)
constexpr int BAR_YIELD = 14;  // word of the pause: step + 1 (0: none), see yield_pass
constexpr size_t BAR_DONE = 64 + 4 * 256;
constexpr size_t BAR_DIAG = 64 + 8 * 256;
constexpr size_t BAR_EXT = BAR_DIAG + 64;  // eviction credit (wait_clock.h, EvictLink), 4 words
constexpr size_t BAR_BYTES = BAR_EXT + 64;
constexpr size_t BAR_HIST_BYTES =
    64 + 64;  // never reset by a launch: TONA_MEGA_GAP_HIST (8 words), pointer of the credit mailbox
static_assert(BAR_EXT == kEvictExtOffset && BAR_BYTES + 64 == kEvictLinkOffset);
constexpr int DS = 128;  // DeltaNet head size
// Raw indexer keys: ring over RAW_RING positions (the pool needs the last blk, plus one pass)
constexpr int RAW_RING = MEGA_RAW_RING;
// Split-K of the QSA attention: one wave per (token, head, chunk of ATT_CH cells)
constexpr int ATT_CH = 256, ATT_SPL = MAX_CELLS / ATT_CH;

}  // namespace
}  // namespace tona::gpu
