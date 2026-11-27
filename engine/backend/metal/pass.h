// One forward pass of nb consecutive tokens (1 <= nb <= BMAX) of one sequence, encoded into a command buffer: layer by
// layer, mat-vecs over all nb tokens at once, the recurrent parts token by token inside their kernels. Every token is
// computed with the same arithmetic whatever nb is, so all pass widths give bit-identical logits. The order of
// operations follows engine/cpu/ref_qwen4exp.cpp Qwen4ExpOracle::step.
#pragma once
#include <algorithm>
#include <functional>
#include <vector>

#include "backend/env.h"
#include "backend/metal/device.h"
#include "backend/metal/seq.h"
#include "backend/metal/weights.h"
#include "backend/ple_stats.h"
#include "model/qwen4exp.h"

namespace tona::metal {

constexpr int BMAX = 8;     // tokens per pass (backend::BMAX)
constexpr int MV_BMAX = 4;  // tokens per f32 mat-vec dispatch with TONA_METAL_MV5=0 (kernels for B = 1..5)

// TONA_METAL_MMA: half MMA mat-vec for chunks of 4 tokens (kernels/matvec_mma.metal). On by default: lossy, but within
// the quality check (research/measurements/2026-11-12-mac-mma-quality.md) and decode +7.4 % code, +4.9 % text in the
// server (2026-11-12-mac-mma-server-ab.md); 0 brings the f32 path back, and so does TONA_DETERMINISTIC (every width
// bit-identical, as the reference drops its wide kernel). Read by the pass and by the pass costs, which must agree.
inline bool mma_from_env() { return !env::flag("TONA_DETERMINISTIC") && env::integer("TONA_METAL_MMA", 1) != 0; }
// TONA_METAL_MMA_WIDE: up to 8 tokens per MMA dispatch (W5 to W8 in one) instead of chunks of 4 plus an f32 rest. On
// by default: W8 rows bit-equal to the 4-token chunks, decode +2.6 % code, +1.2 % text in the server
// (research/measurements/2026-11-13-mac-mma-wide-server-ab.md); 0 brings the 4-token chunks back.
inline int mma_tokens_from_env() { return env::integer("TONA_METAL_MMA_WIDE", 1) != 0 ? 8 : 4; }
// TONA_METAL_MV5: up to 5 tokens per dense f32 mat-vec dispatch (mv, mvs; 6 to 8 split as 3+3, 4+3, 4+4) instead of
// 4: W5 4.4 ms, W6 1.3 ms faster. With 8 tokens per threadgroup the kernels spilled (W8 +31 ms per pass); llama.cpp's
// mul_mv_ext holds 5 at most. Bit-identical per token (rows_dot keeps the tokens apart). On by default (maintainer
// decision 2026-11-13: server decode code +3.6 %, text -0.2 % within tolerance,
// research/measurements/2026-11-13-mac-mv5-server-ab.md); 0 brings the 4-token chunks back. Read by the pass and by
// the pass costs.
inline bool mv5_from_env() { return env::integer("TONA_METAL_MV5", 1) != 0; }
// TONA_METAL_MVG_SPLIT (probe, off): the slots of one routed expert over several threadgroup rows of at most this
// many slots each instead of one row that takes them 4 at a time; 1 and 2 use an instance of at most 2 slots per
// weight pass (fewer registers). Bit-identical per slot (rows_dot_at keeps the slots apart).
inline uint32_t mvg_split_from_env() {
    return (uint32_t)std::clamp<int64_t>(env::integer("TONA_METAL_MVG_SPLIT", 0), 0, 4);
}
// Prefill attention tokens per dispatch, each slice its own command buffer: in a 2048-token chunk at positions 2048 to
// 4095 the longest buffer took 0.53 s with the whole QSA layer in one, 0.22 s with slices of 512
// (research/measurements/2026-11-11-mac-prefill-gemm64.md)
constexpr uint32_t ATTN_SLICE = 512;

// Buffers of a pass, [nb][...] for every per-token quantity
struct Scratch {
    Buf R, xn, lo, gate, mixed, inject, y;             // residual streams and gated residual
    Buf qkv, z, beta, alpha, conv, o;                  // DeltaNet (o also QSA output)
    Buf qfull, qi, score, cells;                       // QSA; cells [nb][cell_cap]
    Buf rlogits, ids, ew, g, u, eout, sg, su, so, gd;  // MoE
    Buf e, key, val;                                   // PLE
    Buf cs_pos, cs_blk, tok, ple_rows, nan;            // host inputs of the pass; NaN counter of the router
    Buf goff, perm;                                    // prefill: slots grouped by expert (moe_group)
    Buf tscore, bits;                                  // prefill: indexer scores and block bitmaps per token
    Buf ple_stage;                                     // the n-gram rows of the pass, copied from the mapping
    Buf glist;                                         // the experts the pass uses (moe_group), for mvg
};

// Routed experts of a mat-vec: one slot per chosen expert (expert ids[s]), slot s reads input s / xdiv
struct Slots {
    uint32_t n = 1;
    Buf ids;
    uint32_t xstride = 0, xdiv = 1;
    bool grouped = false;  // moe_group ran: experts grouped (mvg); else one slot per expert (one token)
    Buf tab;               // the arena's offset table of the matrix (kernels/moe_table.metal); nil: expert * bytes
};

// Inputs of the DeltaNet recurrence of nb tokens: qkv [nb][ch], alpha and beta [nb][v heads]
struct DnIn {
    Buf qkv, alpha, beta;
    int nb = 0;
};

struct PassIn {
    const int32_t* tokens = nullptr;
    int nb = 1;
    int64_t pos = 0;                                // position of tokens[0]
    const std::vector<int32_t>* history = nullptr;  // all tokens of the sequence, ending with these nb
    Buf logits;                                     // [nb][vocab]; no buffer: the LM head is skipped
    bool prefill = false;                           // GEMMs instead of mat-vecs; nb up to the chunk
    bool mix_head = false;                          // gated read of the head without logits (for head())
};

// One MTP pass: entries [pos, pos + nb) of the MTP head (engine/gpu/mega_build_mtp.cpp). The caller writes the inputs
// first: e [nb][hc][d] the token's embedding once per stream, hin [nb][hc][d] the h of each entry (0: unknown).
// kv_only: only the KV and indexer keys of the entries; otherwise R after the block is the h of the next step and
// logits [rows] holds the truncated LM head of the last entry.
struct MtpPassIn {
    int nb = 1;
    int64_t pos = 0;
    bool kv_only = false;
    bool prefill = false;  // GEMMs, KV only, nb up to mtp_part()
};

// MTP entries of a prompt per GEMM part (the reference's C/2, prefill_mtp.cpp): 4 buffers of 16 MB on the Coder
constexpr int MTP_PART = 512;

struct MtpBufs {
    Buf e, hin, x, R;  // inputs, [enorm(e) | hnorm(h)] per (entry, stream), the head's own residual streams
    Buf logits;        // [rows] of the truncated head
    Mat head;          // rows of the target's LM head (MtpVocab)
    Buf rows;          // token of each head row on the device (-1: free); with pick_out: the pick runs on the GPU
    Buf pick_part, pick_out;  // TONA_METAL_GPU_PICK: per threadgroup and merged {row, max, sum, NaN count}
    Buf pick_done;            // finished threadgroups of the pick (uint32, 0 between passes)
    int slot = 0;             // index of the head's QSA buffers in a SeqState
};

class Pass {
public:
    // cap: most tokens of one pass (BMAX for decode, the prefill chunk); mtp: the MTP head or nullptr
    Pass(Device& dev, Weights& w, const Qwen4ExpModel& m, const MtpHead* mtp, int64_t max_ctx, int cap);
    // A whole pass into one encoder
    void encode(id<MTLComputeCommandEncoder> enc, SeqState& st, const PassIn& in);
    // The same in parts, so that a long pass can spread over several command buffers: begin (host inputs, embedding),
    // one call per layer, finish (PLE ring, gated read of the head, LM head if in.logits)
    void begin(id<MTLComputeCommandEncoder> enc, const PassIn& in);
    void layer(id<MTLComputeCommandEncoder> enc, SeqState& st, const PassIn& in, uint32_t l);
    void finish(id<MTLComputeCommandEncoder> enc, SeqState& st, const PassIn& in);
    // LM head of tokens [t0, t0 + n) of the last pass after finish, into logits [n][vocab]
    void head(id<MTLComputeCommandEncoder> enc, int t0, int n, const Buf& logits);
    uint32_t router_nans() const { return (uint32_t)s_.nan.i32()[0]; }
    // At the part "cold" (split): the arena layer whose router just ran; its routing is route_ids() of the pass
    uint32_t cold_layer() const { return cold_layer_; }
    // TONA_METAL_ARENA_AHEAD: the next layer's router applied to this layer's input, [nb][used] ids, written before
    // the part "ahead"; false when this stop predicts nothing (last layer, prefill)
    bool ahead() const { return ahead_now_; }
    const int32_t* ahead_ids() const { return ahead_ids_.i32(); }
    uint32_t ahead_k() const { return ahead_k_; }
    int cap() const { return cap_; }
    // Profiling (profile.h): called at every part boundary, returns the encoder of the next part
    std::function<id<MTLComputeCommandEncoder>(const char*)> split;
    uint32_t layers() const { return h_.n_layer; }
    // MTP (pass_mtp.mm): its buffers, written by the caller before mtp()
    MtpBufs& mtp_bufs() { return mb_; }
    const float* mtp_logits() const { return mb_.logits.f32(); }
    const float* mtp_pick_out() const { return mb_.pick_out.f32(); }
    // TONA_GPU_GAPS: seconds of the last write_inputs in the n-gram rows and in the positions
    double rows_s() const { return rows_s_; }
    double positions_s() const { return positions_s_; }
    // Routing of the last pass: [nb][used] expert ids and weights
    const int32_t* route_ids() const { return s_.ids.i32(); }
    const float* route_weights() const { return s_.ew.f32(); }
    Buf residual() const { return s_.R; }  // [nb][hc][d] of the last pass of the target: its h
    // Read-ahead of the n-gram rows of seq[from, to) (seq from its start or with the n-gram's predecessors) for the
    // next target pass, while the MTP head drafts it (backend/ple_prefetch.h)
    void prefetch_rows(const int32_t* seq, int64_t from, int64_t to);
    void mtp(id<MTLComputeCommandEncoder> enc, SeqState& st, const MtpPassIn& in);
    // After commit(m < nb) restored the DeltaNet state (restore_after): the recurrence of the first m tokens again
    void replay_dn(id<MTLComputeCommandEncoder> enc, SeqState& st, int m);
    int mtp_part() const { return std::min(MTP_PART, cap_); }

private:
    void mv(Dispatch& d, const Mat& w, const Buf& x, const Buf& y, int nb);
    bool use_mma(const Mat& w, int nb) const;
    int mv_chunk(int n) const;
    void mv_experts(Dispatch& d, const Mat& w, const Buf& x, const Buf& y, const Slots& sl);
    void norm_rows(Dispatch& d, const Buf& w, uint32_t rows, uint32_t w_rows, const Buf& xn);
    void hc_mix(Dispatch& d, const GatedResidual& gr, bool inject, int nb);
    struct HcOut {
        Buf xn, lo, gate, mixed;
    };
    void hc_mix_to(Dispatch& d, const GatedResidual& gr, int nb, const HcOut& o);
    void hc_combine(Dispatch& d, int nb);
    void ple(Dispatch& d, const Layer& L, Recurrent& r, int nb);
    // i: index of the layer's buffers in the SeqState (DeltaNet or QSA)
    void deltanet(Dispatch& d, const Layer& L, int i, SeqState& st, int nb);
    void dn_recur(Dispatch& d, const Layer& L, int i, Recurrent& r, const DnIn& in);
    void qsa_kv(Dispatch& d, const Layer& L, int i, SeqState& st, int64_t pos, int nb);
    void qsa(Dispatch& d, const Layer& L, int i, SeqState& st, const PassIn& in);
    void qsa_attend(Dispatch& d, const Layer& L, int i, SeqState& st, const PassIn& in);
    void qsa_select(Dispatch& d, int i, SeqState& st, const PassIn& in);
    void attend_tiles(Dispatch& d, int i, SeqState& st, const PassIn& in);
    void moe(Dispatch& d, const Layer& L, int nb);
    void group(Dispatch& d, uint32_t n_expert, uint32_t slots);
    void shared_expert(Dispatch& d, const Layer& L, int nb);
    void write_inputs(const PassIn& in);
    void write_positions(int64_t pos, int nb);  // RoPE tables, completed blocks, tails of the indexer cells
    void mtp_init(const MtpHead& mtp);
    void part(Dispatch& d, const char* name) {
        if (split) d.enc = split(name);
    }

    Device& dev_;
    Weights& w_;
    const Qwen4ExpModel& m_;
    const Hparams& h_;
    const MtpHead* mtp_;
    MtpBufs mb_;
    int64_t max_ctx_;
    int cap_;
    bool gemm_ = false;  // the pass being encoded is a prefill
    bool mma_ = mma_from_env();
    int mma_tokens_ = mma_tokens_from_env();     // most tokens of one MMA dispatch
    bool mv5_ = mv5_from_env();                  // TONA_METAL_MV5
    uint32_t mvg_split_ = mvg_split_from_env();  // TONA_METAL_MVG_SPLIT
    uint32_t cell_cap_;                          // indexer cells one token can have
    Buf attn_part_;                              // decode attention: [head][part][hd | max | sum] (attend_part)
    uint32_t pool_cap_;                          // indexer blocks one token can have, rounded up to whole tiles of 32
    Mat ple_;                                    // the n-gram table, read on the host
    backend::PleStats ple_stats_;                // TONA_PLE_STATS
    double rows_s_ = 0, positions_s_ = 0;        // TONA_GPU_GAPS
    Scratch s_;
    std::vector<int> slot_of_layer_;  // index into the DeltaNet or the QSA buffers of a SeqState
    std::vector<int64_t> blocks_;     // first positions of the indexer blocks this pass completes
    uint32_t cold_layer_ = 0;
    bool ahead_now_ = false;
    Buf ahead_logits_, ahead_ids_, ahead_ew_, ahead_nan_;  // TONA_METAL_ARENA_AHEAD, BMAX tokens
    uint32_t ahead_k_ = 0;  // experts predicted per token (TONA_METAL_ARENA_AHEAD_K, default the router's top-k)
    HcOut ahead_hc_;        // TONA_METAL_ARENA_AHEAD=2: the next layer's gated read of the residual, BMAX tokens
    void predict(Dispatch& d, uint32_t l, int nb);
};

}  // namespace tona::metal
