// The host side of the prefill: one PrefillRun executes the layers for the chunks of one Mega::prefill call. The class
// is a friend of Mega (it reads weights, caches and buffers) and is spread over prefill_*.cpp by topic.
#pragma once

#include <hip/hip_runtime.h>

#include <any>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "gpu/gemm.h"
#include "gpu/mega.h"
#include "gpu/prefill_kernels.h"

namespace tona::gpu {

constexpr int EB = 8;          // size of w16 in experts (slices of dense matrices, copy buffers of the experts)
constexpr int MB = 32;         // experts per batch in the MoE at most (staging buffer in RAM: 40)
constexpr int SCORE_TOK = 32;  // tokens per pass when scoring the indexer blocks

// Areas of the workspace; with base = nullptr it only measures
struct Carve {
    uint8_t* base;
    size_t off = 0;
    const uint8_t* end = nullptr;  // with base: the end of the memory behind it; carving past it throws
    template <class T>
    T* take(size_t n) {
        off = (off + 255) / 256 * 256;
        T* p = reinterpret_cast<T*>(base + off);
        off += n * sizeof(T);
        if (end && base + off > end)
            throw std::runtime_error("gpu::Mega: a prefill buffer reaches past the end of the workspace");
        return p;
    }
};

// Fixed buffers of the prefill (for all layers), then the area of the phase that is running
struct PfBufs {
    float *e, *R, *sc, *mixed, *y, *feat;
    _Float16 *mixed16, *w16;
    size_t w16_elems = 0;
    uint8_t* x8 = nullptr;  // FP8 path (TONA_PREFILL_FP8): input per token, factors per token and per weight row
    float *sx = nullptr, *sw = nullptr;
    uint8_t* phase = nullptr;
};

// Model dimensions the prefill uses, as ints like the kernel arguments
struct PfDims {
    int d, ff, sff, ne;
    int max_e;    // size class of the experts per layer (expert_cap)
    int64_t hcd;  // floats of the residual per token (the Coder: n_hc streams of d)
    int kd, vd, ch, nk, nv;
    int nh, nkv, hd, idim, ih, blk, topk;
    float eps, base;
    int nfeat;
    int dn_conv;  // taps of the DeltaNet convolution
    int in_row;   // floats per token of the host-side input row (embedding, plus the PLE rows of the Coder)
};

// Environment switches, read once
struct PfFlags {
    int fp8_env;  // TONA_PREFILL_FP8: -1 unset (the architecture's default), 0 off, else on
    bool full, mtp_tail, dn_old, mtp_mega, idx_scalar, attn_scalar, check, fp8_out, wcache;
};
const PfFlags& pf_flags();
class PrefillSequence;
// Whether the dense weights take the FP8 route: TONA_PREFILL_FP8, else the architecture's default
bool pf_fp8_on(const ModelBase& m);
// The dense weight w takes the FP8 route of the prefill (gemmq_fp8); the planner uses the same test to size the weight
// cache
bool pf_fp8_route(const PfW& w, bool fp8_on, int64_t max_in, int64_t max_rows);

// Chunk size and group (tokens per layer-major sweep) of one prefill; layered: layer by layer, else chunk by chunk
struct PfPlan {
    int chunk, group;
    bool layered;
    bool stage = false;   // the KV cache of the running layer lies in the borrowed slots (pinned KV cache only)
    bool wcache = false;  // the unpacked dense weights of the running layer stay in the loan (PrefillRun::wc_get)
};

// One group of tokens handed to PrefillRun
struct PfJob {
    const int32_t* tokens;
    int n;
    int32_t next;  // the token after them (target of the last one for nll, -1: none)
    std::vector<float>* nll;
    int64_t win0;  // first position for the drafter
    const std::function<const float*(int, int)>*
        emb;  // emb(j0, m): embedding and PLE vector of tokens j0 ... j0 + m - 1
    PfPlan plan;
};

// Input (rows of f16, row stride ld) and output (rows of float, row stride ld) of a matmul
struct GemmIn {
    const _Float16* x;
    int ld;
};
struct GemmOut {
    float* y;
    int ld;
};

// Buffers of the DeltaNet phase
struct DnBufs {
    float *qkv, *z, *ba, *conv, *raw;
    _Float16* o16;
};
// Buffers of the MoE phase
struct MoeBufs {
    float *rl, *sw, *G, *U, *O, *sg, *su, *sho, *shi;
    int32_t *sel, *rowmap, *cnt_d;
    _Float16 *xg, *A16, *sa16;
};

// The phase buffers are carved in this order, both when measuring the workspace and in the run
DnBufs carve_dn(Carve& p, const PfDims& D, int C);
MoeBufs carve_moe(Carve& p, const PfDims& D, int C);

// Which experts one batch of the MoE covers and where its copies go: buf selects one of the two copy buffers (stg)
struct BatchPlan {
    size_t b0;
    int m, buf;
    uint8_t* stg[2];
};

// State while the weights of a batch are collected
struct BatchState {
    bool copied = false;  // a copy on the copy stream is under way
    int staged = 0;       // experts read from the file into the staging area (neither in VRAM nor pinned)
};

// Weights of one batch of experts (pointers into the cache, the layer buffer or the staging area)
struct ExpertBatch {
    const uint8_t* wg[MB];
    const uint8_t* wu[MB];
    const uint8_t* wd[MB];
    int m = 0;
};

class PrefillRun;

// Where an architecture writes its records into the program dump (mega_dump.cpp)
struct PfDumpSink {
    std::function<void(const char*, const PfW&)> weight;
    std::function<void(const char*, const void*)> pointer;
};

// Sizing of the unpacked dense weights of a layer (PrefillRun::wc_get): bytes of a weight on the route the prefill
// takes for it. The architecture adds up the weights of its layer kinds.
struct WcSizer {
    const PfFlags& flags;
    int64_t max_in, max_rows;
    size_t w16;  // PfBufs::w16_elems
    bool fp8;
    size_t sliced(const PfW& w, bool as_fp8) const;
    size_t dense(const PfW& w) const;  // the route of PrefillRun::gemmq
    size_t out(const PfW& w) const;    // the route of PrefillRun::gemmq_out
    size_t moe(const PfMoe& m) const;  // router (f16) and the shared expert
};

// Bytes one phase area takes: the carve function runs on a Carve without memory
template <class F>
size_t phase_area(F carve) {
    Carve p{nullptr};
    carve(p);
    return p.off;
}

// What happens to chunk j at layer l, per architecture; the driver (PrefillRun) around it chunks, streams the experts,
// double-buffers and records the events. One implementation per architecture, under gpu/arch/<name>/.
class PrefillSequence {
public:
    virtual ~PrefillSequence() = default;
    // The model's sizes as the prefill uses them; n_feat_layers: layers whose output the drafter reads
    virtual PfDims dims(const ModelBase& m, size_t n_feat_layers) const = 0;
    // Dense weights through FP8 GEMMs unless TONA_PREFILL_FP8 says otherwise
    virtual bool fp8_default() const = 0;
    // Bytes of the largest phase area of this architecture's layers (the driver adds the MoE and the LM head)
    virtual size_t phase_bytes(const ModelBase& m, const PfDims& D, int C, int64_t sstride) const = 0;
    // Unpacked dense weights of the largest layer from layer 2 on (the driver's sizer knows the routes)
    // arch: the architecture's records (MegaPrefillData::pf_arch_)
    virtual size_t wcache_bytes(const std::vector<PfLayer>& layers, const std::any& arch, const MoeShape& h,
                                const WcSizer& w) const = 0;
    // The dense weights in the architecture's own records (residual weights), for the FP8 sizing
    virtual void each_arch_weight(const std::any& arch, const std::function<void(const PfW&)>& f) const = 0;
    // The architecture's records of layer l, and of the head, for the program dump
    virtual void dump_layer(const std::any& arch, size_t l, const PfDumpSink& out) const = 0;
    virtual void dump_head(const std::any& arch, const PfDumpSink& out) const = 0;
    // Buffers of the architecture in the fixed workspace (after the shared rows e, R and sc). With a null `run` it only
    // measures (the carve has no memory); otherwise it keeps the pointers in the run's architecture state (arch_run_)
    virtual void carve_fixed(Carve& c, const ModelBase& m, const PfDims& D, int C, std::any* run) const = 0;
    // The architecture's per-slot state pointers in its records (the pointers of the shared descriptors are
    // retargeted by the caller); tr moves one pointer to the copy of the other slot
    virtual void retarget_state(std::any& arch, const std::function<void(float*&)>& tr) const = 0;
    // The weights of its MTP entries for the program dump
    virtual void dump_mtp(const std::any& arch, const PfDumpSink& out) const = 0;
    virtual void layer(PrefillRun& run, uint32_t l, int j) const = 0;
    // Closes what the layer left open before the chunk switches or the run ends
    virtual void end_chunk(PrefillRun& run) const = 0;
    // Layer l of a chunk whose previous step was another chunk's (layer by layer): the norm that the previous layer
    // of this chunk left in B.mixed16 for it was overwritten by the chunks in between, so it is made again from the
    // chunk's residual stream. Architectures whose layers end with the norm of the next layer need it; the others
    // keep the residual add open between layers and start every layer from it.
    virtual void resume(PrefillRun&, uint32_t) const {}
    // The input of the LM head of chunk j in B.mixed16 (final norm over the residual streams)
    virtual void head_input(PrefillRun& run, int j) const = 0;
    // MTP entries of the prompt that the architecture computes itself (before the generic bookkeeping); returns whether
    // it did. False: the generic one catches the head up through MTP passes.
    virtual bool mtp_entries(PrefillRun& run) const = 0;
    // Shortest prompt (tokens) that goes layer by layer with one chunk of its own size, so that the experts of the
    // next layer are copied ahead of the routing (the chunk-by-chunk plan waits for each batch of experts on PCIe).
    // 0: only prompts of more than one chunk go layer by layer, with the full layered chunk.
    virtual int short_layered_min() const { return 0; }
};
const PrefillSequence& prefill_sequence(const ModelBase& m);

// Runs n tokens in chunks of C. One chunk: all layers one after the other. Several chunks (layer by layer): layers
// 0-1 per chunk, from layer 2 each layer for all chunks; the experts of each layer enter one of two layer buffers
// once, the next one already during the running one (layered prefill arXiv 2510.08055, FreeToken 2608.16157,
// MoE-CORE 2610.01950). Residual streams, features and layer buffers then lie in borrowed slot ranges of whole layers,
// which are refilled with the same experts afterwards; otherwise C = pf_chunk_ in the fixed workspace and the experts
// are used as in decode.
class PrefillRun {
public:
    PrefillRun(Mega& m, const PfJob& job);
    void run();
    // The prompt group by group; set_base(i) is called with the first token of each group
    static void run_groups(Mega& m, const PfJob& all, const std::function<void(int)>& set_base);
    // Checks before a prefill of n tokens; false: nothing to do
    static bool begin(Mega& m, const int32_t* tokens, int n);
    // Chunk size and group for n tokens from the current position (win0: first position for the drafter)
    static PfPlan choose_plan(const Mega& m, int n, int64_t win0);
    static bool plan_fits(const Mega& m, int n, int64_t win0, const PfPlan& p);
    static PfPlan choose_plan_for(const Mega& m, int n, int64_t win0, bool wc);
    static PfPlan balanced_plan(const Mega& m, int n, int64_t win0, PfPlan p);
    static int short_chunk(const Mega& m, int n);
    // Extra expert slots in the workspace (mega_extra.h): given up before a chunk-by-chunk prefill, back after it
    static void extras_out(Mega& m, const PfPlan& plan, int n);
    static void extras_back(Mega& m);

    // The code of an architecture's layer, head input and MTP entries: arch/<name>/prefill_*.cpp specializes this for a
    // tag type of its own. A member, so it sees what the driver and, as the driver is a friend of Mega, Mega expose.
    template <class Arch>
    class Step;

private:
    // prefill_run.cpp: setup, order of the layers, progress
    void ensure_copy_stream();
    uint32_t loan_fit(size_t need) const;
    uint32_t loan_count(size_t need) const;
    void check_loan_end(const uint8_t* q) const;
    void borrow_slots();
    uint8_t* carve_wcache(uint8_t* q);
    // prefill_run.cpp: KV of the running layer in borrowed VRAM (pinned KV cache: the fast attention variant)
    bool staged(uint32_t l) const;
    void stage_copy(const PfLayer& F, int64_t c0, int64_t c1, bool in);
    void stage_in(const PfLayer& F);
    void stage_out(const PfLayer& F);
    void run_layer(uint32_t l, int j);
    void resume_input(const std::vector<std::pair<uint32_t, int>>& order, size_t i);
    void hash_residual(uint32_t l) const;  // TONA_PF_HASH
    void carve_buffers();
    std::vector<std::pair<uint32_t, int>> layer_order() const;
    void step(size_t i, const std::vector<std::pair<uint32_t, int>>& order);
    // prefill_dense.cpp: matmuls on the dense weights, gated residual
    void gemmq(const PfW& w, const _Float16* x, int ldx, float* y, int ldy, int rows_t);
    // Always f16, whatever TONA_PREFILL_FP8 says: the router, the indexer, the position embeddings and the MTP weights
    void gemmq_f16(const PfW& w, const _Float16* x, int ldx, float* y, int ldy, int rows_t);
    void gemmq_out(const PfW& w, const _Float16* x, int ldx, float* y, int ldy, int rows_t);
    void gemmq_fp8(const PfW& w, const _Float16* x, int ldx, float* y, int ldy, int rows_t);
    // Between share_x8(true) and share_x8(false) the FP8 copy of the input stays in B.x8: GEMMs that read the same
    // f16 input (q/k/v, gate/up) convert it once. The caller closes the group before the input is rewritten.
    void share_x8(bool open) {
        x8_share_ = open;
        x8_key_ = {};
    }
    struct X8Key {
        const _Float16* x = nullptr;
        int ld = 0, rows = 0, in = 0;
    };
    X8Key x8_key_;
    bool x8_share_ = false;
    // Unpacked copy of a dense weight (or a slice of it) for the chunks of one layer: the first chunk unpacks into it
    // (fill), the later ones only multiply. Null: not cached, the caller unpacks into B.w16 every time.
    uint8_t* wc_get(const void* w, int64_t r0, int tag, size_t bytes, bool& fill);
    void wc_reset() {
        wc_map_.clear();
        wc_used_ = 0;
    }
    uint8_t* wc_base_ = nullptr;
    size_t wc_size_ = 0;
    size_t wc_used_ = 0, wc_peak_ = 0;
    bool wc_on_ = false;
    std::map<std::tuple<const void*, int64_t, int>, uint8_t*> wc_map_;
    // prefill_moe_run.cpp: MoE with the expert weights streamed in batches
    void moe(uint32_t l, int j);
    MoeBufs moe_buffers();
    void trace_loan(const char* name, const void* p, size_t bytes) const;
    void moe_route(uint32_t l, const MoeBufs& b);
    // x: the tokens' input rows (f16), width: values per token of the rows the experts read (d, or the latent width);
    // ne: experts per token of layer l
    std::vector<int> moe_group(uint32_t l, const MoeBufs& b, const _Float16* x, int width, int ne);
    // arch/nemotron_h/moe_prefill.cpp: the latent MoE of a layer (router, shared expert, latent projections around the
    // shared streaming of the experts); the result is B.y, as for moe()
    void moe_latent(uint32_t l, int j);
    void run_batch_ungated(uint32_t l, const MoeBufs& b, const std::vector<int>& used, const BatchPlan& bp,
                           const ExpertBatch& eb);
    void moe_check(uint32_t l, const MoeBufs& b, int stage, int ne);
    void log_check(uint32_t l, const char* what, int a, int e, int r) const;
    void moe_experts(uint32_t l, int j, const MoeBufs& b, const std::vector<int>& used);
    const uint8_t* expert_weights(uint32_t l, int e, int s, const BatchPlan& bp, BatchState& bs);
    ExpertBatch stage_batch(uint32_t l, const std::vector<int>& used, const BatchPlan& bp);
    void run_batch(uint32_t l, const MoeBufs& b, const std::vector<int>& used, const BatchPlan& bp,
                   const ExpertBatch& eb);
    void issue_layer(uint32_t l);
    bool is_tail(uint32_t l) const;
    // prefill_mtp.cpp: LM head check, MTP entries, bookkeeping at the end
    void head(int j);
    void head_targets(int j, int32_t* tgt);
    const float* mtp_h_at(int64_t q);
    void mtp_publish();
    void inject_features();
    void finish();

    Mega& M;
    PfJob job_;
    PfDims D;
    const PfFlags& flags_;
    hipStream_t st = nullptr;
    int C, nsub;
    bool layered;
    bool nan_reported = false;
    bool mtp_by_pass_ = false;  // the MTP entries of the prompt come from passes (mtp_publish)
    void report_nan_rows(uint32_t l, int32_t rows);
    int64_t check_steps = 0, check_bad = 0;  // TONA_PF_CHECK
    int64_t pos_all;
    int n;         // tokens of the running chunk
    int64_t pos0;  // its first position
    uint8_t* arena;
    const uint8_t* arena_end = nullptr;  // end of the memory the arena may use
    uint8_t* stage = nullptr;    // KV of one layer, rows [0, pos_all + n) at their real offsets; null: not staged
    uint8_t* kv_now_ = nullptr;  // stage while a staged layer runs
    PfBufs B;
    size_t max_sb = 0;
    std::vector<float*> Rp, Fp;
    std::vector<const uint8_t*> res[2];  // per layer buffer: expert -> weights (slot in the cache or layer buffer)
    std::vector<uint8_t*> lbuf[2];       // places of the layer buffers per expert
    std::any arch_run_;                  // the architecture's state across the layers of one run (Step's constructor)
    std::chrono::steady_clock::time_point t_mtp;
};

}  // namespace tona::gpu
