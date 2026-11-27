// The reference backend: forward pass as a megakernel. One persistent kernel per pass (instead of ~4,000 single
// kernels), int8 matvecs, own grid barrier with a time limit. The program is a list of steps (ops) that all blocks
// run together, with a barrier in between. At every router the kernel reports through a mailbox in host memory; the
// CPU chooses the experts, reads misses into a mapped buffer, the GPU copies them into their slots itself.
//
// Verify pass of the speculation: a pass computes up to MEGA_BMAX tokens at consecutive positions (forward), then
// commit(m) decides how many of them count. KV cache, indexer and ring buffers are simply overwritten later at the
// positions of the rejected tokens; the DeltaNet state is committed by the next pass (the m accepted tokens from
// delta, k and alpha of the previous pass, like ReplaySSM, arXiv 2610.13778).
// Same computation as gpu/forward.hip (and thus cpu/ref.cpp); orders of sums may differ.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backend/backend.h"
#include "backend/cpu_cost.h"
#include "backend/env.h"
#include "gpu/matvec.h"
#include "gpu/mega_cache.h"
#include "gpu/mega_data.h"
#include "gpu/mega_pinned.h"
#include "gpu/mega_prefill_types.h"
#include "gpu/mega_seq.h"
#include "gpu/mega_types.h"
#include "io/uring.h"
#include "model/qwen4exp.h"

namespace tona::cpu {
struct ExpertScratch;
}

namespace tona::gpu {
using backend::deterministic;
class ProgramDump;
class ProgramUploader;
class ExpertSelect;

struct MegaOptions {
    int expert_slots = -1;  // experts per layer in the VRAM cache; <= 0: as many as the free VRAM per sysfs allows
                            // minus TONA_VRAM_RESERVE_MB (1024), at most up to TONA_VRAM_GB
    int64_t max_ctx = 0;    // length of the KV cache
    int blocks = 0;         // blocks of the kernel (<= 0: one per CU)
    bool int8 = true;       // matvecs on the int8 path
    std::vector<int64_t> feat_layers;  // inputs of these layers are written per token, for the drafter
    const Gguf* drafter = nullptr;     // DFlash2 drafter (feat_layers = its dflash.target_layers) as a second program
    const MtpHead* mtp = nullptr;      // MTP head as a third program
    bool release_maps = false;         // after the upload remove the model file mappings nobody reads again
};

class Mega final : public backend::Backend,
                   private MegaMtpData,
                   private MegaPrefillData,
                   private MegaTileData,
                   private MegaExpertData {
public:
    Mega(const ModelBase& m, MegaOptions opt);
    ~Mega() override;
    Mega(const Mega&) = delete;
    Mega& operator=(const Mega&) = delete;
    // nb tokens at positions pos() ... pos() + nb - 1; logits [nb][vocab], valid until the next call. Then exactly one
    // commit(m), 1 <= m <= nb: the first m tokens count, pos() advances by m.
    const float* forward(const int32_t* tokens, int nb) override;
    void commit(int m) override;
    // one token: forward + commit(1)
    const std::vector<float>& step(int32_t token);
    // Prefill: n tokens from pos() in chunks of prefill_chunk() tokens with kernels of their own (GEMM on WMMA instead
    // of matvecs, experts directly from VRAM or pinned RAM). Afterwards as after forward + commit(n): states, KV cache
    // and drafter context stand, pos() is n further. It returns no logits, the last token of the prompt goes through
    // forward. nll (optional): per token i < n - 1 the -log p(tokens[i + 1]), to compare with the decode path.
    void prefill(const int32_t* tokens, int n, std::vector<float>* nll = nullptr) override;
    int prefill_chunk() const override { return pf_chunk_; }
    // last prefill: chunk size and group (tokens per layer-major sweep, 0: chunk by chunk)
    int prefill_last_chunk() const { return pf_last_chunk_; }
    int prefill_last_group() const { return pf_last_group_; }
    // back to position 0 (clear the states; the expert cache stays)
    void reset() override;
    // After an abort (exception from forward/draft): wait for the kernel, clear the states, clean up the cache
    void recover() override;
    // A restore point in VRAM: states (DeltaNet, convolutions, open commits) and position. KV cache and drafter need no
    // copy, they are written per position. Only between passes (after commit).
    void save_state() override;
    void load_state() override;
    bool has_state() const override { return snap_valid_; }
    const std::vector<int32_t>& snap_tokens() const override { return snap_tokens_; }
    // State of a sequence in RAM: restore point with the KV rows, indexer pool and drafter ring up to its position.
    // stash reads the restore point (not the running state); KV and pool before snap_pos are unchanged there because
    // only higher positions are written afterwards. unstash sets running state and restore point.
    using HostSnap = backend::HostSnap;
    void stash(HostSnap& s) const override;
    size_t state_bytes() const override {
        size_t b = 0;
        for (const auto& [p, n] : state_bufs_) b += n;
        return b;
    }
    size_t dring_bytes() const override {
        size_t b = 0;
        for (const auto& [p, n] : dring_bufs_) b += n;
        return b;
    }
    size_t stash_bytes() const override;
    void unstash(const HostSnap& s) override;  // throws on foreign sizes (snap_fits)
    // does the stash fit this model and context (sizes of state, KV, pool, drafter ring, position)?
    bool snap_fits(const HostSnap& s) const override;
    // bytes per position in the KV cache: part of the fingerprint of the restore points on the SSD
    size_t kv_row_bytes() const override { return kv_row_; }
    const std::vector<int32_t>& tokens() const override { return tokens_; }  // accepted tokens up to pos()
    // after forward: inputs of feat_layers per token of the pass, [nb][feat_layers * d]
    const float* features() const { return feat_host_.data(); }
    // Drafter: first feed the m last accepted tokens (features of the last forward, positions pos() - m ... pos() - 1),
    // then compute a block of nb tokens at pos() with emb ([nb][d] embeddings). Afterwards draft_logits() [nb -
    // 1][vocab] and draft_gate() [nb - 1][rank] of the positions 1 ... nb - 1.
    void draft(int m, const float* emb, int nb) override;
    const float* draft_logits() const override { return dlog_host_.data(); }
    // MTP: entry p in the KV of the MTP head = token at p with h of the target at p - 1 (h after the last layer, before
    // the head mixer; p = 0: h = 0). mtp_fill catches the entries up to pos() - 1 (h from the last forward or the saved
    // h at pos() - 1) and with next >= 0 computes the entry at pos(): logits [vocab] for pos() + 1 (first draft),
    // otherwise nullptr. mtp_step: tok after the last MTP entry, h from the MTP head itself (further drafts).
    // Draft: argmax of the MTP head (truncated head: over its rows) and its probability
    using MtpPick = backend::MtpPick;

    MtpPick mtp_fill(int32_t next) override;
    MtpPick mtp_step(int32_t tok) override;
    // MTP step of several sequences in one pass: slot slots[i] gets toks[i] after its last MTP entry, like mtp_step per
    // slot (bit-identical); out[i] as its return value. mtp_logits() afterwards per active slot
    void mtp_step_multi(const int* slots, const int32_t* toks, int k, MtpPick* out) override;
    size_t mtp_rows() const override { return mtp_vocab_.empty() ? h_.vocab : mtp_vocab_.size(); }
    // Sampling: mtp_fill/mtp_step copy the draft's logits [mtp_rows()] to the host (row r: token mtp_token(r))
    void mtp_keep_logits(bool on) override;
    const float* mtp_logits() const override {
        return mlog_host_ ? mlog_host_ + (size_t)cur_slot_ * h_.vocab : nullptr;
    }
    void clock_hold(bool on) override;  // TONA_CLOCK_HOLD: the GPU clock stays up while the server is idle
    int32_t mtp_token(size_t r) const override { return mtp_vocab_.empty() ? (int32_t)r : mtp_vocab_[r]; }
    bool has_mtp() const override { return mtp_on_; }
    const float* draft_gate() const override { return dgate_host_.data(); }

    // Several sequences: slots 0 ... seqs() - 1 (TONA_SEQS, default 2) with their own state, KV cache, pool and restore
    // point, allocated at start. use_slot(s) selects the slot all other calls act on (forward, commit, prefill, mtp_*,
    // save/load_state, stash/unstash, reset, recover); slot 0 uses the addresses of the single-sequence case, a lone
    // sequence computes the same program in every slot.
    int seqs() const override { return n_seqs_; }
    int slot() const override { return cur_slot_; }
    // Tokens of the restore point of slot s (empty: none), also for slots other than the active one
    const std::vector<int32_t>& snap_tokens_of(int s) const override;
    void use_slot(int s) override;
    // One pass over several slots: slot slots[k] with n[k] tokens (tok one after the other, at most BMAX together),
    // logits [sum n][vocab] in this order, valid until the next pass. Afterwards use_slot + commit per slot. Matvecs,
    // MoE and head see all rows; steps with state run per slot (program of several sequences).
    const float* forward_multi(const int* slots, const int* n, int k, const int32_t* tok) override;

    int64_t pos() const override { return pos_; }
    int64_t max_ctx() const override { return max_ctx_; }
    const MegaStats& stats() const override { return stats_; }
    backend::Info info() const override { return {"hip-gfx1200", vram_, pin_bytes_, kv_bytes_, kv_host_, slots_}; }
    size_t vram_bytes() const { return vram_; }
    int ops() const { return (int)ops_.size(); }
    const std::vector<std::pair<std::string, float*>>& taps() const { return taps_; }  // diagnostics (TONA_BOLT_TAP)
    int blocks() const { return blocks_; }
    size_t kv_bytes() const { return kv_bytes_; }
    bool kv_host() const { return kv_host_; }
    int slots() const { return slots_; }
    size_t pinned_bytes() const { return pin_bytes_; }
    bool vram_tick() override;  // gpu/mega_vram.cpp: the elastic pool follows the free VRAM (TONA_VRAM_GOVERNOR)
    // Occupancy per HIP: at most this many blocks at once; LDS per block, VGPRs (the larger of the two kernels)
    int max_blocks() const { return max_blocks_; }
    int lds_bytes() const { return lds_bytes_; }
    int vgprs() const { return vgprs_; }
    std::string batch_info() const;
    ChainDiag diag();
    // with TONA_MEGA_PROF=1: mean time per pass by kind of step in us, descending; reset: start a new average after it
    std::vector<std::pair<std::string, double>> profile(bool reset = false);

private:
    friend class ProgramBuilder;
    friend class Qwen4ExpProgram;   // gpu/arch/qwen4exp: the Coder's buffers and programs
    friend class Qwen35MoeProgram;  // gpu/arch/qwen35moe: Ornith's
    friend class NemotronHState;    // gpu/arch/nemotron_h/state.cpp: the state of the Mamba-2 layers
    friend class LatentMoe;         // gpu/arch/nemotron_h/moe.cpp: Bolt's latent MoE block
    friend class NemotronHProgram;  // gpu/arch/nemotron_h: Bolt's
    friend class ProgramDump;
    friend class ProgramUploader;
    friend class ExpertSelect;
    friend class PrefillRun;  // prefill_*.cpp: the prefill driver reads weights, caches and buffers

    // --- allocation and uploads (mega_alloc.cpp)
    const void* upload(const TensorInfo& t);
    const float* upload_f32(const TensorInfo& t);
    const void* upload_q8_0(const TensorInfo& t);  // BF16 -> Q8_0
    float* alloc(size_t floats);
    // One vector of length n per token of the pass, one after the other (distance n, noted for the matvec tasks)
    float* alloc_b(size_t n);
    Q8Vec alloc_q8_b(size_t n);
    int64_t stride_of(const void* p) const;
    // Per-token buffers: start -> (bytes, bytes per token); for the row offset in the program of several sequences
    std::map<uintptr_t, std::pair<size_t, int64_t>> rows_;
    int64_t row_bytes_at(const void* p) const;
    float* alloc_state(size_t floats);  // cleared by reset()

    // --- construction (mega_init.cpp; init_kernels, init_barrier and the launches in mega_launch.cpp)
    void check_shapes() const;
    void init_kernels();
    void begin_load_progress() const;
    void warm_experts();
    void drop_page_cache(bool release_maps);
    void init_barrier();
    void report_prediction() const;
    void write_profile() const;
    void release_device();
    void release_host();

    // --- programs (mega_build*.cpp, mega_upload.cpp, mega_seq.cpp)
    void build();
    void upload_prog(std::vector<MegaOp>& ops, const std::vector<MegaTask>& tasks, void*& d_ops, void*& d_tasks);

    // --- selection and experts (mega_select*.cpp, mega_experts.cpp, mega_prefetch.cpp)
    // Router logits of the nb tokens -> union of the experts, slots, weights into the mailbox, read misses
    void select(uint32_t l, int nb);
    // CPU mode, after select: compute the experts of cpu_list_, then read up to fill_n_ of them for the cache
    void cpu_experts(uint32_t l, int nb);
    void compute_cpu_experts(uint32_t l, int nb);
    void refill_cache(uint32_t l);
    bool share_wait_ = true;  // shared expert during host.wait (TONA_SHARE_WAIT=0: in moe.gate_up)
    int pass_nb_ = 1;         // tokens of the running pass (commit at most this many)

    // --- running passes (mega_launch.cpp, mega_serve.cpp)
    void launch(int nb);
    void serve_pass(int nb);  // serve the mailbox of the running pass (router, CPU experts), continue yields
    // Yield of the kernel (MegaParams::yield_ms): set? Then restart at the reported step
    bool resume_if_yielded(uint64_t pass_seq);
    bool
    finish_yields();  // after the sync of the pass: continue a yield after the last router (true: fetch copies again)
    struct LaunchArgs {
        int nb = 0, o1 = 0;
        const void *ops = nullptr, *tasks = nullptr;
        MegaParams* dparams = nullptr;
        unsigned long long* prof = nullptr;
    } last_launch_;
    void launch_program(const LaunchArgs& a, int o0, const MegaParams* hparams);
    void start_kernel(const LaunchArgs& a, int o0);
    void check_aborts();
    std::string abort_info() const;  // " (step k: kind label)" of the first abort in the last launch
    std::string slowest_step_info(unsigned k) const;
    void log_slow_steps(const std::vector<unsigned long long>& t) const;  // TONA_MEGA_STALL_MS with TONA_MEGA_PROF
    std::string arrival_spread(size_t o, unsigned long long from) const;
    std::string gap_hist_info() const;
    std::string cause_info() const;              // the wait that counted the abort (mega_abort.cpp)
    std::string arrival_info(unsigned k) const;  // TONA_MEGA_ABORT_TRACE: blocks that reached the barrier of step k
    std::string straggler_info(const std::vector<std::string>& labels) const;
    void check_slots_fit(int nb) const;
    void check_draft_args(int m, const float* emb, int nb) const;
    void check_tokens(const int32_t* tokens, int nb) const;
    void stage_embeddings(const int32_t* tokens, int nb);
    void begin_pass(const int32_t* tokens, int nb);
    float* fetch_logits(int nb);
    void record_step_times(int nb);
    void write_barrier_trace(int nb);
    int check_multi(const int* slots, const int* n, int k, const int32_t* tok) const;
    void fetch_multi_logits(int nb);
    void fetch_draft(int nb);
    void stage_multi(const int* slots, const int* n, int k, const int32_t* tok);

    // --- several sequences (mega_seq.cpp)
    std::vector<SeqBuf> seq_bufs_;
    void seq_buf(void* dev0, size_t bytes, void* host0 = nullptr) {
        SeqBuf b;
        b.dev[0] = static_cast<uint8_t*>(dev0), b.host[0] = static_cast<uint8_t*>(host0), b.bytes = bytes;
        seq_bufs_.push_back(b);
    }
    void* seq_tr(const void* p, int from, int to) const;  // address in the buffer of slot `from` -> same place in `to`
    bool in_seq(const void* p) const;
    size_t alloc_seq_buffer(SeqBuf& b, int s);
    void alloc_mtp_seq_bufs();
    void alloc_seqs();       // copies for slot 1 ..., before the expert slots
    void build_seq_progs();  // programs per slot and for several sequences
    void retarget_slot_pointers(int from, int to);
    void* upload_slot_ops(const char* name, const std::vector<MegaOp>& ops, const std::vector<std::string>& labels,
                          int s);
    void expand_op(SeqExpand& e, size_t k, int sl);
    void expand_program(SeqExpand& e);
    void publish_x_program(const std::vector<MegaOp>& ops, MegaOp*& host, void*& dev);
    int n_seqs_ = 1, cur_slot_ = 0;
    std::vector<SeqHost> seq_host_;
    void swap_seq(SeqHost& h);
    // Program of several sequences: steps with state per slot (seq); pointers to per-token buffers are set by
    // forward_multi per pass to the first row of the slot (xpatch_)
    std::vector<MegaOp> xops_;
    std::vector<std::string> xlabels_;
    std::vector<XPatch> xpatch_;
    MegaOp* xops_host_ = nullptr;  // pinned
    void* d_xops_ = nullptr;
    // the same for the MTP head (mtp_step_multi); the pick per slot in buffers of its own (mpick_x_ + 4 * slot)
    std::vector<MegaOp> xmops_;
    std::vector<std::string> xmlabels_;
    std::vector<XPatch> xmpatch_;
    MegaOp* xmops_host_ = nullptr;
    void* d_xmops_ = nullptr;
    float *mpick_x_ = nullptr, *mpickx_host_ = nullptr;
    // h for the next MTP step per slot (device): a row of mtp_hout_ or, if a pass of another slot would overwrite it,
    // its copy in mtp_hsave_ + slot * hcd
    const float* mtp_hsrc_[MEGA_SMAX] = {};
    float* mtp_hsave_ = nullptr;
    bool last_multi_ = false, multi_pass_ = false;  // multi_pass_: the running pass computes several sequences
    // OEA (TONA_OEA_K0, default 7, 0: off): see ExpertSelect; TONA_OEA_ALL=1 also in passes of one sequence (measuring
    // mode)
    int oea_k0_ = (int)env::integer("TONA_OEA_K0", 7);
    bool oea_all_ = env::flag("TONA_OEA_ALL");
    int64_t fw_row0_ = 0;  // first row of the active slot in the last pass (h of the target for MTP)

    // --- state
    const ModelBase& m_;
    const MoeShape shape_, &h_;  // what the toolbox needs of the model; the rest stays in the architecture
    int64_t pos_ = 0, max_ctx_;
    int slots_, blocks_, max_blocks_ = 0, lds_bytes_ = 0, vgprs_ = 0;
    int kblocks_[3] = {}, kmax_[3] = {}, kvgpr_[3] = {};  // k_mega<1>, <4>, <8>
    static int kix(int nb) { return nb == 1 ? 0 : nb <= 4 ? 1 : 2; }
    bool int8_;
    MegaMailbox *mbox_host_ = nullptr, *mbox_dev_ = nullptr;
    uint8_t* staging_dev_ = nullptr;
    uint64_t seq_ = 0;
    std::vector<LayerCache> cache_;
    std::vector<int32_t> tokens_;  // accepted tokens
    std::vector<int32_t> batch_;   // tokens of the last forward
    int pending_commit_ = 0, parity_ = 0;
    bool awaiting_commit_ = false;

    // Program: ops and matvec tasks
    std::vector<MegaOp> ops_;
    std::vector<MegaTask> tasks_;
    std::vector<std::string> labels_;
    std::map<uintptr_t, std::pair<size_t, int64_t>> strides_;  // buffer -> (bytes, distance per token in elements)
    std::vector<std::pair<void*, size_t>> state_bufs_;
    std::vector<std::pair<void*, size_t>> dring_bufs_;  // KV ring of the drafter (K and V per layer)
    uint8_t* snap_dring_ = nullptr;  // pinned: drafter ring of the restore point (in VRAM it cost ~1.5 expert slots)
    std::vector<uint8_t*> kv_hptr_;  // KV per QSA layer, host pointer (empty: KV in VRAM)
    uint8_t* snap_ = nullptr;
    bool snap_valid_ = false;
    int64_t snap_pos_ = 0;
    std::vector<int32_t> snap_tokens_;
    int snap_commit_ = 0, snap_parity_ = 0;
    unsigned long long *d_prof_ = nullptr, *d_arr_ = nullptr;
    struct MegaTap* tap_ = nullptr;  // TONA_TAP_OUT: per-layer residual of the pass for the oracle (gpu/mega_tap.h)
    FILE* arr_fp_ = nullptr;
    std::vector<double> prof_us_;
    int64_t prof_tokens_ = 0, prof_width_sum_ = 0;
    void *d_ops_ = nullptr, *d_tasks_ = nullptr, *d_bar_ = nullptr;
    MegaParams *params_host_ = nullptr, *d_params_ = nullptr;  // host copy pinned
    int wall_khz_ = 0;
    int64_t aborts_seen_ = 0;

    // Device buffers (selection; the rest only in the program)
    float* e_ = nullptr;  // per token [embedding | PLE vector], 2 * d
    float* logits_ = nullptr;
    float* in_host_ = nullptr;  // pinned: embedding and PLE vectors to upload
    PinnedVec logits_host_, logits1_;
    std::vector<float> step_out_;
    std::vector<int64_t> feat_layers_;
    float* feat_ = nullptr;
    PinnedVec feat_host_;

    // --- MTP head: third program (mega_mtp.cpp; data in MegaMtpData)
    MtpPick pick() const;
    void mtp_sync_vocab(int32_t next);
    void head_row(int32_t t, uint8_t* dst) const;
    void upload_mtp_experts();
    void mtp_experts_alloc(bool q8);
    void mtp_experts_matrix(int k, bool q8, uint8_t*& stage);
    const float* mtp_h_of(int64_t q) const;
    // kv_only: only the KV entries (prompt tokens, catching up without a draft), otherwise up to the draft
    void mtp_pass(const int32_t* toks, int nb, int64_t pos0, const float* const* hs, bool want, bool kv_only = false);
    const float* mtp_stage_inputs(const int32_t* toks, int nb, const float* const* hs);
    void mtp_save_other_h();
    MtpPick multi_pick(int slot) const;
    void mtp_run_multi(const int* slots, int k, const int64_t* pos);
    void ensure_profile_buffer();
    void fetch_pick(int nb);
    void mtp_profile_step();
    void mtp_stage_multi(const int* slots, const int32_t* toks, int k, int64_t* pos);
    void mtp_check_slot(int slot, int32_t tok, int64_t pos, const MtpPick& got, const float* hin, const float* hout);
    void mtp_multi_check(const int* slots, const int32_t* toks, int k, const int64_t* pos, const MtpPick* out);
    std::vector<std::pair<uint8_t*, float*>> qsa_bufs() const;
    size_t kv_vram_bytes(size_t i, size_t n, int64_t pos) const;  // of buffer i of n: bytes of pos rows in VRAM
    void ensure_snap_dring(size_t bytes), save_drafter_ring();
    bool restore_state(const HostSnap& s);

    // --- drafter: second program (feeding up to dinject_end_, then the block)
    const Gguf* drafter_ = nullptr;
    std::vector<MegaOp> dops_;
    std::vector<MegaTask> dtasks_;
    std::vector<std::string> dlabels_;
    void *d_dops_ = nullptr, *d_dtasks_ = nullptr;
    int dinject_end_ = 0, drank_ = 0;
    const void* head_w_ = nullptr;
    float *e2_ = nullptr, *dlog_ = nullptr, *dgate_ = nullptr, *in2_host_ = nullptr;
    MegaParams *dparams_host_ = nullptr, *d_dparams_ = nullptr;
    PinnedVec dlog_host_, dgate_host_;

    uint8_t* staging_ = nullptr;
    // CPU mode (TONA_CPU_EXPERTS, default on): the CPU computes misses from RAM (cpu/experts), the cache is filled on
    // the side: per layer up to fill_n_ of the misses, read after computing, entered in the next pass
    bool cpu_mode_ = false;
    int fill_n_ = 0, fill_cap_ = 0;
    uint8_t *fill_ = nullptr, *fill_dev_ = nullptr;  // [layer][2 halves][fill_cap_][slot bytes], pinned
    size_t fill_stride_ = 0;
    uint8_t* cpu_io_ = nullptr;  // pinned, mapped: x of the experts as int8 + metadata (GPU -> CPU), y (CPU -> GPU)
    uint8_t* cpu_io_dev_ = nullptr;
    size_t cpu_xm_off_ = 0, cpu_y_off_ = 0;
    // Measurement TONA_PREDICT_STATS: router of layer l + 1 on the x of layer l; hits against the real choice
    std::vector<std::vector<float>> router_f_;  // per layer [E][d], unpacked on demand
    std::vector<std::vector<int>> pred_;        // per token the predicted experts (by rank) for the next layer
    uint32_t pred_layer_ = UINT32_MAX;
    double pred_hit_[4] = {}, pred_n_ = 0;    // top 10 of the real choice among the top 10/16/24/32 of the prediction
    double pred_ms_hit_ = 0, pred_ms_n_ = 0;  // misses of the union that were in the top-24 prediction
    void predict_next(uint32_t l, int nb);
    // Prefetching (TONA_PREFETCH = at most this many per layer, 0: off): at layer l the router of l + 1 (on the GPU in
    // the same matvec) predicts experts; pinned misses are copied by a stream of their own via DMA into slots of l + 1
    // while l computes
    int prefetch_n_ = 8;
    // From this many tokens per pass the dense matvecs run on WMMA (TONA_WIDE_MIN, 0 = off): faster than int8 from 6
    // (research/measurements/2026-10-25-width-8) but rounds differently; off with TONA_DETERMINISTIC=1
    int wide_min_ = (int)env::integer("TONA_WIDE_MIN", deterministic() ? 0 : 6);
    bool last_drafter_ = false;         // last launched: program of the drafter
    std::unordered_set<int> pre_keep_;  // predicted for l + 1 and already cached: not evicted by prefetching
    std::vector<uint8_t, PinnedAlloc<uint8_t>> pre_check_buf_;  // TONA_PREFETCH_CHECK
    hipStream_t pre_st_ = nullptr;
    std::vector<hipEvent_t> pre_ev_;
    std::vector<char> pre_busy_;             // per layer: copies in flight
    std::vector<std::vector<int>> pre_got_;  // per layer: prefetched, not yet evaluated
    std::vector<int> pre_cand_;              // candidates for the next layer, by rank
    void prefetch_choose(uint32_t l, int nb);
    bool is_prefetchable(uint32_t layer, int e) const;
    void vote_predicted(int nb, std::vector<int>& cnt, std::vector<float>& best) const;
    void prefetch_issue(uint32_t l);
    void prefetch_settle(uint32_t l, const std::vector<int>& uni);
    void prefetch_check(uint32_t l);
    void prefetch_drain();
    std::vector<int> cpu_list_;     // experts the CPU computes in this layer
    std::vector<float> cpu_w_;      // [nb][cpu_list_]
    std::vector<int> fill_choice_;  // from cpu_list_ for the cache
    std::unique_ptr<cpu::ExpertScratch> cpu_scratch_;
    void load_profile(), prefill_cache();
    // Pinned stage: as many experts as RAM is left (see pin_budget), in pinned host memory; the GPU reads them over
    // PCIe
    void pin_experts();
    int64_t pin_budget() const, pin_budget_from_limit() const;
    void pin_layer(uint32_t l, const std::vector<int>& chosen);
    std::vector<std::vector<int>> choose_pinned(int64_t budget) const;
    size_t all_expert_bytes() const;
    std::vector<PinLayer> pin_;
    std::vector<void*> pin_allocs_;
    size_t kv_bytes_ = 0;   // KV cache int8 of all QSA layers
    bool kv_host_ = false;  // in pinned RAM instead of VRAM (with kv_split_: from some cell on)
    KvSplit kv_split_;      // by position, when planned (kv_place.h)
    size_t pin_bytes_ = 0;
    // Cost per expert for the planner (us): CPU (the base; cpu_cost_ follows the measured time under CPU load, not with
    // TONA_DETERMINISTIC, TONA_PLAN, TONA_CPU_ADAPT=0), GPU directly from RAM, hit in VRAM. A learned CPU time without
    // a dead zone was slower (research/measurements/2026-10-24-planner); 80 -> 60 with the 512-bit kernels
    // (2026-10-25-cpu-experts); c_gd_us_: PCIe 5.0 x16, scaled to the GPU's link (gpu/pcie_link.h)
    double c_cpu_us_ = 60, c_gd_us_ = 15, c_hit_us_ = 12;
    backend::CpuCost cpu_cost_;
    int gd_experts_ = 0;  // GPU directly from pinned RAM, in the running layer
    std::vector<void*> allocs_;
    size_t vram_ = 0;
    MegaStats stats_;
    // --- prefill (gpu/prefill.hip; data in MegaPrefillData; the chunks run in PrefillRun, prefill_run.h)
    size_t prefill_arena_bytes(int chunk) const;
    int64_t pf_max_in() const;    // widest input of the dense weights in the prefill (multiple of 64)
    int64_t pf_max_rows() const;  // most rows of the dense weights in the prefill
    // Borrowed slot ranges: layers whose slots the prefill occupies, with their saved cache
    // Layered prefill possible: all experts pinned (otherwise the prefill would have to read from the NVMe)
    bool pf_layered_ok() const;
    uint32_t pf_loan_layers() const;
    size_t pf_loan_cap() const;
    size_t pf_layered_need(int m, int C, int64_t pos, int64_t win0, bool stage, bool wc) const;
    size_t pf_wcache_bytes() const;  // unpacked dense weights of the largest layer, kept for all chunks (0: off)
    bool pf_stage_on(int64_t end = INT64_MAX) const;
    void pf_return_loans();
    // Embedding (and PLE vector) per token on the CPU, for prefill and decode. seq: history (at least the last
    // ple_ngram tokens, otherwise up to the start), from off the n tokens; out [n][h_.in_row()]
    void embed_tokens(const int32_t* seq, size_t off, int n, float* out) const;
    // n-gram table of the Coder (null: none), its rows through io_uring (TONA_PLE_IO=mmap: through the mapping)
    const TensorInfo* ple_table_ = nullptr;
    mutable std::unique_ptr<DirectRows> ple_io_;
    mutable std::once_flag ple_io_once_;
    void inject_draft(const float* feat_rows, int64_t p0, int m);
    std::unique_ptr<ProgramDump> dump_;  // TONA_MEGA_DUMP_PROGRAM, only during build
    int hb_fd_ = -1;                     // heartbeat for scripts/gpu-watch.sh: TONA_HEARTBEAT=file, one line per layer
    void beat(const char* phase, int layer);
};

}  // namespace tona::gpu
