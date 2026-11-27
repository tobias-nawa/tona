// Construction and destruction of the Mega backend.
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "backend/env.h"
#include "backend/expert_profile.h"
#include "backend/rope_yarn.h"
#include "cpu/experts.h"
#include "gpu/arch_program.h"
#include "gpu/cu_mask.h"
#include "gpu/gtt.h"
#include "gpu/mega.h"
#include "gpu/mega_dump.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_tap.h"
#include "gpu/mega_util.h"
#include "model/progress.h"

namespace tona::gpu {

// TONA_FREE_CUS=n (multiple of 4): n of the GPU's CUs stay with the desktop. The desktop draws on the same GPU; if the
// megakernel occupies all CUs, the driver has to preempt it for every frame, and when that fails it resets the whole
// GPU. The mask must be set before the HIP runtime starts, hence a constructor that runs before main; the CU count
// therefore comes from the KFD topology (gpu/cu_mask.h), not from HIP.
int free_cus() { return (int)std::max<int64_t>(0, env::integer("TONA_FREE_CUS", 0)); }

namespace {
__attribute__((constructor)) void set_cu_mask() {
    const int f = free_cus();
    if (f <= 0 || env::text("HSA_CU_MASK")) return;
    const int cus = first_gpu_cus();
    const std::string mask = cu_mask_string(cus, f);
    if (mask.empty()) {
        // Ignoring it silently would leave every CU to the kernel, the case TONA_FREE_CUS exists to prevent
        std::fprintf(stderr, "TONA_FREE_CUS=%d ignored: GPU has %d CUs (KFD topology), no CU mask set\n", f, cus);
        return;
    }
    setenv("HSA_CU_MASK", mask.c_str(), 1);
}
}  // namespace

namespace {

// An out-of-memory error of a host buffer says what to do if the host memory is GTT (limited, gpu/gtt.h)
template <class F>
void with_gtt_hint(F&& f) {
    try {
        f();
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        const std::string hint = what.contains("out of memory") ? gtt_full_hint() : std::string();
        if (hint.empty()) throw;
        throw std::runtime_error(what + hint);
    }
}

// Experts per layer as the layer's own tensors have them (the last dimension of its expert matrices; the same in every
// layer unless the model says otherwise, as Bolt's pruned layers do)
backend::ExpertLayout expert_layout_of(const ModelBase& m) {
    std::vector<uint32_t> n(m.moe_shape().n_layer);
    for (uint32_t l = 0; l < n.size(); l++) n[l] = (uint32_t)m.moe_weights(l).up_exps->dims[2];
    return backend::ExpertLayout(std::move(n));
}

}  // namespace

Mega::Mega(const ModelBase& m, MegaOptions opt)
    : MegaExpertData(expert_layout_of(m)),
      m_(m),
      shape_(m.moe_shape()),
      h_(shape_),
      max_ctx_(opt.max_ctx),
      // TONA_EXPERT_SLOTS=N: a fixed count where the option is not given (the server: tests with few slots on a small
      // model)
      slots_(std::min<int>(opt.expert_slots > 0 ? opt.expert_slots : (int)env::integer("TONA_EXPERT_SLOTS", 0),
                           m.moe_shape().n_expert)),
      blocks_(opt.blocks),
      int8_(opt.int8),
      feat_layers_(std::move(opt.feat_layers)),
      drafter_(opt.drafter) {
    mtp_ = opt.mtp;
    mtp_on_ = (opt.mtp != nullptr || m.mtp_source() == MtpSource::EmbeddedInModel) && !env::flag("TONA_NO_MTP");
    if (h_.ple_ngram) ple_table_ = m.ngram_input()->table();
    kernels_ = &kernels_of(m);
    check_shapes();
    init_kernels();
    log_host_memory();
    begin_load_progress();
    with_gtt_hint([&] { build(); });
    dump_.reset();
    with_gtt_hint([&] { warm_experts(); });
    drop_page_cache(opt.release_maps);
    init_barrier();
}

// The kernels assume fixed shapes of the model
void Mega::check_shapes() const {
    if (mtp_ && (mtp_->n_used * BMAX > UMAX || 64 + 2 * (int)mtp_->n_expert + 32 + (int)mtp_->n_used > SM_FLOATS))
        throw std::runtime_error("gpu::Mega: the MTP experts do not fit UMAX or the LDS of the routing");
    if (drafter_ && feat_layers_ != drafter_->get_int_array("dflash.target_layers"))
        throw std::runtime_error("gpu::Mega: feat_layers do not match the drafter");
    if (mtp_ && mtp_->rope_base != h_.rope_base && rope::yarn_active(h_.rope_base).factor > 1.0)
        throw std::runtime_error(
            "gpu::Mega: the MTP head has another RoPE base than the model, the YaRN table is shared");
    // 262,144: native context of the model (model card, rope_yarn.h); beyond it the 12 QSA layers rotate by angles
    // they were not trained on
    if (max_ctx_ > (int64_t)rope::YARN_ORIG_CTX && rope::yarn_active(h_.rope_base).factor <= 1.0)
        std::fprintf(
            stderr,
            "warning: --ctx %lld is beyond the native context of 262144 tokens and TONA_ROPE_YARN is off; quality "
            "beyond 262144 is not measured\n",
            (long long)max_ctx_);
    check_arch_shapes(m_, experts_);
}

// Progress: bytes of the source tensors (the experts of the target run separately)
void Mega::begin_load_progress() const {
    double total = 0;
    for (const auto& t : m_.gguf().tensors())
        if (!t.name.contains("_exps")) total += (double)t.bytes;
    if (mtp_)
        for (const auto& t : mtp_->gguf.tensors()) total += (double)t.bytes;
    if (drafter_)
        for (const auto& t : drafter_->tensors()) total += (double)t.bytes;
    progress::begin("Weights into VRAM", total);
}

// Initial contents of the expert caches: the VRAM cache from a profile, the pinned stage in host RAM
void Mega::warm_experts() {
    use_.assign(experts_.total(), 0);
    load_profile();
    if (!profile_.empty() && !fake_experts()) {
        progress::begin("Expert cache in VRAM", h_.n_layer);
        prefill_cache();
    }
    if (cpu_mode_) {
        progress::begin("Pinning experts in RAM", h_.n_layer);
        pin_experts();
    }
}

namespace {

// A file mapping that was the source of a hipMemcpy keeps the process exposed to queue evictions while page migration
// invalidates its pages (research/measurements/2026-11-01-slow-decode.md). Removes the mappings nobody reads any
// more: the dense tensors and the output head of the target (head_row reads rows with pread) and the whole MTP file.
// Kept: the experts (the CPU reads the unpinned ones in place) and the embeddings (rows are read on the CPU).
void release_upload_mappings(const Gguf& g, const TensorInfo* const (&keep)[2], const Gguf* mtp) {
    size_t freed = 0;
    for (const auto& t : g.tensors()) {
        const bool kept = t.name.contains("_exps") || &t == keep[0] || &t == keep[1];
        if (!kept) freed += g.release(t);
    }
    if (mtp)
        for (int i = 0; i < mtp->files(); i++) freed += mtp->release_file(i);
    std::fprintf(stderr, "Releasing page cache: %.2f GB of file mappings removed from the address space\n",
                 (double)freed / 1e9);
}

}  // namespace

// Everything now in VRAM leaves the page cache: the experts need the RAM (the CPU computes them from RAM, refilling
// reads from there). From now on the CPU reads only experts, embedding and PLE rows.
void Mega::drop_page_cache(bool release_maps) {
    progress::begin("Releasing page cache");
    if (env::flag("TONA_KEEP_CACHE")) return;
    for (const auto& t : m_.gguf().tensors())
        if (!t.name.contains("_exps") && &t != m_.embedding_table() && &t != ple_table_) m_.gguf().drop(t);
    if (release_maps) {
        const TensorInfo* keep[] = {m_.embedding_table(), ple_table_};
        release_upload_mappings(m_.gguf(), keep, mtp_ ? &mtp_->gguf : nullptr);
    }
}

Mega::~Mega() {
    (void)hipDeviceSynchronize();
    report_prediction();
    report_tokhist(stats_);
    write_profile();
    release_device();
    release_host();
}

// TONA_TOKHIST=1: how many tokens use each union expert, per pass width (single-sequence passes, all layers); printed
// every 500 passes (the server does not run the destructor on a signal)
void report_tokhist(const backend::Stats& stats) {
    if (!env::text("TONA_TOKHIST")) return;
    for (int w = 2; w <= 8; w++) {
        int64_t n = 0;
        for (int k = 1; k <= w; k++) n += stats.tokhist[w][k];
        if (n == 0) continue;
        std::fprintf(stderr, "tokhist W=%d experts=%lld", w, (long long)n);
        for (int k = 1; k <= w; k++) std::fprintf(stderr, " %d:%lld", k, (long long)stats.tokhist[w][k]);
        std::fprintf(stderr, "\n");
    }
}

void Mega::report_prediction() const {
    if (pred_n_ <= 0) return;
    std::fprintf(stderr,
                 "Prediction of layer l+1 from x of l: top-10 hit with 10/16/24/32 candidates: %.1f / %.1f / %.1f / "
                 "%.1f %%; misses in the top 24: %.1f %% of %.0f\n",
                 100 * pred_hit_[0] / pred_n_, 100 * pred_hit_[1] / pred_n_, 100 * pred_hit_[2] / pred_n_,
                 100 * pred_hit_[3] / pred_n_, 100 * pred_ms_hit_ / std::max(1.0, pred_ms_n_), pred_ms_n_);
}

// TONA_EXPERT_PROFILE_OUT: at exit the sum of the loaded and the newly counted profile (format:
// backend/expert_profile.h)
void Mega::write_profile() const {
    const char* out = env::text("TONA_EXPERT_PROFILE_OUT");
    if (!out) return;
    std::vector<uint32_t> sum = use_;
    if (profile_.size() == sum.size())
        for (size_t i = 0; i < sum.size(); i++) sum[i] += profile_[i];
    (void)backend::write_expert_profile(out, experts_, sum);
}

void Mega::release_device() {
    if (const std::string h = d_bar_ ? gap_hist_info() : ""; !h.empty()) std::fprintf(stderr, "%s\n", h.c_str());
    for (void* p : allocs_) (void)trace::dev_free(p);
    for (void* p : {d_ops_, d_tasks_, d_bar_, static_cast<void*>(d_params_), static_cast<void*>(d_prof_), d_dops_,
                    d_dtasks_, static_cast<void*>(d_dparams_), static_cast<void*>(d_arr_), d_mops_, d_mtasks_, d_mkops_,
                    d_mktasks_, static_cast<void*>(d_mparams_), d_mprof_})
        if (p) (void)trace::dev_free(p);
    if (arr_fp_) std::fclose(arr_fp_);
    delete tap_;
    for (hipEvent_t e : pre_ev_) (void)hipEventDestroy(e);
    for (hipEvent_t e : pf_ev_)
        if (e) (void)hipEventDestroy(e);
    for (hipEvent_t e : pf_lev_)
        if (e) (void)hipEventDestroy(e);
    for (hipStream_t st : {pre_st_, pf_cs_})
        if (st) (void)hipStreamDestroy(st);
}

void Mega::release_host() {
    if (hb_fd_ >= 0) close(hb_fd_);
    for (void* p : {static_cast<void*>(fill_), static_cast<void*>(cpu_io_)})
        if (p) (void)trace::host_free(p);
    for (void* p : pin_allocs_) (void)trace::host_free(p);
    release_registered_host_memory();
    if (mtp_pf_e_host_) (void)trace::host_free(mtp_pf_e_host_);
    for (void* p : {static_cast<void*>(pf_host_), static_cast<void*>(pf_hostc_)})
        if (p) (void)trace::host_free(p);
    for (void* p : {static_cast<void*>(staging_), static_cast<void*>(params_host_), static_cast<void*>(in_host_),
                    static_cast<void*>(mbox_host_), static_cast<void*>(dparams_host_), static_cast<void*>(in2_host_),
                    static_cast<void*>(mparams_host_), static_cast<void*>(mtp_in_host_)})
        if (p) (void)trace::host_free(p);
}

}  // namespace tona::gpu
