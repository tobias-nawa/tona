// Buffers of the op programs, the layout of the expert caches and the host buffers for the CPU experts.
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

#include "backend/env.h"
#include "cpu/experts.h"
#include "gpu/clock_hold.h"
#include "gpu/evict_counter.h"
#include "gpu/expert_parts.h"
#include "gpu/kfd_smi.h"
#include "gpu/mega_build.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"
#include "gpu/pcie_link.h"

namespace tona::gpu {

ProgramBuilder::ProgramBuilder(Mega& g)
    : g_(g),
      h_(g.h_),
      m_(g.m_),
      d_((int)g.h_.d),
      ff_((int)g.h_.expert_ff),
      ne_((int)g.h_.n_expert_used),
      tg_{&g.ops_, &g.tasks_, &g.labels_} {}

uint8_t* ProgramBuilder::alloc_kv(size_t bytes) { return alloc_kv(bytes, g_.kv_host_); }

uint8_t* ProgramBuilder::alloc_kv(size_t bytes, bool host) {
    if (!host) return reinterpret_cast<uint8_t*>(g_.alloc(bytes / sizeof(float) + 16));
    void* h;
    HIPCHECK(trace::host_malloc(&h, bytes + 64, hipHostMallocMapped | hipHostMallocNonCoherent));
    std::memset(h, 0, bytes + 64);
    g_.pin_allocs_.push_back(h);
    void* dptr;
    HIPCHECK(hipHostGetDevicePointer(&dptr, h, 0));
    return static_cast<uint8_t*>(dptr);
}

KvBuf ProgramBuilder::kv_buffer(int64_t hot) {
    const size_t row = (size_t)g_.kv_row_, cells = (size_t)g_.max_ctx_;
    KvBuf k;
    if (hot > 0 && hot < (int64_t)cells) {
        k.hot = hot;
        k.kv = alloc_kv((size_t)hot * row, false);
        k.cold = alloc_kv((cells - (size_t)hot) * row, true);
        g_.kv_hptr_.push_back(static_cast<uint8_t*>(g_.pin_allocs_.back()));
        g_.seq_buf(k.kv, (size_t)hot * row);
        g_.seq_buf(k.cold, (cells - (size_t)hot) * row, g_.kv_hptr_.back());
        return k;
    }
    const bool host = hot <= 0;
    k.kv = alloc_kv(cells * row, host);
    if (g_.kv_host_) g_.kv_hptr_.push_back(host ? static_cast<uint8_t*>(g_.pin_allocs_.back()) : nullptr);
    g_.seq_buf(k.kv, cells * row, host ? g_.kv_hptr_.back() : nullptr);
    return k;
}

// Device buffer with the contents of src (zeros without src), counted as VRAM of the engine
void* ProgramBuilder::dev_buf(const void* src, size_t bytes) {
    void* p;
    HIPCHECK(trace::dev_malloc(&p, bytes + kMatvecReadPad));
    HIPCHECK(hipMemset(p, 0, bytes + kMatvecReadPad));
    if (src) HIPCHECK(hipMemcpy(p, src, bytes, hipMemcpyHostToDevice));
    g_.allocs_.push_back(p);
    g_.vram_ += bytes;
    return p;
}

// Mailbox, the per-pass parameters and the host buffers for embedding and features
void ProgramBuilder::alloc_io() {
    const int nfeat = (int)g_.feat_layers_.size() * d_;
    if (nfeat) g_.feat_ = g_.alloc_b(nfeat), g_.feat_host_.resize((size_t)MEGA_BMAX * nfeat);
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.mbox_host_), mailbox_bytes(expert_cap(g_.h_.n_expert)),
                                hipHostMallocMapped | hipHostMallocCoherent));
    std::memset(g_.mbox_host_, 0, mailbox_bytes(expert_cap(g_.h_.n_expert)));
    g_.mrows_ = (g_.h_.n_expert > MEGA_MAX_EXPERTS)
                    ? MailboxRows{mailbox_router<MEGA_MAX_EXPERTS_WIDE>(g_.mbox_host_),
                                  mailbox_pred<MEGA_MAX_EXPERTS_WIDE>(g_.mbox_host_), MEGA_MAX_EXPERTS_WIDE}
                    : MailboxRows{g_.mbox_host_->router, g_.mbox_host_->pred, MEGA_MAX_EXPERTS};
    HIPCHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&g_.mbox_dev_), g_.mbox_host_, 0));
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.in_host_), (size_t)MEGA_BMAX * h_.in_row() * sizeof(float),
                                hipHostMallocDefault));
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.params_host_), sizeof(MegaParams), hipHostMallocDefault));
    std::memset(g_.params_host_, 0, sizeof(MegaParams));
    g_.params_host_->aborted = &g_.mbox_dev_->aborted;
    // Deadline per pass (default 1,500 ms, 0 = none): below the 2 s limit of the driver
    g_.params_host_->budget_ms = (int32_t)env::integer("TONA_MEGA_BUDGET_MS", 1500);
    // Yield instead of abort after waiting this long for the CPU (default 100, 0 = abort as before): the kernel
    // holds the desktop up for no longer (research/23-waiting-in-kernel)
    g_.params_host_->yield_ms = (int32_t)env::integer("TONA_MEGA_YIELD_MS", 100);
    // Eviction credit (wait_clock.h): the blocks may ask the host how long the queues were evicted, if the KFD counter
    // exists
    evict_check_test(g_.params_host_->budget_ms);
    evict_credit_init();
    // Diagnostics (TONA_KFD_SMI_LOG, set by TONA_DIAG): the evictions of this process with their trigger and time
    smi::start_log(kfd_gpu_id());
    HIPCHECK(trace::dev_malloc(reinterpret_cast<void**>(&g_.d_params_), sizeof(MegaParams)));
    HIPCHECK(hipMemset(g_.d_params_, 0, sizeof(MegaParams)));
}

// Slot size per layer (gate | up | down, each padded to 256 bytes). The slots themselves come last (alloc_slots), so
// the tasks get placeholder addresses (layer + 1) << 52 until then.
void ProgramBuilder::init_cache_layout() {
    g_.cache_.resize(h_.n_layer);
    g_.pf_.assign(m_.n_blocks(), {});  // one descriptor per block of the trunk
    // TONA_EXPERT_POLICY: how the expert cache picks its victims; lru (default) or classlru (class_lru.h)
    const char* const policy = env::text("TONA_EXPERT_POLICY");
    if (policy && std::strcmp(policy, "lru") != 0 && std::strcmp(policy, "classlru") != 0)
        throw std::runtime_error(std::format("gpu::Mega: TONA_EXPERT_POLICY=\"{}\": expected lru or classlru", policy));
    const bool classlru = policy && std::strcmp(policy, "classlru") == 0;
    for (uint32_t l = 0; l < h_.n_layer; l++) {
        const MoeWeights& S = m_.moe_weights(l);
        LayerCache& C = g_.cache_[l];
        // Per layer from the tensors (expert_parts.h): its own widths, and no gate part for an ungated layer
        const SlotLayout SL = slot_layout(S);
        C.off_up = SL.off_up;
        C.off_down = SL.off_down;
        C.slot_bytes = SL.slot_bytes;
        C.n_used = (int)m_.expert_used(l);
        C.slots = reinterpret_cast<uint8_t*>((uintptr_t)(l + 1) << 52);
        if (classlru) C.order.emplace((int)g_.experts_.experts(l));
    }
}

// Misses: the CPU reads them here, the GPU copies them into their slots itself
void ProgramBuilder::setup_host_io() {
    max_slot_ = 0;
    for (const auto& C : g_.cache_) max_slot_ = std::max(max_slot_, C.slot_bytes);
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.staging_), max_slot_ * std::min(UMAX, ne_ * MEGA_BMAX),
                                hipHostMallocMapped | hipHostMallocCoherent));
    HIPCHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&g_.staging_dev_), g_.staging_, 0));
    // CPU mode: cpu/experts must know all formats of the experts
    g_.cpu_mode_ = g_.int8_ && !fake_experts() && env::integer("TONA_CPU_EXPERTS", 1) != 0;
    // the CPU experts compute in the order of row_q8 with 8 lanes (bit-identical)
    if (g_.cpu_mode_ && env::integer("TONA_MEGA_LANES_EXP", 8) != 8)
        throw std::runtime_error("gpu::Mega: CPU experts need TONA_MEGA_LANES_EXP = 8 (TONA_CPU_EXPERTS=0 for others)");
    for (uint32_t l = 0; l < h_.n_layer && g_.cpu_mode_; l++) {
        const MoeWeights& S = m_.moe_weights(l);
        // The CPU experts compute gate, up, SwiGLU, down or, without a gate tensor, up, squared ReLU, down
        // (cpu/experts.h)
        for (const TensorInfo* t : {S.up_exps, S.down_exps})
            g_.cpu_mode_ = g_.cpu_mode_ && t && cpu::expert_type_supported(t->type);
        if (S.gate_exps) g_.cpu_mode_ = g_.cpu_mode_ && cpu::expert_type_supported(S.gate_exps->type);
    }
    if (g_.cpu_mode_) setup_cpu_mode();
    // The shared expert (gate, up, weight) and PLE do not depend on the routing: with share_wait_ the other blocks
    // compute them while block 0 waits for the CPU (host.wait without a barrier after it), otherwise they sit in
    // moe.gate_up
    g_.share_wait_ = env::integer("TONA_SHARE_WAIT", 1) != 0;
}

// Stream and events of the prefetching are created now, not at the first prefetch in the middle of a pass (a new
// stream creates a queue while k_mega waits for the CPU)
void ProgramBuilder::setup_prefetch() {
    HIPCHECK(hipStreamCreateWithFlags(&g_.pre_st_, hipStreamNonBlocking));
    // Two idle streams (queues) on top of the null stream, the prefetch stream and the prefill's copy stream: the
    // prefill of 2,000 tokens runs 6.5-9 % faster with them, one gives it only from the third request on, none not at
    // all (research/measurements/2026-11-12-prefill-bimodal.md). Why the number of queues matters is not known.
    // Process-wide and never destroyed, like the prefetch stream: they only hold their queues.
    static std::vector<hipStream_t> spacers;
    // At most 8: a typo must not create thousands of queues
    if (spacers.empty()) spacers.resize((size_t)std::clamp<int64_t>(env::integer("TONA_QUEUE_SPACERS", 2), 0, 8));
    for (hipStream_t& s : spacers)
        if (!s) HIPCHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    // TONA_CLOCK_HOLD=1: the hold's spin runs on the first spacer (the queue count stays as it is), =2: on a stream of
    // its own (one more queue); other values: no hold (server/hold_controller.h)
    if (const int64_t hold = env::integer("TONA_CLOCK_HOLD", 0); hold == 1 || hold == 2)
        clock_hold_init(hold == 1 && !spacers.empty() ? (void*)spacers[0] : nullptr);
    g_.pre_ev_.resize(h_.n_layer);
    for (auto& ev : g_.pre_ev_) HIPCHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
    g_.pre_busy_.assign(h_.n_layer, 0);
    g_.pre_got_.assign(h_.n_layer, {});
    // TONA_PREFETCH_CHECK: pinned check buffer (a copy into ordinary memory is pinned by the runtime per userptr,
    // freeing it evicts the queues)
    if (env::flag("TONA_PREFETCH_CHECK")) g_.pre_check_buf_.resize(max_slot_);
}

namespace {

// The planner's price of a GPU-direct expert on the link of the GPU in use (gpu/pcie_link.h), logged once so a card
// in a narrower slot shows in the start log
double link_gd_cost_us(double reference_us) {
    int dev = 0;
    char bus[32] = {};
    if (hipGetDevice(&dev) != hipSuccess || hipDeviceGetPCIBusId(bus, sizeof bus, dev) != hipSuccess)
        return reference_us;
    std::string id = bus;
    std::transform(id.begin(), id.end(), id.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    const PcieLink link = read_pcie_link("/sys/bus/pci/devices/" + id);
    const double us = gd_cost_us(reference_us, link);
    if (link.known())
        std::fprintf(stderr, "PCIe link: %.0f GT/s x%d, GPU-direct expert priced at %.1f us (TONA_GD_US overrides)\n",
                     link.gts, link.width, us);
    return us;
}

}  // namespace

void ProgramBuilder::setup_cpu_mode() {
    // Refilling after misses only while slots are free: with prefetching (TONA_PREFETCH) it cost more PCIe than it
    // brought in hits (server decode +3-6 %, research/measurements/2026-10-24-misses)
    g_.fill_n_ = (int)env::integer("TONA_FILL", 0);
    // The next layer's experts are predicted from the router of that layer on this layer's input: only where the next
    // MoE layer follows right behind (the model says so)
    g_.prefetch_n_ = (int)env::integer("TONA_PREFETCH", m_.predicts_next_router() ? g_.prefetch_n_ : 0);
    if (g_.prefetch_n_ > 0) setup_prefetch();
    g_.c_gd_us_ = env::number("TONA_GD_US", link_gd_cost_us(g_.c_gd_us_));
    g_.c_cpu_us_ = env::number("TONA_CPU_US", g_.c_cpu_us_);
    // A fixed plan (TONA_DETERMINISTIC, TONA_PLAN) stays fixed; TONA_CPU_ADAPT=2 adapts anyway (test: the text must not
    // change)
    const int64_t adapt = env::integer("TONA_CPU_ADAPT", 1);
    g_.cpu_cost_ = backend::CpuCost(g_.c_cpu_us_,
                                    adapt == 2 || (adapt != 0 && !backend::deterministic() && !env::text("TONA_PLAN")));
    g_.fill_cap_ = std::max(g_.fill_n_, 4);  // while slots are free, more is refilled (start-up)
    g_.fill_stride_ = 2 * max_slot_ * g_.fill_cap_;
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.fill_), g_.fill_stride_ * h_.n_layer,
                                hipHostMallocMapped | hipHostMallocCoherent));
    HIPCHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&g_.fill_dev_), g_.fill_, 0));
    g_.cpu_xm_off_ = (size_t)MEGA_BMAX * d_;
    g_.cpu_y_off_ = g_.cpu_xm_off_ + (size_t)MEGA_BMAX * d_ / 32 * sizeof(Q8Meta);
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.cpu_io_),
                                g_.cpu_y_off_ + (size_t)MEGA_BMAX * d_ * sizeof(int64_t),
                                hipHostMallocMapped | hipHostMallocCoherent));
    HIPCHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&g_.cpu_io_dev_), g_.cpu_io_, 0));
    g_.cpu_scratch_ = std::make_unique<cpu::ExpertScratch>();
}

}  // namespace tona::gpu
