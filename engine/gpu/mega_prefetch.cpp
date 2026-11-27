// CPU experts and prefetching: after the router message the CPU computes the experts the GPU does not, copies pinned
// experts of the next layer into VRAM ahead of time, and refills the cache.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <numeric>

#include "backend/env.h"
#include "backend/nan_guard.h"
#include "cpu/experts.h"
#include "gpu/expert_parts.h"
#include "gpu/mega.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

namespace {

// TONA_CPU_COST_LOG=file (measuring mode): one line per call, wall-clock seconds, experts, elapsed us, estimate ratio
void log_cpu_cost(size_t n, double elapsed_us, double ratio) {
    static FILE* const f = [] {
        const char* path = env::text("TONA_CPU_COST_LOG");
        FILE* const fp = path ? std::fopen(path, "w") : nullptr;
        if (fp) std::setvbuf(fp, nullptr, _IOLBF, 0);
        return fp;
    }();
    if (!f) return;
    const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::fprintf(f, "%.3f %zu %.1f %.2f\n", now, n, elapsed_us, ratio);
}
}  // namespace

void Mega::compute_cpu_experts(uint32_t l, int nb) {
    if (cpu_list_.empty()) return;
    const MoeWeights& S = m_.moe_weights(l);
    const LayerCache& C = cache_[l];
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<cpu::ExpertW> E;
    int n_pinned = 0;
    // From the layer's tensors: the width in (the latent space for Bolt) and the expert width, and no gate part for an
    // ungated layer (the slot has up | down then)
    const TensorInfo* gate = S.gate_exps;
    const int64_t d = (int64_t)S.up_exps->dims[0], ff = (int64_t)S.up_exps->dims[1];
    const uint32_t t_gate = gate ? gate->type : 0;
    for (int e : cpu_list_) {
        const int j = pin_.empty() ? -1 : pin_[l].idx[e];
        if (j >= 0) {  // pinned: from there, it was dropped from the page cache
            n_pinned++;
            const uint8_t* b = pin_[l].host + j * C.slot_bytes;
            E.push_back({t_gate, S.up_exps->type, S.down_exps->type, gate ? b : nullptr, b + C.off_up, b + C.off_down});
        } else {
            E.push_back({t_gate, S.up_exps->type, S.down_exps->type,
                         gate ? gate->data + e * bytes_per_expert(gate) : nullptr,
                         S.up_exps->data + e * bytes_per_expert(S.up_exps),
                         S.down_exps->data + e * bytes_per_expert(S.down_exps)});
        }
    }
    // Measuring mode TONA_CPU_FAKE=1: do not compute the CPU experts (upper bound for faster CPU kernels; the output is
    // meaningless)
    static const bool fake_cpu = env::text("TONA_CPU_FAKE") != nullptr;
    if (!fake_cpu)
        cpu::experts_ffn_fix({E, d, ff, reinterpret_cast<const int8_t*>(cpu_io_),
                              reinterpret_cast<const cpu::Q8Meta*>(cpu_io_ + cpu_xm_off_), nb, cpu_w_.data()},
                             reinterpret_cast<int64_t*>(cpu_io_ + cpu_y_off_), *cpu_scratch_);
    const double dt = seconds_since<std::chrono::steady_clock>(t0);
    stats_.cpu_s += dt;
    // Experts that are read from the page cache would count their page faults as slowness of the CPU
    if (!fake_cpu && n_pinned == (int)E.size()) cpu_cost_.observe(n_pinned, dt * 1e6);
    log_cpu_cost(E.size(), dt * 1e6, cpu_cost_.ratio());
}

// Refill while the GPU finishes the layer; entered in the next pass
void Mega::refill_cache(uint32_t l) {
    const MoeWeights& S = m_.moe_weights(l);
    LayerCache& C = cache_[l];
    const ExpertParts P = expert_parts(S, C);
    const auto& parts = P.t;
    const auto& offs = P.off;
    C.fill_half ^= 1;
    uint8_t* dst = fill_ + l * fill_stride_ + C.fill_half * (fill_stride_ / 2 / C.slot_bytes) * C.slot_bytes;
    const auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < (int)fill_choice_.size(); i++)
        for (int k = 0; k < P.n; k++) {
            const int e = fill_choice_[i];
            const int j = pin_.empty() ? -1 : pin_[l].idx[e];
            const uint8_t* src =
                j >= 0 ? pin_[l].host + j * C.slot_bytes + offs[k] : parts[k]->data + e * bytes_per_expert(parts[k]);
            std::memcpy(dst + i * C.slot_bytes + offs[k], src, bytes_per_expert(parts[k]));
        }
    stats_.fill_s += seconds_since<std::chrono::steady_clock>(t0);
    stats_.upload_bytes += (double)fill_choice_.size() * C.slot_bytes;
    C.pending = fill_choice_;
}

void Mega::cpu_experts(uint32_t l, int nb) {
    const uint32_t want = (uint32_t)(seq_ + l + 1);
    compute_cpu_experts(l, nb);
    __atomic_store_n(&mbox_host_->cpu_done, want, __ATOMIC_RELEASE);
    // Candidates only now: before, the GPU waited for this computation (mailbox); it is needed only now
    prefetch_choose(l, nb);
    prefetch_issue(l);
    if (!fill_choice_.empty()) refill_cache(l);
}

// Wait for all copies (before prefill, reset, after an abort)
void Mega::prefetch_drain() {
    if (!pre_st_) return;
    HIPCHECK(hipStreamSynchronize(pre_st_));
    std::fill(pre_busy_.begin(), pre_busy_.end(), 0);
    for (auto& g : pre_got_) g.clear();
}

// TONA_PREFETCH_CHECK: compare the copies byte for byte with the RAM
void Mega::prefetch_check(uint32_t l) {
    const LayerCache& C = cache_[l];
    std::vector<uint8_t, PinnedAlloc<uint8_t>>& buf = pre_check_buf_;  // allocated at load time (hipHostMalloc blocks)
    for (int e : pre_got_[l]) {
        auto it = C.where.find(e);
        if (it == C.where.end()) continue;  // evicted again
        // Not hipMemcpy: it waits for the running Mega kernel, which waits for us (GPU hang 2026-10-24)
        HIPCHECK(hipMemcpyAsync(buf.data(), C.slots + (size_t)it->second.first * C.slot_bytes, C.slot_bytes,
                                hipMemcpyDeviceToHost, pre_st_));
        HIPCHECK(hipStreamSynchronize(pre_st_));
        if (std::memcmp(buf.data(), pin_[l].host + (size_t)pin_[l].idx[e] * C.slot_bytes, C.slot_bytes) != 0)
            throw std::runtime_error(std::format("prefetch: layer {} expert {} wrong in the slot", l, e));
        stats_.prefetch_checked++;
    }
}

void Mega::prefetch_settle(uint32_t l, const std::vector<int>& uni) {
    if (pre_busy_.empty() || !pre_busy_[l]) return;
    const auto t0 = std::chrono::steady_clock::now();
    HIPCHECK(hipEventSynchronize(pre_ev_[l]));
    stats_.prefetch_wait_s += seconds_since<std::chrono::steady_clock>(t0);
    pre_busy_[l] = 0;
    static const bool check = env::text("TONA_PREFETCH_CHECK") != nullptr;
    if (check) prefetch_check(l);
    for (int e : pre_got_[l]) stats_.prefetch_used += std::find(uni.begin(), uni.end(), e) != uni.end();
    pre_got_[l].clear();
}

// Per token the top 16 of the predicted router: votes and best logit per expert
void Mega::vote_predicted(int nb, std::vector<int>& cnt, std::vector<float>& best) const {
    std::vector<int> ix(cnt.size());
    for (int b = 0; b < nb; b++) {
        const float* r = mrows_.pred_row(b);
        std::iota(ix.begin(), ix.end(), 0);
        // The GPU clamps the copy in the mailbox (op_host_wait); the comparison stays valid for NaN anyway
        const auto v = [&](int a) { return tona::nan::clean_one(r[a]); };
        std::partial_sort(ix.begin(), ix.begin() + 16, ix.end(), [&](int a, int c) { return v(a) > v(c); });
        for (int k = 0; k < 16; k++) cnt[ix[k]]++, best[ix[k]] = std::max(best[ix[k]], v(ix[k]));
    }
}

// Pinned in RAM, not in the cache and not already queued for a refill
bool Mega::is_prefetchable(uint32_t layer, int e) const {
    const LayerCache& N = cache_[layer];
    return !N.where.contains(e) && pin_[layer].idx[e] >= 0 &&
           std::find(N.pending.begin(), N.pending.end(), e) == N.pending.end();
}

// Candidates for layer l + 1: per token the top 16 of the predicted router, by the number of tokens, then logit
void Mega::prefetch_choose(uint32_t l, int nb) {
    pre_cand_.clear();
    if (prefetch_n_ <= 0 || l + 1 >= h_.n_layer || pin_.empty()) return;
    const int E = (int)experts_.experts(l + 1);
    std::vector<int> cnt(E, 0);
    std::vector<float> best(E, -INFINITY);
    vote_predicted(nb, cnt, best);
    const LayerCache& N = cache_[l + 1];
    // Predicted ones already in the cache: prefetching must not evict them (TONA_PRE_KEEP=0: it may)
    static const bool keep_on = env::integer("TONA_PRE_KEEP", 1) != 0;
    pre_keep_.clear();
    for (int e = 0; e < E && keep_on; e++)
        if (cnt[e] && N.where.contains(e)) pre_keep_.insert(e);
    for (int e = 0; e < E; e++)
        if (cnt[e] && is_prefetchable(l + 1, e)) pre_cand_.push_back(e);
    std::sort(pre_cand_.begin(), pre_cand_.end(),
              [&](int a, int c) { return cnt[a] > cnt[c] || (cnt[a] == cnt[c] && best[a] > best[c]); });
    if ((int)pre_cand_.size() > prefetch_n_) pre_cand_.resize(prefetch_n_);
}

// Copies of the candidates into slots of l + 1 (free or at the end of the LRU list); layer l + 1 runs only after the
// next router, the GPU does not read its slots now
void Mega::prefetch_issue(uint32_t l) {
    if (pre_cand_.empty() || !pre_st_) return;
    const uint32_t n = l + 1;
    LayerCache& N = cache_[n];
    // Prefetched ones go to the back of the LRU list: a good half is never needed and leaves first instead of
    // evicting useful ones (server decode rust +3 %, research/measurements/2026-10-24-misses); TONA_PRE_BACK=0 to the
    // front. Choose the slots first so that no prefetched expert evicts another of the same round
    static const bool back = env::integer("TONA_PRE_BACK", 1) != 0;
    std::vector<int> slots;
    for (size_t i = 0; i < pre_cand_.size(); i++) {
        const int s = N.take_slot(&pre_keep_);
        if (s < 0) {
            pre_cand_.resize(i);
            break;
        }
        slots.push_back(s);
    }
    for (size_t i = 0; i < pre_cand_.size(); i++) {
        const int e = pre_cand_[i], slot = slots[i];
        elastic_.require_live(N, (int)n, slot);
        // Enqueue first, then register: if the copy fails, where must not claim an expert in an empty slot
        if (hipMemcpyAsync(N.slots + (size_t)slot * N.slot_bytes, pin_[n].host + (size_t)pin_[n].idx[e] * N.slot_bytes,
                           N.slot_bytes, hipMemcpyHostToDevice, pre_st_) != hipSuccess) {
            for (size_t k = i; k < slots.size(); k++) N.free_slots.push_back(slots[k]);
            pre_cand_.clear();
            throw std::runtime_error("gpu::Mega: prefetching failed");
        }
        if (back)
            N.add_back(e, slot);
        else
            N.add_front(e, slot);
        pre_got_[n].push_back(e);
    }
    HIPCHECK(hipEventRecord(pre_ev_[n], pre_st_));
    pre_busy_[n] = 1;
    stats_.prefetched += (int64_t)pre_cand_.size();
    stats_.upload_bytes += (double)pre_cand_.size() * N.slot_bytes;
    pre_cand_.clear();
}

// TONA_PREDICT_STATS: top 32 experts of layer l + 1 from its router on the x the CPU received for layer l
void Mega::predict_next(uint32_t l, int nb) {
    const int E = (int)experts_.experts(l + 1), d = (int)h_.d;
    if (router_f_.empty()) router_f_.resize(h_.n_layer);
    auto& W = router_f_[l + 1];
    if (W.empty()) {
        const TensorInfo& t = *m_.moe_weights(l + 1).router;
        W.resize((size_t)E * d);
        for (int e = 0; e < E; e++)
            dequantize_row(t.type, t.data + (size_t)e * type_row_bytes(t.type, d), W.data() + (size_t)e * d, d);
    }
    const int8_t* q = reinterpret_cast<const int8_t*>(cpu_io_);
    const auto* meta = reinterpret_cast<const cpu::Q8Meta*>(cpu_io_ + cpu_xm_off_);
    pred_.assign(nb, {});
    std::vector<float> x(d), lg(E);
    for (int b = 0; b < nb; b++) {
        for (int i = 0; i < d; i++) x[i] = q[(size_t)b * d + i] * meta[((size_t)b * d + i) / 32].d;
        for (int e = 0; e < E; e++) {
            const float* w = W.data() + (size_t)e * d;
            float s = 0;
            for (int i = 0; i < d; i++) s += w[i] * x[i];
            lg[e] = s;
        }
        tona::nan::report("predicted router logits", tona::nan::clean(lg.data(), lg.size()),
                          std::format("layer {}", l + 1));
        auto& p = pred_[b];
        p.resize(E);
        std::iota(p.begin(), p.end(), 0);
        std::partial_sort(p.begin(), p.begin() + 32, p.end(), [&](int a, int c) { return lg[a] > lg[c]; });
        p.resize(32);
    }
    pred_layer_ = l + 1;
}

}  // namespace tona::gpu
