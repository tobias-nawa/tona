// CPU mode of the expert selection: hits are computed by the GPU, experts refilled in the previous pass enter their
// slots first, the other misses go to the CPU or, if pinned in RAM, to the GPU reading them over PCIe.
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <thread>

#include "backend/env.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_select.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

std::vector<double>& prof_gpu_experts() {
    static std::vector<double> v;
    return v;
}

// The places u of the union get the plan: hit/gd for the GPU (slot or pinned place), cu for the CPU, ms all misses
void ExpertSelect::plan_cpu() {
    // TONA_TEST_STALL_MS: the CPU hangs once (layer 10, 20th call); tests the time limit of the kernel against the
    // desktop
    static const int stall_ms = (int)env::integer("TONA_TEST_STALL_MS", 0);
    static int stall_calls = 0;
    if (stall_ms > 0 && l_ == 10 && ++stall_calls == 20)
        std::this_thread::sleep_for(std::chrono::milliseconds(stall_ms));
    g_.prefetch_settle(l_, uni_);
    std::unordered_set<int> keep;  // needed in this pass: not evicted
    C_.credit(uni_);
    for (int e : uni_) {
        auto it = C_.where.find(e);
        if (it == C_.where.end()) continue;
        keep.insert(e);
        C_.touch(it);
    }
    const int n_miss = install_pending(keep);
    mb_.fill_half = C_.fill_half;
    backend::CpuPlan P;
    P.ew.assign(mb_.ew, mb_.ew + MEGA_BMAX * MEGA_UMAX);
    P.um.assign(mb_.umask, mb_.umask + MEGA_UMAX);
    std::memset(mb_.ew, 0, sizeof mb_.ew);
    std::memset(mb_.umask, 0, sizeof mb_.umask);
    static const float drop = (float)env::number("TONA_MISS_DROP", 0.0);
    backend::classify_misses(P, {uni_, nb_, MEGA_UMAX}, drop, [&](int e) { return C_.where.contains(e); });
    g_.stats_.dropped += P.dropped;
    split_misses(P);
    publish_plan(P, n_miss);
    choose_fill(P);
    C_.tick(nb_);
}

// Put the refilled experts into their slots, in the order of their place in the buffer; returns how many (moe.copy
// copies the first n_miss)
int ExpertSelect::install_pending(std::unordered_set<int>& keep) {
    int n_miss = 0;
    for (int e : C_.pending) {
        // All slots taken by hits of this pass (few slots, 4 tokens): drop the remaining refilled ones, they exist
        // only in the buffer
        const int slot = C_.take_slot(&keep);
        if (slot < 0) break;
        g_.elastic_.require_live(C_, (int)l_, slot);  // the never-remap rule: a write only into mapped memory
        C_.add_front(e, slot);
        C_.installed.push_back(e);
        keep.insert(e);
        mb_.miss_slot[n_miss++] = slot;
        g_.stats_.fills++;
    }
    C_.pending.clear();
    return n_miss;
}

// Planner (backend/expert_split.h) with the pinned stage and the measured costs of this machine
void ExpertSelect::split_misses(backend::CpuPlan& P) {
    static const char* plan = env::text("TONA_PLAN");  // "gpu": all pinned ones to the GPU, "cpu": all to the CPU
    const PinLayer* pin = g_.pin_.empty() ? nullptr : &g_.pin_[l_];
    const backend::SplitCost cost{g_.c_hit_us_, g_.c_gd_us_, g_.cpu_cost_.us(), plan};
    backend::split_misses(P, {uni_, nb_, MEGA_UMAX}, cost, [&](int e) { return pin && pin->idx[e] >= 0; });
    if (P.cost_skipped) g_.cpu_cost_.skipped();
}

// Mailbox: the GPU's places in the order of the union (no longer needed for bit equality since the fixed-point sum in
// op_moe_sum, but free). Slots of the GPU-direct ones can be negative (pinned memory below the slots), hence a marker
// of their own.
void ExpertSelect::publish_plan(const backend::CpuPlan& P, int n_miss) {
    std::vector<int64_t> slot_of(uni_.size(), 0);
    std::vector<char> on_gpu(uni_.size(), 0);
    const PinLayer* pin = g_.pin_.empty() ? nullptr : &g_.pin_[l_];
    for (int u : P.hit) slot_of[u] = C_.where[uni_[u]].first, on_gpu[u] = 1;
    for (int u : P.gd) slot_of[u] = pin->slot0 + pin->idx[uni_[u]], on_gpu[u] = 1;
    int n_u = 0;
    for (size_t u = 0; u < uni_.size(); u++) {
        if (!on_gpu[u]) continue;
        mb_.slot[n_u] = (int32_t)slot_of[u];
        mb_.umask[n_u] = P.um[u];
        for (int b = 0; b < nb_; b++) mb_.ew[b * MEGA_UMAX + n_u] = P.ew[b * MEGA_UMAX + u];
        n_u++;
    }
    g_.cpu_list_.clear();
    for (int u : P.cu) g_.cpu_list_.push_back(uni_[u]);
    const int nc = (int)P.cu.size();
    g_.cpu_w_.assign((size_t)nb_ * nc, 0.0f);
    for (int k = 0; k < nc; k++)
        for (int b = 0; b < nb_; b++) g_.cpu_w_[b * nc + k] = P.ew[b * MEGA_UMAX + P.cu[k]];
    g_.stats_.expert_hits += (int64_t)P.hit.size();
    g_.stats_.expert_misses += (int64_t)P.ms.size();
    g_.stats_.cpu_experts += nc;
    g_.stats_.gpu_direct += (int64_t)P.gd.size();
    g_.gd_experts_ = (int)P.gd.size();
    if (!g_.prof_us_.empty() && env::flag("TONA_PROF_LAYERS")) {
        auto& nu = prof_gpu_experts();
        nu.resize(std::max<size_t>(nu.size(), 2 * l_ + 2), 0.0);
        nu[2 * l_] += n_u, nu[2 * l_ + 1] += (double)P.gd.size();  // computed by the GPU, of them read over PCIe
    }
    mb_.n_u = n_u;
    mb_.n_miss = n_miss;
    mb_.n_cpu = nc;
}

// For the cache: the misses used by most tokens, then those with the largest weight; while slots are free, up to the
// size of the buffer
void ExpertSelect::choose_fill(backend::CpuPlan& P) {
    const backend::FillLimits lim{g_.fill_n_, g_.fill_cap_, (int)C_.free_slots.size()};
    backend::choose_fill(P, {uni_, nb_, MEGA_UMAX}, lim, g_.fill_choice_);
}

}  // namespace tona::gpu
