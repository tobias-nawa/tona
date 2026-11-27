// Where the misses of a layer are computed when the CPU takes part (CPU mode): hits stay with the GPU, misses pinned in
// host RAM go round robin to the side that would finish earlier (the GPU reading them over the link, or the CPU), the
// others to the CPU; and which misses are read into the cache afterwards. Decisions on plain arrays; the mailbox, the
// cache and the statistics are the backend's.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "backend/cpu_cost.h"

namespace tona::backend {

// The experts of one pass split by where they are computed (places u of the union)
struct CpuPlan {
    std::vector<float> ew;             // [nb][umax] weights, as the union builder wrote them
    std::vector<uint32_t> um;          // [umax] mask of tokens that use the place
    std::vector<int> hit, gd, cu, ms;  // hits (GPU, from the cache), GPU direct (pinned), CPU, all misses
    int dropped = 0;                   // misses left out (TONA_MISS_DROP)
    bool cost_skipped = false;         // pinned experts went to the GPU and none to the CPU: the cost estimate decays
};

// The union and its weights: places u hold experts uni[u], ew is [nb][umax]
struct SplitIn {
    const std::vector<int>& uni;
    int nb, umax;
};

// Sum of the weights per token
inline std::vector<float> weight_sums(const CpuPlan& P, const SplitIn& in) {
    std::vector<float> sums(in.nb, 0.0f);
    for (int b = 0; b < in.nb; b++)
        for (size_t u = 0; u < in.uni.size(); u++) sums[b] += P.ew[(ptrdiff_t)b * in.umax + (ptrdiff_t)u];
    return sums;
}

// Scale the remaining weights of each token back to its old sum
inline void stretch_weights(CpuPlan& P, const SplitIn& in, const std::vector<float>& old_sums) {
    const std::vector<float> now = weight_sums(P, in);
    for (int b = 0; b < in.nb; b++) {
        if (now[b] <= 0) continue;
        for (size_t u = 0; u < in.uni.size(); u++) P.ew[(ptrdiff_t)b * in.umax + (ptrdiff_t)u] *= old_sums[b] / now[b];
    }
}

// Hits stay with the GPU. drop > 0 (TONA_MISS_DROP=w, experiment, lossy): misses whose weight is below w for all tokens
// are left out and the other weights per token stretched to the old sum (Research Square rs-11268552).
template <class Resident>
void classify_misses(CpuPlan& P, const SplitIn& in, float drop, Resident&& resident) {
    std::vector<float> wsum0;
    if (drop > 0) wsum0 = weight_sums(P, in);
    for (size_t u = 0; u < in.uni.size(); u++) {
        if (resident(in.uni[u])) {
            P.hit.push_back((int)u);
            continue;
        }
        float mw = 0;
        for (int b = 0; b < in.nb && drop > 0; b++) mw = std::max(mw, P.ew[(ptrdiff_t)b * in.umax + (ptrdiff_t)u]);
        if (drop > 0 && mw < drop) {
            for (int b = 0; b < in.nb; b++) P.ew[(ptrdiff_t)b * in.umax + (ptrdiff_t)u] = 0;
            P.dropped++;
            continue;
        }
        P.ms.push_back((int)u);
    }
    if (drop > 0) stretch_weights(P, in, wsum0);
}

// What the planner assumes: per hit and per GPU-direct expert on the GPU, per CPU expert (us); plan: nullptr, or
// "gpu": every pinned miss to the GPU, anything else: every pinned miss to the CPU (TONA_PLAN)
struct SplitCost {
    double hit_us, gd_us, cpu_us;
    const char* plan;
};

// Planner: misses pinned in RAM go round robin to the side that would finish earlier; pinned(e): e is in the pinned
// stage
template <class Pinned>
void split_misses(CpuPlan& P, const SplitIn& in, const SplitCost& c, Pinned&& pinned) {
    double t_gpu = P.hit.size() * c.hit_us, t_cpu = 0;
    bool pinned_on_cpu = false;
    for (int u : P.ms)
        if (!pinned(in.uni[u])) t_cpu += c.cpu_us;
    for (int u : P.ms) {
        if (!pinned(in.uni[u])) {
            P.cu.push_back(u);
            continue;
        }
        const bool gpu_side = c.plan ? c.plan[0] == 'g' : to_gpu(t_gpu, t_cpu, c.gd_us, c.cpu_us);
        if (gpu_side)
            P.gd.push_back(u), t_gpu += c.gd_us;
        else
            P.cu.push_back(u), t_cpu += c.cpu_us, pinned_on_cpu = true;
    }
    // No pinned expert on the CPU, but some on the GPU: nothing measures the CPU, so the estimate decays. A layer with
    // pinned and unpinned experts on the CPU gives no sample (reads of the unpinned ones would count as slowness) and
    // no decay: the estimate stays. Without load it is the base, so the plan is the old one; after a load burst it
    // falls as soon as a layer sends only pinned experts to the GPU, with an unpinned one on the CPU as well.
    P.cost_skipped = !pinned_on_cpu && !P.gd.empty();
}

// How many misses are read into the cache after the CPU computed them: fill_n while no slot is free, else up to
// fill_cap and the free slots
struct FillLimits {
    int fill_n, fill_cap, free_slots;
};

// For the cache: the misses used by most tokens, then those with the largest weight (sorts P.ms); the experts to read
// go to `choice` (cleared first, so that its storage is reused from one layer to the next)
inline void choose_fill(CpuPlan& P, const SplitIn& in, const FillLimits& lim, std::vector<int>& choice) {
    auto score = [&](int u) {
        float w = 0;
        for (int b = 0; b < in.nb; b++) w += P.ew[(ptrdiff_t)b * in.umax + (ptrdiff_t)u];
        return std::pair<int, float>{std::popcount(P.um[u]), w};
    };
    std::sort(P.ms.begin(), P.ms.end(), [&](int a, int b) { return score(a) > score(b); });
    const int nf =
        std::min((int)P.ms.size(), lim.free_slots == 0 ? lim.fill_n : std::min(lim.fill_cap, lim.free_slots));
    choice.clear();
    for (int k = 0; k < nf; k++) choice.push_back(in.uni[P.ms[k]]);
}

}  // namespace tona::backend
