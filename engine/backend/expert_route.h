// Which experts a pass uses in one layer, the part every backend shares: softmax and top-k of the router logits per
// token, OEA (fewer different experts in passes of several sequences), and the union over the tokens with the weight of
// each token for each expert. Pure host code on arrays: the router logits come in, the union and the weights go out;
// reading them from the device, the cache lookups, statistics, logs and traces are the backend's.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "backend/backend.h"
#include "backend/env.h"
#include "backend/route_policy.h"
#include "backend/route_rank.h"

namespace tona::backend {

// OEA variants (research and audit 2026-10-26): TONA_OEA_TAU keeps own experts from this router probability,
// TONA_OEA_HITS own ones that are already in fast memory (only misses are swapped; default on, best measured ratio,
// research/measurements/2026-10-26-oea; TONA_OEA_HITS=0 switches it off), TONA_OEA_ROW0=1 the first row of each
// sequence entirely (it is certainly accepted), TONA_OEA_EXTRA=n takes n more experts by the sum of their probability
// over the rows that want them (XShare, arXiv 2602.07265). Kept ones belong to the base set.
struct OeaConfig {
    float tau;
    bool hits, row0;
    int extra;
};

inline const OeaConfig& oea_config() {
    static const OeaConfig c = {(float)env::number("TONA_OEA_TAU", 0.0), env::integer("TONA_OEA_HITS", 1) != 0,
                                env::flag("TONA_OEA_ROW0"), (int)env::integer("TONA_OEA_EXTRA", 0)};
    return c;
}

// Sizes of one layer and pass: experts, experts per token, tokens, places of the union (row stride of the weights), and
// the routing policy of the layer (route_policy.h; default: softmax, renormalized, the Coder's)
struct RouteShape {
    int n_expert, n_used, nb, umax;
    RoutePolicy policy = {};
};

class ExpertRoute {
public:
    explicit ExpertRoute(const RouteShape& s)
        : E_(s.n_expert),
          ne_(s.n_used),
          nb_(s.nb),
          umax_(s.umax),
          policy_(s.policy),
          p_((size_t)s.nb * s.n_expert),
          key_(s.policy.gate == Gate::Sigmoid ? (size_t)s.nb * s.n_expert : 0),
          idx_((size_t)s.nb * s.n_expert),
          u_of_(s.n_expert, -1) {}

    // OEA (arXiv 2511.02237): in passes of several sequences each token keeps its best k0 experts; it fills the other
    // places, in its ranking, only from the union of these base sets, weights renormalized over the chosen ones. Fewer
    // different experts per layer, no new ones. Before the first rank_token; mark_first_row for each sequence's first
    // row (row 0 always) when TONA_OEA_ROW0 is on.
    // Only for the softmax policy: OEA ranks and fills by probability, while the sigmoid policy chooses by score +
    // bias, so a swap by score would change the choice rule; with another policy this does nothing.
    void begin_oea(int k0) {
        if (policy_.gate != Gate::Softmax) return;
        oea_ = true, k0_ = k0;
        base_.assign(E_, 0);
    }
    void mark_first_row(int b) { row0_[b] = true; }

    // The policy's choice: softmax, top-k, renormalize (as cpu/ref.cpp), or sigmoid, top-k of score + bias, scaled
    // weights. r: the router logits of token b. resident(e): is expert e in fast memory (for OEA_HITS). Returns the
    // number of unusable logits (NaN or infinite), for the backend to report.
    template <class Resident>
    int rank_token(int b, const float* r, Resident&& resident) {
        float* p = p_.data() + (size_t)b * E_;
        int* idx = idx_.data() + (size_t)b * E_;
        if (policy_.gate == Gate::Sigmoid) {
            float* key = key_.data() + (size_t)b * E_;
            const int bad = route_sigmoid(r, policy_.bias, E_, p, key);
            route_top(key, E_, ne_, idx);
            return bad;
        }
        const int bad = route_softmax(r, E_, p);
        route_top(p, E_, ne_, idx);
        if (oea_) oea_keep(b, resident);
        return bad;
    }

    // More base experts by the summed probability over the rows that want them among their top-k
    void oea_extra_experts() {
        const int extra = oea_config().extra;
        if (!oea_ || extra <= 0) return;
        std::vector<double> mass(E_, 0.0);
        for (int b = 0; b < nb_; b++)
            for (int k = keep_n_[b]; k < ne_; k++)
                mass[idx_[(size_t)b * E_ + k]] += p_[(size_t)b * E_ + idx_[(size_t)b * E_ + k]];
        std::vector<int> cand;
        for (int e = 0; e < E_; e++)
            if (!base_[e] && mass[e] > 0) cand.push_back(e);
        const int m = std::min<int>(extra, (int)cand.size());
        std::partial_sort(cand.begin(), cand.begin() + m, cand.end(),
                          [&](int a, int c) { return mass[a] > mass[c] || (mass[a] == mass[c] && a < c); });
        for (int i = 0; i < m; i++) base_[cand[i]] = 1;
    }

    // The union of the experts of all tokens (places u, in order of first use) with the renormalized weight of token b
    // for place u in ew[b * umax + u] and the mask of tokens that use it in umask[u]; both zeroed by the caller.
    void build_union(float* ew, uint32_t* umask) {
        for (int b = 0; b < nb_; b++) {
            if (oea_) oea_fill(b);
            add_token_to_union(b, ew, umask);
        }
    }

    const std::vector<int>& uni() const { return uni_; }
    const float* probs(int b) const { return p_.data() + (size_t)b * E_; }
    // Experts of token b by rank (-1 behind the last one OEA could not fill)
    const int* ranked(int b) const { return idx_.data() + (size_t)b * E_; }
    int oea_swapped() const { return oea_swapped_; }  // experts a token got that are not among its own top-k
    int oea_short() const { return oea_short_; }      // places OEA could not fill

private:
    // Keep the best k0 and more by variant; stable to the front, the ranking stays
    template <class Resident>
    void oea_keep(int b, Resident&& resident) {
        const OeaConfig& cfg = oea_config();
        const float* p = p_.data() + (size_t)b * E_;
        int* idx = idx_.data() + (size_t)b * E_;
        int n = 0;
        for (int k = 0; k < ne_; k++) {
            const int e = idx[k];
            const bool keep =
                k < k0_ || (cfg.row0 && row0_[b]) || (cfg.tau > 0 && p[e] >= cfg.tau) || (cfg.hits && resident(e));
            if (keep) std::rotate(idx + n, idx + k, idx + k + 1), n++;
        }
        keep_n_[b] = n;
        for (int k = 0; k < n; k++) base_[idx[k]] = 1;
    }

    // idx[0 ... keep) stays, then the best others of the base set (the ranking over all E restricted to the base set:
    // algorithm 1 of the paper with maxP = N)
    void oea_fill(int b) {
        const float* p = p_.data() + (size_t)b * E_;
        int* idx = idx_.data() + (size_t)b * E_;
        const int n0 = keep_n_[b];
        std::vector<int> fill;
        for (int e = 0; e < E_; e++)
            if (base_[e]) fill.push_back(e);
        for (int k = 0; k < n0; k++) base_[idx[k]] = 2;  // own ones not twice
        std::erase_if(fill, [&](int e) { return base_[e] == 2; });
        for (int k = 0; k < n0; k++) base_[idx[k]] = 1;
        const int m = std::min<int>(ne_ - n0, (int)fill.size());
        std::partial_sort(fill.begin(), fill.begin() + m, fill.end(),
                          [&](int a, int c) { return p[a] > p[c] || (p[a] == p[c] && a < c); });
        int own[kMaxUsed];
        std::copy(idx + n0, idx + ne_, own);
        for (int i = 0; i < m; i++) {
            if (std::find(own, own + ne_ - n0, fill[i]) == own + ne_ - n0) oea_swapped_++;  // not among the own top-k
            idx[n0 + i] = fill[i];
        }
        oea_short_ += ne_ - n0 - m;
        for (int k = n0 + m; k < ne_; k++) idx[k] = -1;
    }

    // The experts of token b into the union with their renormalized weights and the mask of tokens. Softmax: p / sum of
    // the chosen p in double (as before the policies). Sigmoid: the float sum in rank order and the weight of
    // route_policy.h, the bits the GPU computes.
    void add_token_to_union(int b, float* ew, uint32_t* umask) {
        const float* p = p_.data() + (size_t)b * E_;
        const int* idx = idx_.data() + (size_t)b * E_;
        double wsum = 0;
        float fsum = 0;
        for (int k = 0; k < ne_ && idx[k] >= 0; k++) wsum += p[idx[k]], fsum += p[idx[k]];
        const bool sigmoid = policy_.gate == Gate::Sigmoid;
        for (int k = 0; k < ne_ && idx[k] >= 0; k++) {
            const int e = idx[k];
            if (u_of_[e] < 0) u_of_[e] = (int)uni_.size(), uni_.push_back(e);
            ew[b * umax_ + u_of_[e]] =
                sigmoid ? sigmoid_route::weight(p[e], fsum, policy_.scale) : (float)(p[e] / wsum);
            umask[u_of_[e]] |= 1u << b;
        }
    }

    // Largest n_used of a model (the gpu kernels take 16 as well, mega_types.h MEGA_UMAX >= n_used * BMAX)
    static constexpr int kMaxUsed = 128;

    const int E_, ne_, nb_, umax_;
    const RoutePolicy policy_;
    std::vector<float> p_;    // [nb][E] router probabilities (sigmoid policy: scores)
    std::vector<float> key_;  // [nb][E] sigmoid policy: what the choice ranks by
    std::vector<int> idx_;    // [nb][E] experts by rank
    std::vector<char> base_;  // OEA: base set
    int keep_n_[BMAX] = {};
    bool row0_[BMAX] = {};
    bool oea_ = false;
    int k0_ = 0, oea_swapped_ = 0, oea_short_ = 0;
    std::vector<int> uni_, u_of_;
};

}  // namespace tona::backend
