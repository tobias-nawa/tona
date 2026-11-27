#include "gpu/mega_select.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "backend/env.h"
#include "backend/omp_guard.h"
#include "backend/route_rank.h"
#include "gpu/expert_parts.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

namespace {

// TONA_ROUTE_LOG=file: per token and layer "position layer experts..." (overlap of several sequences); discarded drafts
// are in it too, the last line holds for a position
FILE* route_log() {
    static FILE* f = env::text("TONA_ROUTE_LOG") ? std::fopen(env::text("TONA_ROUTE_LOG"), "a") : nullptr;
    return f;
}

// TONA_ROUTE_TRACE=FILE: per layer and pass layer, count, experts of the union (one byte each, two with more than 256
// experts), for cache analyses
FILE* union_trace() {
    static FILE* f = env::text("TONA_ROUTE_TRACE") ? std::fopen(env::text("TONA_ROUTE_TRACE"), "wb") : nullptr;
    return f;
}

// TONA_ROUTE_TRACE_P=FILE: per layer and pass layer, nb, then per token the router logits of this layer and the
// prediction for l + 1 (stride floats each: 256, or 512 for the wide size class); for simulations of soft eviction
FILE* logit_trace() {
    static FILE* f = env::text("TONA_ROUTE_TRACE_P") ? std::fopen(env::text("TONA_ROUTE_TRACE_P"), "wb") : nullptr;
    return f;
}

}  // namespace

ExpertSelect::ExpertSelect(Mega& g, uint32_t l, int nb)
    : g_(g),
      l_(l),
      nb_(nb),
      E_((int)g.experts_.experts(l)),
      ne_(g.cache_[l].n_used),
      mb_(*g.mbox_host_),
      C_(g.cache_[l]),
      route_({E_, ne_, nb, MEGA_UMAX, C_.route}),
      uni_(route_.uni()) {}

void Mega::select(uint32_t l, int nb) {
    if (l == 0) pf_pass_ = std::min(pf_pass_ + 1, 1 << 30);
    if (l == 0 && pf_pass_ % 500 == 0) report_tokhist(stats_);
    const int64_t h0 = stats_.expert_hits, m0 = stats_.expert_misses;
    ExpertSelect(*this, l, nb).run();
    if (pf_pass_ <= 10) stats_.early_hits += stats_.expert_hits - h0, stats_.early_misses += stats_.expert_misses - m0;
}

void ExpertSelect::run() {
    C_.installed.clear();
    std::memset(mb_.umask, 0, sizeof mb_.umask);
    std::memset(mb_.ew, 0, sizeof mb_.ew);
    // OEA is off with TONA_DETERMINISTIC (the output would depend on the other sequence)
    const bool oea =
        g_.oea_k0_ > 0 && g_.oea_k0_ < ne_ && nb_ > 1 && (g_.multi_pass_ || g_.oea_all_) && !deterministic();
    if (oea) mark_first_rows();
    for (int b = 0; b < nb_; b++) rank_token(b);
    route_.oea_extra_experts();
    route_.build_union(mb_.ew, mb_.umask);
    note_selection();
    count_union();
    record_traces();
    predict_stats();
    if (g_.cpu_mode_) return plan_cpu();
    assign_slots();
}

// First row of every sequence of the pass (for TONA_OEA_ROW0)
void ExpertSelect::mark_first_rows() {
    route_.begin_oea(g_.oea_k0_);
    route_.mark_first_row(0);
    if (!g_.multi_pass_) return;
    for (int sl = 0; sl < g_.n_seqs_; sl++)
        if (g_.params_host_->snb[sl] > 0 && g_.params_host_->sb0[sl] < nb_)
            route_.mark_first_row(g_.params_host_->sb0[sl]);
}

void ExpertSelect::count_union() {
    g_.stats_.oea_swapped += route_.oea_swapped();
    g_.stats_.oea_short += route_.oea_short();
    if (nb_ <= 1) return;
    static const bool tokhist = env::text("TONA_TOKHIST") != nullptr;
    if (tokhist && !g_.multi_pass_ && nb_ <= 8)
        for (size_t u = 0; u < uni_.size(); u++) g_.stats_.tokhist[nb_][std::popcount(mb_.umask[u])]++;
    (g_.multi_pass_ ? g_.stats_.uni_multi : g_.stats_.uni_single) += (double)uni_.size();
    (g_.multi_pass_ ? g_.stats_.uni_multi_n : g_.stats_.uni_single_n)++;
}

// Router logits of token b; TONA_FAKE_ROUTE=1 (measuring mode): values from a hash of position, layer and expert,
// independent of the data (for comparisons in which the data are wrong, e.g. TONA_MEGA_NOBAR)
const float* ExpertSelect::router_logits(int b, float* scratch) const {
    static const bool fake_route = env::text("TONA_FAKE_ROUTE") != nullptr;
    if (!fake_route) return g_.mrows_.router_row(b);
    for (int e = 0; e < E_; e++) {
        uint64_t h = (uint64_t)(g_.pos_ + b) * 0x9E3779B97F4A7C15ull ^
                     (uint64_t)(l_ * g_.mrows_.stride + e) * 0xC2B2AE3D27D4EB4Full;
        h ^= h >> 29, h *= 0xBF58476D1CE4E5B9ull, h ^= h >> 32;
        scratch[e] = (float)(h >> 40) / (float)(1 << 24) * 8.0f;
    }
    return scratch;
}

// A NaN or infinite router logit means that something before the router is wrong (stale buffer, a race): the softmax
// takes NaN as -inf so that the ranking stays well defined, but the weights of that token are wrong. Counted, and
// reported once per request (the prefill does the same)
void ExpertSelect::report_nan_logits(int b, int bad) {
    g_.stats_.nan_rows++;
    if (g_.stats_.nan_reported) return;
    g_.stats_.nan_reported = true;
    std::fprintf(stderr,
                 "WARNING: %d unusable (NaN or infinite) router logits of token %d at layer %u, position %lld\n", bad,
                 b, l_, (long long)g_.pos_ + b);
}

// Softmax, top-k, renormalize: as cpu/ref.cpp
void ExpertSelect::rank_token(int b) {
    float scratch[MEGA_MAX_EXPERTS_WIDE];
    const float* r = router_logits(b, scratch);
    const int bad = route_.rank_token(b, r, [&](int e) { return C_.where.contains(e); });
    if (bad) report_nan_logits(b, bad);
}

void ExpertSelect::log_route(int b) {
    FILE* rlog = route_log();
    if (!rlog) return;
    const int* idx = route_.ranked(b);
    std::fprintf(rlog, "%lld %d", static_cast<long long>(g_.pos_) + b, l_);
    for (int k = 0; k < ne_ && idx[k] >= 0; k++) std::fprintf(rlog, " %d", idx[k]);
    std::fputc('\n', rlog);
}

// What the pass chose: the route log and the use count per expert, token by token as the union was built
void ExpertSelect::note_selection() {
    for (int b = 0; b < nb_; b++) {
        log_route(b);
        const int* idx = route_.ranked(b);
        for (int k = 0; k < ne_ && idx[k] >= 0; k++) g_.use_[g_.experts_.offset(l_) + idx[k]]++;
    }
}

void ExpertSelect::record_traces() {
    if (FILE* trace = union_trace()) {
        // One byte per expert id up to 256 experts per layer, two (little endian) above
        const size_t w = g_.mrows_.stride > MEGA_MAX_EXPERTS ? 2 : 1;
        uint8_t rec[2 + 2 * MEGA_UMAX] = {(uint8_t)l_, (uint8_t)uni_.size()};
        for (size_t u = 0; u < uni_.size(); u++)
            for (size_t k = 0; k < w; k++) rec[2 + w * u + k] = (uint8_t)(uni_[u] >> (8 * k));
        std::fwrite(rec, 1, 2 + w * uni_.size(), trace);
        if (l_ + 1 == g_.h_.n_layer) std::fflush(trace);
    }
    if (FILE* trace_p = logit_trace()) {
        const uint8_t hd[2] = {(uint8_t)l_, (uint8_t)nb_};
        std::fwrite(hd, 1, 2, trace_p);
        std::fwrite(g_.mrows_.router, sizeof(float), (size_t)nb_ * g_.mrows_.stride, trace_p);
        std::fwrite(g_.mrows_.pred, sizeof(float), (size_t)nb_ * g_.mrows_.stride, trace_p);
        if (l_ + 1 == g_.h_.n_layer) std::fflush(trace_p);
    }
}

// TONA_PREDICT_STATS: router of layer l + 1 on the x of layer l; hits against the real choice
void ExpertSelect::predict_stats() {
    static const bool on = env::text("TONA_PREDICT_STATS") != nullptr;
    if (!on || !g_.cpu_mode_) return;
    if (g_.pred_layer_ == l_ && (int)g_.pred_.size() == nb_) {
        for (int b = 0; b < nb_; b++) {
            const float* r = g_.mrows_.router_row(b);
            std::vector<int> act(E_);
            std::iota(act.begin(), act.end(), 0);
            std::partial_sort(act.begin(), act.begin() + ne_, act.end(), [&](int a, int c) { return r[a] > r[c]; });
            const int Ks[4] = {10, 16, 24, 32};
            for (int q = 0; q < 4; q++) {
                int h = 0;
                const auto& pr = g_.pred_[b];
                for (int k = 0; k < ne_; k++)
                    h += std::find(pr.begin(), pr.begin() + Ks[q], act[k]) != pr.begin() + Ks[q];
                g_.pred_hit_[q] += (double)h / ne_;
            }
            g_.pred_n_++;
        }
        for (int e : uni_) {
            if (C_.where.contains(e)) continue;
            bool in = false;
            for (int b = 0; b < nb_ && !in; b++)
                in = std::find(g_.pred_[b].begin(), g_.pred_[b].begin() + 24, e) != g_.pred_[b].begin() + 24;
            g_.pred_ms_hit_ += in, g_.pred_ms_n_++;
        }
    }
    if (l_ + 1 < g_.h_.n_layer) g_.predict_next(l_, nb_);
}

// Slots (backend::assign_slots): the union is at the front of the LRU list and is not evicted (slots_ >= |union|)
void ExpertSelect::assign_slots() {
    backend::SlotPlan plan;
    if (fake_experts()) {
        C_.credit(uni_);
        for (size_t u = 0; u < uni_.size(); u++) mb_.slot[u] = uni_[u] % g_.slots_;
        C_.tick(nb_);
    } else {
        backend::assign_slots(C_, uni_, nb_, mb_.slot, plan);
        g_.stats_.expert_hits += plan.hits;
        g_.stats_.expert_misses += (int64_t)plan.miss.size();
        if (plan.out_of_slots) throw std::runtime_error("gpu::Mega: no expert slots");
        for (const auto& m : plan.miss) C_.installed.push_back(m.first);
    }
    const std::vector<std::pair<int, int>>& miss = plan.miss;
    mb_.n_u = (int32_t)uni_.size();
    mb_.n_miss = (int32_t)miss.size();
    for (size_t i = 0; i < miss.size(); i++) {
        g_.elastic_.require_live(C_, (int)l_, miss[i].second);  // the moe.copy writes there
        mb_.miss_slot[i] = miss[i].second;
    }
    if (!miss.empty()) read_misses(miss);
}

void ExpertSelect::read_misses(const std::vector<std::pair<int, int>>& miss) {
    const MoeWeights& S = g_.m_.moe_weights(l_);
    const ExpertParts P = expert_parts(S, C_);
    const auto& parts = P.t;
    const auto& offs = P.off;
    const auto t0 = std::chrono::steady_clock::now();
    backend::OmpErrors err;  // a failing pread must reach the request, not end the process
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < (int)miss.size(); i++)
        for (int k = 0; k < P.n; k++)
            err.run([&] {
                const size_t n = parts[k]->bytes / parts[k]->dims[2];
                g_.m_.gguf().read(*parts[k], (size_t)miss[i].first * n, g_.staging_ + i * C_.slot_bytes + offs[k], n);
            });
    err.rethrow();
    g_.stats_.read_s += seconds_since<std::chrono::steady_clock>(t0);
    g_.stats_.upload_bytes += (double)miss.size() * C_.slot_bytes;
}

}  // namespace tona::gpu
