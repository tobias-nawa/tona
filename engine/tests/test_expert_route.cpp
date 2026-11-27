// backend/expert_route.h (softmax, top-k, OEA, union) against a literal copy of the code ExpertSelect had before the
// move: random router logits with ties, NaN and infinity, one to eight tokens, OEA on and off. Argument "variants" runs
// the other OEA settings (the config is read from the environment once, so a second process).
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "backend/expert_route.h"
#include "tests/test_support.h"

namespace {

using tona::backend::ExpertRoute;
using tona::backend::oea_config;
using tona::backend::RouteShape;

constexpr int kUmax = 128;

// ExpertSelect::rank_token, oea_keep, oea_extra_experts, oea_fill, add_token_to_union and build_union as they were
struct OldRoute {
    int E, ne, nb;
    int k0 = 0;
    bool oea = false;
    std::vector<float> p;
    std::vector<int> idx;
    std::vector<char> base;
    int keep_n[8] = {};
    bool row0[8] = {};
    int oea_swapped = 0, oea_short = 0;
    std::vector<int> uni, u_of;
    const std::unordered_set<int>* where = nullptr;
    float* ew = nullptr;
    uint32_t* umask = nullptr;

    OldRoute(int e, int n, int b) : E(e), ne(n), nb(b), p((size_t)b * e), idx((size_t)b * e), u_of(e, -1) {}

    void oea_keep(int b) {
        const auto& cfg = oea_config();
        const float* pp = p.data() + (size_t)b * E;
        int* ix = idx.data() + (size_t)b * E;
        int n = 0;
        for (int k = 0; k < ne; k++) {
            const int e = ix[k];
            const bool keep = k < k0 || (cfg.row0 && row0[b]) || (cfg.tau > 0 && pp[e] >= cfg.tau) ||
                              (cfg.hits && where->contains(e));
            if (keep) std::rotate(ix + n, ix + k, ix + k + 1), n++;
        }
        keep_n[b] = n;
        for (int k = 0; k < n; k++) base[ix[k]] = 1;
    }

    int rank_token(int b, const float* r) {
        float* pp = p.data() + (size_t)b * E;
        int* ix = idx.data() + (size_t)b * E;
        const int bad = tona::backend::route_softmax(r, E, pp);
        tona::backend::route_top(pp, E, ne, ix);
        if (oea) oea_keep(b);
        return bad;
    }

    void extra() {
        const int extra = oea_config().extra;
        if (!oea || extra <= 0) return;
        std::vector<double> mass(E, 0.0);
        for (int b = 0; b < nb; b++)
            for (int k = keep_n[b]; k < ne; k++)
                mass[idx[(size_t)b * E + k]] += p[(size_t)b * E + idx[(size_t)b * E + k]];
        std::vector<int> cand;
        for (int e = 0; e < E; e++)
            if (!base[e] && mass[e] > 0) cand.push_back(e);
        const int m = std::min<int>(extra, (int)cand.size());
        std::partial_sort(cand.begin(), cand.begin() + m, cand.end(),
                          [&](int a, int c) { return mass[a] > mass[c] || (mass[a] == mass[c] && a < c); });
        for (int i = 0; i < m; i++) base[cand[i]] = 1;
    }

    void oea_fill(int b) {
        const float* pp = p.data() + (size_t)b * E;
        int* ix = idx.data() + (size_t)b * E;
        const int n0 = keep_n[b];
        std::vector<int> fill;
        for (int e = 0; e < E; e++)
            if (base[e]) fill.push_back(e);
        for (int k = 0; k < n0; k++) base[ix[k]] = 2;
        std::erase_if(fill, [&](int e) { return base[e] == 2; });
        for (int k = 0; k < n0; k++) base[ix[k]] = 1;
        const int m = std::min<int>(ne - n0, (int)fill.size());
        std::partial_sort(fill.begin(), fill.begin() + m, fill.end(),
                          [&](int a, int c) { return pp[a] > pp[c] || (pp[a] == pp[c] && a < c); });
        int own[kUmax];
        std::copy(ix + n0, ix + ne, own);
        for (int i = 0; i < m; i++) {
            if (std::find(own, own + ne - n0, fill[i]) == own + ne - n0) oea_swapped++;
            ix[n0 + i] = fill[i];
        }
        oea_short += ne - n0 - m;
        for (int k = n0 + m; k < ne; k++) ix[k] = -1;
    }

    void add_token(int b) {
        const float* pp = p.data() + (size_t)b * E;
        const int* ix = idx.data() + (size_t)b * E;
        double wsum = 0;
        for (int k = 0; k < ne && ix[k] >= 0; k++) wsum += pp[ix[k]];
        for (int k = 0; k < ne && ix[k] >= 0; k++) {
            const int e = ix[k];
            if (u_of[e] < 0) u_of[e] = (int)uni.size(), uni.push_back(e);
            ew[b * kUmax + u_of[e]] = (float)(pp[e] / wsum);
            umask[u_of[e]] |= 1u << b;
        }
    }

    void build_union() {
        for (int b = 0; b < nb; b++) {
            if (oea) oea_fill(b);
            add_token(b);
        }
    }
};

std::vector<float> random_logits(std::mt19937& rng, int E) {
    std::vector<float> r(E);
    const bool coarse = rng() % 3 == 0;  // many ties
    for (float& v : r) v = coarse ? (float)(rng() % 5) : (float)(rng() % 20000) / 1000.0f - 10.0f;
    const int special = (int)(rng() % 12);
    if (special == 0) r[rng() % E] = std::nanf("");
    if (special == 1) r[rng() % E] = INFINITY;
    if (special == 2) r[rng() % E] = -INFINITY;
    return r;
}

void one_case(std::mt19937& rng) {
    const int E = 16 + (int)(rng() % 49), ne = 2 + (int)(rng() % 7), nb = 1 + (int)(rng() % 8);
    const bool oea = nb > 1 && rng() % 3 != 0;
    const int k0 = 1 + (int)(rng() % ne);
    std::unordered_set<int> resident;
    for (int e = 0; e < E; e++)
        if (rng() % 3 == 0) resident.insert(e);
    OldRoute old(E, ne, nb);
    std::vector<float> ew_old((size_t)nb * kUmax, 0.0f), ew_new((size_t)nb * kUmax, 0.0f);
    std::vector<uint32_t> um_old(kUmax, 0), um_new(kUmax, 0);
    old.oea = oea, old.k0 = k0, old.where = &resident, old.ew = ew_old.data(), old.umask = um_old.data();
    ExpertRoute now(RouteShape{E, ne, nb, kUmax});
    const bool first_rows = rng() % 2 == 0;
    if (oea) {
        old.base.assign(E, 0), old.row0[0] = true;
        now.begin_oea(k0), now.mark_first_row(0);
        if (first_rows && nb > 2) old.row0[2] = true, now.mark_first_row(2);
    }
    for (int b = 0; b < nb; b++) {
        const std::vector<float> r = random_logits(rng, E);
        const int bad_old = old.rank_token(b, r.data());
        const int bad_new = now.rank_token(b, r.data(), [&](int e) { return resident.contains(e); });
        EXPECT(bad_old == bad_new);
    }
    old.extra();
    now.oea_extra_experts();
    old.build_union();
    now.build_union(ew_new.data(), um_new.data());
    EXPECT(old.uni == now.uni());
    EXPECT(ew_old == ew_new && um_old == um_new);
    EXPECT(old.oea_swapped == now.oea_swapped() && old.oea_short == now.oea_short());
    for (int b = 0; b < nb; b++) {
        EXPECT(std::memcmp(old.idx.data() + (size_t)b * E, now.ranked(b), sizeof(int) * (size_t)ne) == 0);
        EXPECT(std::memcmp(old.p.data() + (size_t)b * E, now.probs(b), sizeof(float) * (size_t)E) == 0);
    }
}

}  // namespace

int main(int argc, char** argv) {
    return tona::test::guard("expert_route", [&] {
        // The OEA variants of the environment, before oea_config() reads it for the first time
        if (argc > 1 && std::string(argv[1]) == "variants") {
            setenv("TONA_OEA_TAU", "0.08", 1);
            setenv("TONA_OEA_ROW0", "1", 1);
            setenv("TONA_OEA_EXTRA", "3", 1);
            setenv("TONA_OEA_HITS", "0", 1);
        }
        return tona::test::run("test_expert_route", [] {
            std::mt19937 rng(7);
            for (int i = 0; i < 4000; i++) one_case(rng);
        });
    });
}
