// --batch-bench N: time per pass of the verify pass at widths 1 to 4, all tokens accepted.
// --batch-check N: the verify pass against single steps (deviation, then bitwise).
// TONA_BENCH_NB=k: bench only width k (up to MEGA_BMAX); TONA_BENCH_WIDTHS=1,2,4,8: these widths in turn, the list
//   TONA_BENCH_REPEAT=n times (rounds interleaved); TONA_BENCH_ACCEPT=m: only min(m, width) tokens of each pass are
//   accepted (the rest is recomputed); TONA_BENCH_HASH=1: hash of all logits of the width (equal hashes of two
//   builds: bit-identical output at this width, wide path included). TONA_MEGA_PROF=1: per-step profile of every
//   width after its bench.
// TONA_BATCH_NB=w: widest pass of the check (default 4); TONA_BATCH_REF_NB=k: reference from passes of k tokens
//   instead of single steps.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <vector>

#include "backend/env.h"
#include "tona_gpu_gen.h"

namespace tona::gen {

namespace {

constexpr int kWarmupTokens = 64;

// FNV-1a over the 32-bit words of the logits of a pass
void hash_logits(uint64_t& h, const float* g, size_t n) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(g);
    for (size_t i = 0; i < n; i++) h = (h ^ w[i]) * 0x100000001b3ull;
}

void bench_width(tona::gpu::Mega& mg, size_t vocab, int n_tokens, const std::vector<int32_t>& ids, int nb) {
    const bool hash = env::flag("TONA_BENCH_HASH");
    uint64_t h = 0xcbf29ce484222325ull;
    mg.reset();
    std::vector<int32_t> seq;  // the prompt, repeated
    while ((int)seq.size() < n_tokens + tona::gpu::MEGA_BMAX) seq.insert(seq.end(), ids.begin(), ids.end());
    const int acc = (int)env::integer("TONA_BENCH_ACCEPT", 0);
    const int take = acc > 0 ? std::min(acc, nb) : nb;
    int pos = 0, passes = 0;
    auto pass = [&] {
        const float* lg = mg.forward(seq.data() + pos, nb);
        if (hash) hash_logits(h, lg, (size_t)nb * vocab);
        mg.commit(take);
    };
    for (; pos < kWarmupTokens; pos += take) pass();
    mg.profile(true);  // the average of the timed passes only
    const auto t0 = Clock::now();
    const auto s0 = mg.stats();
    for (; pos < n_tokens; pos += take, passes++) pass();
    const double dt = seconds_since(t0);
    const auto s1 = mg.stats();
    const double pp = std::max(1, passes);
    std::printf("CPU selection %.3f ms per pass\n", 1e3 * (s1.select_s - s0.select_s) / pp);
    std::printf("per pass: %.1f hits, %.1f GPU direct, %.1f CPU (%.1f ms), %.1f refilled\n",
                (s1.expert_hits - s0.expert_hits) / pp, (s1.gpu_direct - s0.gpu_direct) / pp,
                (s1.cpu_experts - s0.cpu_experts) / pp, 1e3 * (s1.cpu_s - s0.cpu_s) / pp, (s1.fills - s0.fills) / pp);
    std::printf("B = %d: %d passes, %.2f ms per pass, %.1f passes/s, %.1f tokens/s\n", nb, passes, 1e3 * dt / passes,
                passes / dt, passes * take / dt);
    if (hash) std::printf("logits hash B = %d: %016llx\n", nb, (unsigned long long)h);
    if (env::flag("TONA_MEGA_PROF")) print_profile(stdout, std::format("Profile B = {}", nb).c_str(), mg);
}

// Widths to bench: TONA_BENCH_WIDTHS (comma list), else TONA_BENCH_NB, else 1 to 4
std::vector<int> bench_widths() {
    std::vector<int> w;
    if (const char* list = env::text("TONA_BENCH_WIDTHS")) {
        for (const char* p = list; *p;) {
            char* end = nullptr;
            const long v = std::strtol(p, &end, 10);
            if (end == p) break;
            if (v >= 1 && v <= tona::gpu::MEGA_BMAX) w.push_back((int)v);
            p = *end == ',' ? end + 1 : end;
        }
        return w;
    }
    const int only = (int)env::integer("TONA_BENCH_NB", 0);
    for (int nb = 1; nb <= (only ? tona::gpu::MEGA_BMAX : 4); nb++)
        if (!only || nb == only) w.push_back(nb);
    return w;
}

// Logits of the first n positions, from single steps (rnb <= 1) or from passes of rnb tokens.
std::vector<std::vector<float>> reference_logits(Session& s, const std::vector<int32_t>& ids, int n, int rnb) {
    tona::gpu::Mega& mg = *s.mg;
    std::vector<std::vector<float>> ref(n);
    if (rnb <= 1) {
        for (int i = 0; i < n; i++) ref[i] = mg.step(ids[i]);
        return ref;
    }
    const size_t vocab = s.model.hp().vocab;
    for (int i = 0; i + rnb <= n; i += rnb) {
        const float* lg = mg.forward(ids.data() + i, rnb);
        for (int b = 0; b < rnb; b++) ref[i + b].assign(lg + b * vocab, lg + (b + 1) * vocab);
        mg.commit(rnb);
    }
    for (int i = n / rnb * rnb; i < n; i++) ref[i] = mg.step(ids[i]);
    return ref;
}

struct Comparison {
    int passes = 0;
    int cmp = 0;
    int same_top1 = 0;
    double worst = 0;
};

// Widths in turn from `widest` down to 1; of each pass 1, 2, 3, ... tokens are accepted, the rest is recomputed.
Comparison compare_mixed_widths(tona::gpu::Mega& mg, const std::vector<int32_t>& ids, int n,
                                const std::vector<std::vector<float>>& ref, int widest) {
    Comparison c;
    const size_t vocab = ref[0].size();
    for (int pos = 0, k = 0; pos < n; k++) {
        const int nb = std::min(widest - k % widest, n - pos);
        const float* lg = mg.forward(ids.data() + pos, nb);
        for (int b = 0; b < nb; b++) {
            const float* g = lg + b * vocab;
            const auto& r = ref[pos + b];
            c.worst = std::max(c.worst, relative_deviation(g, r.data(), vocab));
            c.same_top1 += argmax(g, vocab) == argmax(r);
            c.cmp++;
        }
        const int accepted = std::min(nb, 1 + (k / widest + k) % widest);
        mg.commit(accepted);
        pos += accepted;
        c.passes++;
    }
    return c;
}

// Passes of exactly nb tokens, all accepted, against the reference bit for bit.
void compare_bitwise(tona::gpu::Mega& mg, const std::vector<int32_t>& ids, int n,
                     const std::vector<std::vector<float>>& ref, int nb) {
    const size_t vocab = ref[0].size();
    int first = -1, bit_same = 0;
    for (int pos = 0; pos + nb <= n; pos += nb) {
        const float* lg = mg.forward(ids.data() + pos, nb);
        for (int b = 0; b < nb; b++) {
            const bool eq = !std::memcmp(lg + b * vocab, ref[pos + b].data(), vocab * sizeof(float));
            bit_same += eq;
            if (!eq && first < 0) first = pos + b;
        }
        mg.commit(nb);
    }
    std::printf(
        "bitwise (passes of %d): %d of %d positions equal, first deviation at position %d (token %d in the pass)\n", nb,
        bit_same, n / nb * nb, first, first < 0 ? -1 : first % nb);
}

}  // namespace

int run_batch_bench(Session& s, const Options& o, const std::vector<int32_t>& ids) {
    const std::vector<int> widths = bench_widths();
    for (int round = 0; round < (int)env::integer("TONA_BENCH_REPEAT", 1); round++)
        for (const int nb : widths) bench_width(*s.mg, s.model.hp().vocab, o.batch_bench, ids, nb);
    return 0;
}

int run_batch_check(Session& s, const Options& o, const std::vector<int32_t>& ids) {
    tona::gpu::Mega& mg = *s.mg;
    const int n = std::min<int>((int)ids.size(), o.batch_check);
    mg.reset();
    const auto ref = reference_logits(s, ids, n, (int)env::integer("TONA_BATCH_REF_NB", 1));
    mg.reset();
    const int batch_nb = (int)env::integer("TONA_BATCH_NB", 4);
    const Comparison c = compare_mixed_widths(mg, ids, n, ref, std::max(4, batch_nb));
    std::printf(
        "Verify pass: %d positions, %d passes, %d comparisons, top-1 equal in %d, worst relative deviation %.2e\n", n,
        c.passes, c.cmp, c.same_top1, c.worst);
    mg.reset();
    compare_bitwise(mg, ids, n, ref, batch_nb);
    return 0;
}

}  // namespace tona::gen
