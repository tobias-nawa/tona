// The rest of the shared expert policy against literal copies of the code the reference had before the move: the CPU
// split (classify, planner, fill choice, backend/expert_split.h), the slot assignment of a pass
// (backend/expert_order.h) and the profile file with the order of experts by count (backend/expert_profile.h).
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "backend/expert_order.h"
#include "backend/expert_profile.h"
#include "backend/expert_split.h"
#include "tests/test_support.h"

namespace {

using namespace tona;
constexpr int kUmax = 128;

// the CPU split: ExpertSelect::classify_misses, split_misses, choose_fill as they were

struct OldPlan {
    std::vector<float> ew;
    std::vector<uint32_t> um;
    std::vector<int> hit, gd, cu, ms;
};

std::vector<float> old_sums(const OldPlan& P, int nb, size_t n) {
    std::vector<float> sums(nb, 0.0f);
    for (int b = 0; b < nb; b++)
        for (size_t u = 0; u < n; u++) sums[b] += P.ew[(ptrdiff_t)b * kUmax + (ptrdiff_t)u];
    return sums;
}

int old_classify(OldPlan& P, const std::vector<int>& uni, int nb, float drop, const std::unordered_set<int>& where) {
    int dropped = 0;
    std::vector<float> wsum0;
    if (drop > 0) wsum0 = old_sums(P, nb, uni.size());
    for (size_t u = 0; u < uni.size(); u++) {
        if (where.contains(uni[u])) {
            P.hit.push_back((int)u);
            continue;
        }
        float mw = 0;
        for (int b = 0; b < nb && drop > 0; b++) mw = std::max(mw, P.ew[(ptrdiff_t)b * kUmax + (ptrdiff_t)u]);
        if (drop > 0 && mw < drop) {
            for (int b = 0; b < nb; b++) P.ew[(ptrdiff_t)b * kUmax + (ptrdiff_t)u] = 0;
            dropped++;
            continue;
        }
        P.ms.push_back((int)u);
    }
    if (drop > 0) {
        const std::vector<float> now = old_sums(P, nb, uni.size());
        for (int b = 0; b < nb; b++) {
            if (now[b] <= 0) continue;
            for (size_t u = 0; u < uni.size(); u++) P.ew[(ptrdiff_t)b * kUmax + (ptrdiff_t)u] *= wsum0[b] / now[b];
        }
    }
    return dropped;
}

bool old_split(OldPlan& P, const std::vector<int>& uni, const backend::SplitCost& c, const std::vector<int>* pin_idx) {
    double t_gpu = P.hit.size() * c.hit_us, t_cpu = 0;
    bool pinned_on_cpu = false;
    for (int u : P.ms)
        if (!pin_idx || (*pin_idx)[uni[u]] < 0) t_cpu += c.cpu_us;
    for (int u : P.ms) {
        if (!pin_idx || (*pin_idx)[uni[u]] < 0) {
            P.cu.push_back(u);
            continue;
        }
        const bool gpu_side = c.plan ? c.plan[0] == 'g' : backend::to_gpu(t_gpu, t_cpu, c.gd_us, c.cpu_us);
        if (gpu_side)
            P.gd.push_back(u), t_gpu += c.gd_us;
        else
            P.cu.push_back(u), t_cpu += c.cpu_us, pinned_on_cpu = true;
    }
    return !pinned_on_cpu && !P.gd.empty();
}

std::vector<int> old_fill(OldPlan& P, const std::vector<int>& uni, int nb, int fill_n, int fill_cap, int free_slots) {
    auto score = [&](int u) {
        float w = 0;
        for (int b = 0; b < nb; b++) w += P.ew[(ptrdiff_t)b * kUmax + (ptrdiff_t)u];
        return std::pair<int, float>{std::popcount(P.um[u]), w};
    };
    std::sort(P.ms.begin(), P.ms.end(), [&](int a, int b) { return score(a) > score(b); });
    const int nf = std::min((int)P.ms.size(), free_slots == 0 ? fill_n : std::min(fill_cap, free_slots));
    std::vector<int> out;
    out.reserve((size_t)nf);
    for (int k = 0; k < nf; k++) out.push_back(uni[P.ms[k]]);
    return out;
}

struct SplitCase {
    int nb;
    std::vector<int> uni, pin;
    std::unordered_set<int> where;
    OldPlan old;
    backend::CpuPlan now;
};

SplitCase make_split_case(std::mt19937& rng) {
    const int E = 40, n = 4 + (int)(rng() % 40);
    SplitCase c;
    c.nb = 1 + (int)(rng() % 8);
    for (int e = 0; e < E && (int)c.uni.size() < n; e++)
        if (rng() % 2) c.uni.push_back(e);
    c.pin.assign(E, -1);
    for (int e = 0; e < E; e++) {
        if (rng() % 3 == 0) c.where.insert(e);
        if (rng() % 2) c.pin[e] = e;
    }
    c.old.ew.assign((size_t)8 * kUmax, 0.0f), c.old.um.assign(kUmax, 0);
    for (size_t u = 0; u < c.uni.size(); u++)
        for (int b = 0; b < c.nb; b++)
            if (rng() % 2)
                c.old.ew[(ptrdiff_t)b * kUmax + (ptrdiff_t)u] = (float)(rng() % 100) / 400.0f, c.old.um[u] |= 1u << b;
    c.now.ew = c.old.ew, c.now.um = c.old.um;
    return c;
}

void split_case(std::mt19937& rng) {
    SplitCase c = make_split_case(rng);
    const int nb = c.nb;
    const std::vector<int>& uni = c.uni;
    const std::vector<int>& pin = c.pin;
    const std::unordered_set<int>& where = c.where;
    OldPlan& old = c.old;
    backend::CpuPlan& now = c.now;
    const float drop = rng() % 3 == 0 ? 0.05f : 0.0f;
    const bool use_pin = rng() % 4 != 0;
    const char* plans[] = {nullptr, nullptr, "gpu", "cpu"};
    const backend::SplitCost cost{12.0, 15.0, 60.0 * (1 + rng() % 5), plans[rng() % 4]};
    const backend::SplitIn in{uni, nb, kUmax};

    const int dropped = old_classify(old, uni, nb, drop, where);
    backend::classify_misses(now, in, drop, [&](int e) { return where.contains(e); });
    const bool skipped = old_split(old, uni, cost, use_pin ? &pin : nullptr);
    backend::split_misses(now, in, cost, [&](int e) { return use_pin && pin[e] >= 0; });
    EXPECT(dropped == now.dropped && skipped == now.cost_skipped);
    EXPECT(old.hit == now.hit && old.gd == now.gd && old.cu == now.cu && old.ms == now.ms && old.ew == now.ew);

    const int fn = (int)(rng() % 6), fc = (int)(rng() % 9), fs = (int)(rng() % 4);
    const std::vector<int> want = old_fill(old, uni, nb, fn, fc, fs);
    std::vector<int> got = {99, 98};  // storage reused: cleared by the call
    backend::choose_fill(now, in, {fn, fc, fs}, got);
    EXPECT(want == got && old.ms == now.ms);
}

// The slot assignment: ExpertSelect::assign_slots as it was
struct OldAssign {
    std::vector<std::pair<int, int>> miss;
    int64_t hits = 0;
    bool out = false;
};

void old_assign(backend::ExpertOrder& C, const std::vector<int>& uni, int tokens, int32_t* slot, OldAssign& o) {
    std::unordered_set<int> keep;
    if (C.order) keep.insert(uni.begin(), uni.end());
    const std::unordered_set<int>* keep_ptr = C.order ? &keep : nullptr;
    C.credit(uni);
    for (size_t u = 0; u < uni.size(); u++) {
        const int e = uni[u];
        auto it = C.where.find(e);
        int s;
        if (it != C.where.end()) {
            o.hits++;
            s = it->second.first;
            C.touch(it);
        } else {
            s = C.take_slot(keep_ptr);
            if (s < 0) {
                o.out = true;
                return;
            }
            C.add_front(e, s);
            o.miss.emplace_back(e, s);
        }
        slot[u] = s;
    }
    C.tick(tokens);
}

backend::ExpertOrder make_cache(int slots, bool classlru) {
    backend::ExpertOrder C;
    if (classlru) C.order.emplace(64);
    for (int s = slots - 1; s >= 0; s--) C.free_slots.push_back(s);
    return C;
}

std::vector<int> order_of(const backend::ExpertOrder& C) {
    std::vector<int> v;
    for (const auto& [e, w] : C.where) v.push_back(e * 1000 + w.first);
    std::sort(v.begin(), v.end());
    return v;
}

void assign_run(bool classlru, uint32_t seed) {
    std::mt19937 rng(seed);
    const int slots = 6 + (int)(rng() % 8);
    backend::ExpertOrder a = make_cache(slots, classlru), b = make_cache(slots, classlru);
    for (int pass = 0; pass < 300; pass++) {
        const int nu = 1 + (int)(rng() % 9), tokens = 1 + (int)(rng() % 4);
        std::vector<int> uni;
        while ((int)uni.size() < nu) {
            const int e = (int)(rng() % 40);
            if (std::find(uni.begin(), uni.end(), e) == uni.end()) uni.push_back(e);
        }
        int32_t s_old[kUmax] = {}, s_new[kUmax] = {};
        OldAssign o;
        backend::SlotPlan n;
        old_assign(a, uni, tokens, s_old, o);
        backend::assign_slots(b, uni, tokens, s_new, n);
        EXPECT(o.hits == n.hits && o.miss == n.miss && o.out == n.out_of_slots);
        EXPECT(std::equal(s_old, s_old + uni.size(), s_new));
        EXPECT(order_of(a) == order_of(b) && a.free_slots == b.free_slots);
        EXPECT(a.lru == b.lru);
        if (o.out) return;  // every slot holds an expert of the union: the caller throws, the test ends
    }
}

// the profile file and the order by count

void test_profile(const std::string& dir) {
    const std::string path = dir + "/profile.bin";
    const std::vector<uint32_t> counts = {5, 9, 9, 1, 0, 3, 3, 3};  // 2 layers of 4 experts
    const auto lay = [](uint32_t layers, uint32_t experts) { return backend::ExpertLayout::uniform(layers, experts); };
    EXPECT(backend::write_expert_profile(path.c_str(), lay(2, 4), counts));
    EXPECT(backend::read_expert_profile(path.c_str(), lay(2, 4), "t") == counts);
    // the bytes of the file: "IEXP", 2, 4, then the counts
    std::FILE* f = std::fopen(path.c_str(), "rb");
    uint32_t head[3] = {};
    EXPECT(f && std::fread(head, sizeof head, 1, f) == 1 && head[0] == 0x50584549u && head[1] == 2 && head[2] == 4);
    if (f) std::fclose(f);
    EXPECT(test::throws([&] { backend::read_expert_profile(path.c_str(), lay(2, 5), "t"); }));  // another model
    EXPECT(test::throws([&] { backend::read_expert_profile(path.c_str(), lay(3, 4), "t"); }));
    EXPECT(backend::read_expert_profile((dir + "/none.bin").c_str(), lay(2, 4), "t").empty());  // missing: no profile
    EXPECT(!backend::write_expert_profile((dir + "/no/such/dir/p.bin").c_str(), lay(2, 4), counts));
    // truncated file
    std::filesystem::resize_file(path, 20);
    EXPECT(test::throws([&] { backend::read_expert_profile(path.c_str(), lay(2, 4), "t"); }));
    // order: count descending, ties by index
    EXPECT(backend::experts_by_count(counts.data(), 4) == (std::vector<int>{1, 2, 0, 3}));
    EXPECT(backend::experts_by_count(counts.data() + 4, 4) == (std::vector<int>{1, 2, 3, 0}));
    EXPECT(backend::experts_by_count(nullptr, 4) == (std::vector<int>{0, 1, 2, 3}));
}

}  // namespace

int main() {
    return tona::test::run("test_expert_host", [] {
        std::mt19937 rng(3);
        for (int i = 0; i < 5000; i++) split_case(rng);
        for (uint32_t s = 1; s <= 6; s++) assign_run(s % 2 == 0, s);
        test::TempDir dir("expert-host");
        test_profile(dir.path().string());
    });
}
