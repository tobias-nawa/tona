// ClassLRU order of the expert cache (backend/class_lru.h through gpu/mega_cache.h) and the unchanged LRU order:
// classes by CRF, victim by age / 2^class, preloaded experts leave first, kept experts stay, and the same hit count as
// the simulation tools/cache_sim/classlru.py on a scripted trace.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include "gpu/mega_cache.h"
#include "tests/test_support.h"

using namespace tona::gpu;
namespace backend = tona::backend;

namespace {

// The policy of a cache made by classlru_cache; a plain function so that the checks below need no optional access
backend::ClassLru& order_of(LayerCache& C) {
    if (!C.order) std::abort();  // a test bug: the cache was not made by classlru_cache
    return *C.order;
}

LayerCache classlru_cache(int slots, int n_expert = 64) {
    LayerCache C;
    C.order.emplace(n_expert);
    for (int s = slots - 1; s >= 0; s--) C.free_slots.push_back(s);
    return C;
}

LayerCache lru_cache(int slots) {
    LayerCache C;
    for (int s = slots - 1; s >= 0; s--) C.free_slots.push_back(s);
    return C;
}

// A pass over one layer of ClassLRU as the server does it: credit, move the hits, put the misses in
void demand(LayerCache& C, const std::vector<int>& uni, int tokens, int max_new = 1 << 20) {
    const std::unordered_set<int> keep(uni.begin(), uni.end());
    C.credit(uni);
    std::vector<int> miss;
    for (int e : uni) {
        if (auto it = C.where.find(e); it != C.where.end())
            C.touch(it);
        else
            miss.push_back(e);
    }
    std::sort(miss.begin(), miss.end());
    for (int e : miss) {
        if (max_new-- <= 0) break;
        const int slot = C.take_slot(&keep);
        if (slot >= 0) C.add_front(e, slot);
    }
    C.tick(tokens);
}

void test_lru_order_is_unchanged() {
    LayerCache C = lru_cache(3);
    for (int e = 0; e < 3; e++) C.add_front(e, C.take_slot());  // order front to back: 2 1 0
    C.touch(C.where.find(0));                                   // 0 2 1
    const std::unordered_set<int> keep{1};
    const int slot = C.take_slot(&keep);  // 1 is next but kept: 2 goes
    EXPECT(slot == 2);                    // the slot of expert 2
    EXPECT(C.where.contains(1) == 1 && !C.where.contains(2) && C.where.contains(0) == 1);
    C.add_back(7, slot);  // preloaded: first to leave
    const int next = C.take_slot();
    EXPECT(next == slot && !C.where.contains(7));
}

void test_class_follows_crf() {
    backend::ClassLru o(8);
    for (int i = 0; i < 5; i++) o.use(1), o.tick(1);
    o.use(2);
    EXPECT(o.crf(1) > 2.0 && o.crf(2) == 1.0);
    LayerCache C = classlru_cache(4, 8);
    demand(C, {3}, 1);
    EXPECT(order_of(C).class_of(3) == 0);
    for (int i = 0; i < 4; i++) demand(C, {3}, 1);  // CRF about 5 -> class 2
    EXPECT(order_of(C).class_of(3) == 2);
}

// Age 3 in class 3 is worth 3/8, age 1 in class 1 is worth 1/2: the more recent one goes (test_cache_sim.py)
void test_victim_is_largest_age_over_two_to_the_class() {
    LayerCache C = classlru_cache(2, 8);
    for (int i = 0; i < 8; i++) demand(C, {1}, 1);  // CRF about 7.5 -> class 2 (floor(log2))
    demand(C, {2}, 1);
    demand(C, {2}, 1);
    EXPECT(order_of(C).class_of(1) >= 2 && order_of(C).class_of(2) <= 1);
    const int v = order_of(C).victim(nullptr);
    EXPECT(v == 2);
}

void test_fresh_leave_first_and_keep_protects() {
    LayerCache C = classlru_cache(3, 8);
    demand(C, {1, 2}, 1);
    C.add_back(5, C.take_slot());  // preloaded, not used
    EXPECT(order_of(C).victim(nullptr) == 5);
    const std::unordered_set<int> keep{5};
    EXPECT(order_of(C).victim(&keep) != 5 && order_of(C).victim(&keep) >= 0);
    const std::unordered_set<int> all{1, 2, 5};
    EXPECT(order_of(C).victim(&all) == -1);
    EXPECT(C.take_slot(&all) == -1);  // full and everything kept
    C.touch(C.where.find(5));         // used: no longer first
    EXPECT(order_of(C).victim(nullptr) != 5);
}

void test_erase_and_clear() {
    LayerCache C = classlru_cache(2, 8);
    demand(C, {1, 2}, 1);
    const int s = C.where.at(1).first;
    C.erase(C.where.find(1));
    C.free_slots.push_back(s);
    EXPECT(!C.where.contains(1) && order_of(C).victim(nullptr) == 2);
    C.clear_order();
    EXPECT(C.where.empty() && order_of(C).victim(nullptr) == -1);
}

// 941 hits of 2400: ClassLruCache 4 / 128, cap 12, 40 experts, 400 passes, this generator, run in tools/cache_sim
struct Lcg {
    uint32_t x = 12345;
    int draw(int n) {
        x = (x * 1103515245u + 12345u) & 0x7fffffffu;
        return (int)((x >> 16) % (uint32_t)n);
    }
    std::vector<int> distinct(int k, int n) {
        std::vector<int> v;
        while ((int)v.size() < k) {
            const int a = draw(n), b = draw(n);  // two statements: the order of evaluation of arguments is not fixed
            const int e = std::min(a, b);
            if (std::find(v.begin(), v.end(), e) == v.end()) v.push_back(e);
        }
        return v;
    }
};

void prefetch(LayerCache& C, const std::vector<int>& cand, int n_pre) {
    std::unordered_set<int> protect;
    for (int e : cand)
        if (C.where.contains(e)) protect.insert(e);
    int done = 0;
    for (int e : cand) {
        if (C.where.contains(e)) continue;
        if (done++ >= n_pre) break;
        const int slot = C.take_slot(&protect);
        if (slot < 0) break;
        C.add_back(e, slot);
        protect.insert(e);
    }
}

void test_hit_count_equals_the_simulation() {
    LayerCache C = classlru_cache(12, 40);
    Lcg rng;
    int hits = 0, total = 0;
    for (int t = 0; t < 400; t++) {
        const std::vector<int> uni = rng.distinct(6, 40);
        for (int e : uni) hits += C.where.count(e), total++;
        demand(C, uni, 2);
        prefetch(C, rng.distinct(4, 40), 3);
    }
    EXPECT(total == 2400);
    EXPECT(hits == 941);
}

}  // namespace

int main() {
    return tona::test::guard("class_lru", [&] {
        test_lru_order_is_unchanged();
        test_class_follows_crf();
        test_victim_is_largest_age_over_two_to_the_class();
        test_fresh_leave_first_and_keep_protects();
        test_erase_and_clear();
        test_hit_count_equals_the_simulation();
        return tona::test::finish("class_lru");
    });
}
