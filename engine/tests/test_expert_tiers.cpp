// The pin selection of backend/expert_tiers.h: budget respected, global ranking, determinism, the VRAM seed weight.
#include <cstdint>
#include <vector>

#include "backend/expert_tiers.h"
#include "tests/test_support.h"

namespace {

using namespace tona::backend;

size_t pinned_count(const std::vector<std::vector<int>>& c) {
    size_t n = 0;
    for (const auto& v : c) n += v.size();
    return n;
}

// Two layers of 8 experts: layer 0 is skewed (expert 0 takes almost everything), layer 1 is flat
void test_budget_and_global_ranking() {
    const ExpertLayout layout = ExpertLayout::uniform(2, 8);
    std::vector<uint32_t> counts = {1000, 10, 9, 8, 7, 6, 5, 4, /* layer 1 */ 100, 90, 80, 70, 60, 50, 40, 30};
    const std::vector<size_t> bytes = {10, 10};
    const auto c = select_pinned(layout, counts, bytes, 0, 70);  // no VRAM seeds: pure count order, 7 places
    EXPECT(pinned_count(c) == 7);
    EXPECT(c[0].size() == 1 && c[0][0] == 0);  // 1000 first
    EXPECT(c[1].size() == 6);                  // the flat layer gets the other six: 100..50 beat 10 and below
    EXPECT(select_pinned(layout, counts, bytes, 0, 0).front().empty());
    EXPECT(pinned_count(select_pinned(layout, counts, bytes, 0, 1000)) == 16);  // more than all: all
    EXPECT(pinned_count(select_pinned(layout, counts, bytes, 0, 19)) == 1);     // a place is only taken if it fits
}

// The seeded expert of a layer (the one VRAM slot) weighs VRAM_SEED_WEIGHT: layer 0's 400 becomes 80, layer 1's 100
// becomes 20 and falls behind the unseeded 90, 80, 70
void test_seed_weight() {
    const ExpertLayout layout = ExpertLayout::uniform(2, 4);
    const std::vector<uint32_t> counts = {400, 5, 4, 3, /* layer 1 */ 100, 90, 80, 70};
    const std::vector<size_t> bytes = {1, 1};
    const auto c = select_pinned(layout, counts, bytes, 1, 4);
    EXPECT(pinned_count(c) == 4);
    EXPECT(c[0] == std::vector<int>({0}));
    EXPECT(c[1] == std::vector<int>({1, 2, 3}));  // expert 0 of layer 1 (seed, 20) is left out
}

// The same input gives the same result, ties by layer and expert
void test_deterministic_ties() {
    const ExpertLayout layout = ExpertLayout::uniform(3, 4);
    const std::vector<uint32_t> counts(12, 7);
    const std::vector<size_t> bytes = {1, 1, 1};
    const auto a = select_pinned(layout, counts, bytes, 0, 5), b = select_pinned(layout, counts, bytes, 0, 5);
    EXPECT(a == b);
    EXPECT(a[0].size() == 4 && a[1].size() == 1 && a[2].empty());  // lower layers first
}

void test_worthwhile() {
    EXPECT(partial_pin_worthwhile(100, 100));
    EXPECT(partial_pin_worthwhile(70, 100));
    EXPECT(!partial_pin_worthwhile(69, 100));
    EXPECT(!partial_pin_worthwhile(0, 100));
    EXPECT(partial_pin_worthwhile(20, 100, 0.1) && !partial_pin_worthwhile(5, 100, 0.1));
    EXPECT(!partial_pin_worthwhile(-5, 100, 0.0));
}

}  // namespace

int main() {
    return tona::test::run("test_expert_tiers", [] {
        test_budget_and_global_ranking();
        test_seed_weight();
        test_deterministic_ties();
        test_worthwhile();
    });
}
