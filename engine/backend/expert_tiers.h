// Which experts to pin in RAM when the budget holds only part of them (docs/architecture/expert-tiers.md). Pure
// function of the usage profile, so every backend that pins can use it and a unit test can check it.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "backend/expert_layout.h"
#include "backend/expert_profile.h"

namespace tona::backend {

// Below this share of all expert bytes a partial pin by the profile does not beat the page cache: with room for 20 to
// 40 % of the experts the SSD reads per token were 0 to 18 % higher than with the page cache alone, with 75 to 80 %
// pinned they fell from 4.2 to 1.7 to 1.9 (research/measurements/2026-11-15-expert-tiers.md, step 1).
constexpr double MIN_PARTIAL_PIN_FRACTION = 0.7;

// Weight of the experts that the VRAM cache is seeded with: they are in VRAM most of the time, so a pinned copy is read
// only after an eviction. The design value; the result at 75 to 80 % pinned is the same for 0.2 and 1.0.
constexpr double VRAM_SEED_WEIGHT = 0.2;

// Per layer the experts to pin for `budget` bytes, in descending score order. The score of an expert is its use count,
// times VRAM_SEED_WEIGHT if it is among the `vram_slots` most used of its layer; one ranking over all layers, so a
// layer with a flat distribution gets more places than a skewed one. `bytes[l]` is the size of one expert of layer l
// (all three tensors). Ties go to the lower layer, then the lower expert, so the result is the same on every run.
inline std::vector<std::vector<int>> select_pinned(const ExpertLayout& layout, const std::vector<uint32_t>& counts,
                                                   const std::vector<size_t>& bytes, int vram_slots, int64_t budget) {
    struct Entry {
        double score;
        uint32_t layer;
        int expert;
    };
    std::vector<Entry> all;
    all.reserve(layout.total());
    for (uint32_t l = 0; l < layout.layers(); l++) {
        const uint32_t* cnt = counts.data() + layout.offset(l);
        const std::vector<int> ord = experts_by_count(cnt, (int)layout.experts(l));
        for (size_t r = 0; r < ord.size(); r++)
            all.push_back({cnt[ord[r]] * ((int)r < vram_slots ? VRAM_SEED_WEIGHT : 1.0), l, ord[r]});
    }
    std::stable_sort(all.begin(), all.end(), [](const Entry& a, const Entry& b) { return a.score > b.score; });
    std::vector<std::vector<int>> chosen(layout.layers());
    int64_t left = budget;
    for (const Entry& e : all) {
        if ((int64_t)bytes[e.layer] > left) break;
        left -= (int64_t)bytes[e.layer];
        chosen[e.layer].push_back(e.expert);
    }
    return chosen;
}

// Share of the experts that fits, and whether it is worth pinning that part (always if all fit)
inline bool partial_pin_worthwhile(int64_t budget, size_t all_bytes, double min_fraction = MIN_PARTIAL_PIN_FRACTION) {
    return budget > 0 && (budget >= (int64_t)all_bytes || (double)budget >= min_fraction * (double)all_bytes);
}

}  // namespace tona::backend
