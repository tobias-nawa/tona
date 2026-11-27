// How many experts each layer has. The Coder has the same number in every layer; models with pruned or unevenly sized
// layers have a different one per layer. Every table that is indexed by [layer][expert] (the usage counts, the profile
// file) is stored flat, with offset(l) as the first entry of layer l, so that no code assumes `layers * experts`.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace tona::backend {

class ExpertLayout {
public:
    static ExpertLayout uniform(uint32_t n_layer, uint32_t experts) {
        return ExpertLayout(std::vector<uint32_t>(n_layer, experts));
    }

    explicit ExpertLayout(std::vector<uint32_t> per_layer) : n_(std::move(per_layer)), off_(n_.size() + 1, 0) {
        std::partial_sum(n_.begin(), n_.end(), off_.begin() + 1);
    }

    uint32_t layers() const { return (uint32_t)n_.size(); }
    uint32_t experts(uint32_t l) const { return n_[l]; }
    uint32_t max_experts() const { return n_.empty() ? 0 : *std::max_element(n_.begin(), n_.end()); }
    // Entries of a [layer][expert] table
    size_t total() const { return off_.back(); }
    size_t offset(uint32_t l) const { return off_[l]; }
    bool is_uniform() const {
        return std::all_of(n_.begin(), n_.end(), [&](uint32_t e) { return e == n_[0]; });
    }
    bool operator==(const ExpertLayout& o) const { return n_ == o.n_; }

private:
    std::vector<uint32_t> n_;
    std::vector<size_t> off_;  // layers + 1 entries
};

}  // namespace tona::backend
