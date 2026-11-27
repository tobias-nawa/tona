#include "server/suffix.h"

#include <algorithm>

namespace tona::server {

uint64_t SuffixIndex::key(const int32_t* t) {
    uint64_t k = 0;
    for (int i = 0; i < KEY; i++) k = k * 0x9E3779B97F4A7C15ull + (uint32_t)t[i];
    return k;
}

void SuffixIndex::add(size_t i) {
    if (i + 1 < KEY) return;
    auto& v = idx_[key(&h_[i + 1 - KEY])];
    v.push_back((uint32_t)i);
    if (v.size() > 2 * MAX_POS) v.erase(v.begin(), v.end() - MAX_POS);
}

void SuffixIndex::sync(const std::vector<int32_t>& toks) {
    if (toks.size() < h_.size() || !std::equal(h_.begin(), h_.end(), toks.begin())) {
        h_.clear();
        idx_.clear();
    }
    for (size_t i = h_.size(); i < toks.size(); i++) {
        h_.push_back(toks[i]);
        add(i);
    }
}

int SuffixIndex::draft(int32_t last, int k, std::vector<int32_t>& out) const {
    out.clear();
    const size_t n = h_.size();
    if (n + 1 < (size_t)KEY + 1) return 0;
    // end of the sequence h_ + last
    auto at = [&](size_t j) { return j == n ? last : h_[j]; };
    int32_t tail[KEY];
    for (int i = 0; i < KEY; i++) tail[i] = at(n + 1 - KEY + i);
    const auto it = idx_.find(key(tail));
    if (it == idx_.end()) return 0;
    int best = 0;
    size_t best_i = 0;
    const auto& v = it->second;
    const size_t lo = v.size() > MAX_POS ? v.size() - MAX_POS : 0;
    for (size_t c = v.size(); c-- > lo;) {
        const size_t i = v[c];
        if (i + 1 >= n) continue;  // nothing follows it
        int len = 0;
        while (len < MAX_MATCH && len <= (int)i && at(n - len) == h_[i - len]) len++;
        if (len > best) best = len, best_i = i;  // most recent first: a tie keeps the younger one
    }
    if (best < KEY) return 0;
    for (size_t j = best_i + 1; j < n && (int)out.size() < k; j++) out.push_back(h_[j]);
    return best;
}

}  // namespace tona::server
