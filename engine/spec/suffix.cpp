#include "spec/suffix.h"

#include <algorithm>
#include <queue>

namespace tona::spec {

int32_t SuffixTrie::child(int32_t node, int32_t token) const {
    auto it = edges_.find((uint64_t)(uint32_t)node << 32 | (uint32_t)token);
    return it == edges_.end() ? -1 : it->second;
}

int32_t SuffixTrie::get_or_add(int32_t parent, int32_t token) {
    const uint64_t key = (uint64_t)(uint32_t)parent << 32 | (uint32_t)token;
    auto [it, fresh] = edges_.try_emplace(key, (int32_t)nodes_.size());
    if (fresh) {
        Node n;
        n.token = token;
        n.depth = (int16_t)(nodes_[parent].depth + 1);
        n.next_sibling = nodes_[parent].first_child;
        nodes_.push_back(n);
        nodes_[parent].first_child = it->second;
    }
    return it->second;
}

void SuffixTrie::append(int32_t token) {
    // Every running suffix grows by the token, and a new one starts at the root
    std::vector<int32_t> next;
    next.reserve(active_.size() + 1);
    for (int32_t n : active_) {
        if (nodes_[n].depth >= max_depth_) continue;
        next.push_back(get_or_add(n, token));
    }
    next.push_back(get_or_add(0, token));
    for (int32_t n : next) nodes_[n].count++;
    active_ = std::move(next);
}

void Matcher::push(int32_t token) {
    // Match for p tokens = child of the match for p-1 tokens at the previous position
    std::vector<int32_t> next(std::min<size_t>(match_.size() + 1, max_pattern_), -1);
    next[0] = trie_.child(0, token);
    for (size_t p = 1; p < next.size(); p++)
        if (match_[p - 1] >= 0) next[p] = trie_.child(match_[p - 1], token);
    while (!next.empty() && next.back() < 0) next.pop_back();
    match_ = std::move(next);
}

namespace {

Draft expand(const SuffixTrie& trie, int32_t start, int match_len, const SpecParams& p) {
    Draft d;
    d.match_len = match_len;
    const int limit = std::min(p.max_spec, std::max(1, (int)(p.alpha * match_len)));
    struct Cand {
        float prob;
        int32_t node, parent_idx;
        bool operator<(const Cand& o) const { return prob < o.prob; }
    };
    std::priority_queue<Cand> q;
    auto push_children = [&](int32_t node, float base, int32_t parent_idx) {
        int64_t total = 0;
        for (int32_t c = trie.node(node).first_child; c >= 0; c = trie.node(c).next_sibling)
            total += trie.node(c).count;
        if (total == 0) return;
        if (p.tree) {
            for (int32_t c = trie.node(node).first_child; c >= 0; c = trie.node(c).next_sibling)
                q.push({base * trie.node(c).count / (float)total, c, parent_idx});
        } else {
            // only the most frequent successor
            int32_t best = -1;
            for (int32_t c = trie.node(node).first_child; c >= 0; c = trie.node(c).next_sibling)
                if (best < 0 || trie.node(c).count > trie.node(best).count) best = c;
            q.push({base * trie.node(best).count / (float)total, best, parent_idx});
        }
    };
    push_children(start, 1.0f, -1);
    while (!q.empty() && (int)d.size() < limit) {
        const Cand c = q.top();
        q.pop();
        if (c.prob < p.min_prob) break;
        const int32_t idx = (int32_t)d.size();
        d.token.push_back(trie.node(c.node).token);
        d.parent.push_back(c.parent_idx);
        d.prob.push_back(c.prob);
        d.score += c.prob;
        push_children(c.node, c.prob, idx);
    }
    return d;
}

}  // namespace

Draft propose(const SuffixTrie& trie, const std::vector<int32_t>& matches, const SpecParams& p) {
    Draft best;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i] < 0 || trie.node(matches[i]).first_child < 0) continue;
        Draft d = expand(trie, matches[i], (int)i + 1, p);
        if (d.score > best.score) best = std::move(d);
    }
    return best;
}

int accepted(const Draft& d, const int32_t* actual, int n_actual) {
    // Path in the tree that follows the actual sequence
    int depth = 0;
    int32_t at = -1;
    while (depth < n_actual) {
        int32_t next = -1;
        for (size_t i = 0; i < d.size(); i++)
            if (d.parent[i] == at && d.token[i] == actual[depth]) {
                next = (int32_t)i;
                break;
            }
        if (next < 0) break;
        at = next;
        depth++;
    }
    return depth;
}

}  // namespace tona::spec
