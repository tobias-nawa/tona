// Model-free speculation after SuffixDecoding (arXiv 2411.04975): suffix tries over token sequences, from which
// continuations of the current context are proposed as a draft. One trie per request (prompt and output so far) and
// one global trie over earlier outputs.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace tona::spec {

// Trie of all subsequences up to depth max_depth, with a count per node. It grows token by token: the nodes of the
// last max_depth suffixes stay active.
class SuffixTrie {
public:
    explicit SuffixTrie(int max_depth) : max_depth_(max_depth) { nodes_.push_back({}); }

    void append(int32_t token);
    void append(const std::vector<int32_t>& tokens) {
        for (int32_t t : tokens) append(t);
    }
    // Context boundary: new suffixes start at the root again (between outputs of different requests)
    void cut() { active_.clear(); }

    int32_t child(int32_t node, int32_t token) const;
    int max_depth() const { return max_depth_; }
    size_t size() const { return nodes_.size(); }

    struct Node {
        int32_t count = 0;
        int32_t first_child = -1, next_sibling = -1;
        int32_t token = -1;
        int16_t depth = 0;
    };
    const Node& node(int32_t i) const { return nodes_[i]; }

private:
    int32_t get_or_add(int32_t parent, int32_t token);

    int max_depth_;
    std::vector<Node> nodes_;
    std::unordered_map<uint64_t, int32_t> edges_;  // (parent << 32 | token) -> child
    std::vector<int32_t> active_;                  // nodes of the running suffixes
};

// A draft: tree of tokens, parent[i] < i, parent = -1 means attached to the current context
struct Draft {
    std::vector<int32_t> token, parent;
    std::vector<float> prob;  // estimated acceptance probability D(N)
    float score = 0;          // sum of D = expected accepted tokens
    int match_len = 0;
    size_t size() const { return token.size(); }
};

struct SpecParams {
    int max_pattern = 32;   // longest suffix of the context that is searched
    float alpha = 2.0f;     // MAX_SPEC = alpha * pattern length
    int max_spec = 16;      // fixed upper bound of the draft size
    bool tree = false;      // false: a single path (recurrent layers would otherwise need state per branch)
    float min_prob = 0.0f;  // do not include nodes below this probability
};

// Keeps the matches of the last context suffixes in a trie up to date (O(max_pattern) per token)
class Matcher {
public:
    Matcher(const SuffixTrie& trie, int max_pattern) : trie_(trie), max_pattern_(max_pattern) {}
    void push(int32_t token);
    void reset() { match_.clear(); }
    // match_[p-1] = node for the last p tokens, -1 without a match
    const std::vector<int32_t>& matches() const { return match_; }

private:
    const SuffixTrie& trie_;
    int max_pattern_;
    std::vector<int32_t> match_;
};

// Best draft over all pattern lengths from one trie
Draft propose(const SuffixTrie& trie, const std::vector<int32_t>& matches, const SpecParams& p);

// How many draft tokens would be accepted against the actual continuation (longest matching path)
int accepted(const Draft& d, const int32_t* actual, int n_actual);

}  // namespace tona::spec
