// Drafts from the history (SuffixDecoding, arXiv 2411.04975; Baseten: a long suffix match overrides the model
// draft). Agents repeat files, diffs and paths from the prompt: if the end of the text already occurred in the
// history, what followed there is a good draft without running the drafter.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace tona::server {

class SuffixIndex {
public:
    // Bring the index up to toks: append if toks continues the history so far, otherwise rebuild
    void sync(const std::vector<int32_t>& toks);
    // Longest earlier match of the end of history + last (at least KEY tokens, at most MAX_MATCH); the most recent
    // one on a tie. out: up to k tokens that followed there. Returns the match length (0: none)
    int draft(int32_t last, int k, std::vector<int32_t>& out) const;

    static constexpr int KEY = 4;          // key: the last 4 tokens
    static constexpr int MAX_MATCH = 64;   // comparing further back gains nothing
    static constexpr size_t MAX_POS = 64;  // only the most recent positions per key

private:
    void add(size_t i);  // key that ends at h_[i]
    static uint64_t key(const int32_t* t);

    std::vector<int32_t> h_;
    std::unordered_map<uint64_t, std::vector<uint32_t>> idx_;
};

}  // namespace tona::server
