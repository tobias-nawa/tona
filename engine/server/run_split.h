// How a stretch of a prompt is computed: how many of its tokens go through the prefill kernels and whether the last one
// goes through forward. Kept apart from the generator so a unit test can state the rule without a GPU.
#pragma once

#include <cstddef>

namespace tona::server {

struct RunSplit {
    size_t prefill;  // tokens through the prefill kernels, from the start of the stretch
    size_t forward;  // tokens through forward behind them (passes of 4, they leave logits)
};

// The canonical computation of a prompt: all tokens but the last through the prefill, the last through forward
// (it leaves the logits the sampler needs). A stretch that ends in a checkpoint inside the prompt (final = false) is
// all prefill: a forward token there would round that one position differently from a cold run of the same prompt, and
// a hit on the checkpoint would not reproduce it. `force` keeps short stretches in the prefill when they follow a
// checkpoint (a cold run of the whole prompt prefilled them); otherwise a stretch of at most `min_tokens` goes through
// forward, where the prefill setup costs more than it saves. prefill_on = false (TONA_PREFILL_CHUNK=0): forward only.
inline RunSplit split_run(size_t n, bool final, bool force, size_t min_tokens, bool prefill_on) {
    if (!prefill_on || n == 0) return {0, n};
    if (!final) return {n, 0};
    if (!force && n <= min_tokens) return {0, n};
    return {n - 1, 1};
}

}  // namespace tona::server
