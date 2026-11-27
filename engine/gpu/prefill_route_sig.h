// Prefill routing with the sigmoid policy (backend/route_policy.h): one wave per token, the n_used largest score +
// bias, weights = score / sum of the chosen scores * scale. The counterpart of k_route (softmax) for architectures that
// route by sigmoid; the rules are the CPU side's (route_policy.h), so the selection and the weights are the same bits.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace tona::gpu {

// logits: [n][E] router logits; bias: [E] selection bias or null; sel, sw: [n][ne] chosen experts in rank order and
// their weights; cnt[e]: tokens per expert (added to); nan_word >= 0: cnt[nan_word] counts the tokens with a NaN or
// +inf logit. E <= 512, ne <= 16.
struct RouteSigArgs {
    const float* logits;
    const float* bias;
    int32_t* sel;
    float* sw;
    int32_t* cnt;
    int n, E, ne, nan_word;
    float scale;
};
void launch_route_sig(const RouteSigArgs& a, hipStream_t st);

}  // namespace tona::gpu
