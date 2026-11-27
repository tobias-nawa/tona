// How the router logits of a layer become experts and weights. One header for the host (backend/expert_route.h, the CPU
// side of the selection) and for the GPU (the routing step of the megakernel, the prefill routing kernel): the rules of
// the policies are defined here once, so the sides cannot drift apart.
//
//   Softmax, renormalized (the Coder, Ornith): p = softmax(r), the n_used largest p, weight = p / sum of the chosen p.
//     The code of this policy stays where it was (route_rank.h, op_route, k_route): its rounding is part of the
//     goldens.
//   Sigmoid with bias (Bolt; the rule of DeepSeek-V3 and Nemotron-H with `exp_probs_b`, one group):
//     s = sigmoid(r), the n_used largest s + bias, weight = s / max(sum of the chosen s, 2^-14) * scale.
//     The bias decides the choice only, the weight does not contain it. Ties go to the smaller expert index.
//
// The sigmoid policy is new, so it is written for bit-identical results on both sides: only IEEE operations in a fixed
// order and the deterministic exp of cpu/detmath.h, nothing contracted (clang honors the pragmas below; another
// compiler needs -ffp-contract=off for the including file, as for cpu/detmath.h).
#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "cpu/detmath.h"

namespace tona::backend {

enum class Gate : int32_t { Softmax = 0, Sigmoid = 1 };

// The policy of one layer. The bias belongs to the policy of its layer: it is a tensor of the model (`exp_probs_b`).
struct RoutePolicy {
    Gate gate = Gate::Softmax;
    float scale = 1.0f;           // Sigmoid: factor on the normalized weights (Bolt: 5.0)
    const float* bias = nullptr;  // Sigmoid: selection bias per expert; null: none
};

namespace sigmoid_route {

// Floor of the sum of the chosen scores before the division (llama.cpp clamps the weight sum at 2^-14)
inline constexpr float kMinSum = 6.103515625e-5f;

// A logit as the selection sees it: NaN is -FLT_MAX (score 0), infinities are +-FLT_MAX; keeps every score a number in
// [0, 1] and the ranking a total order (the same cleaning as logit_clamp in the GPU code and nan_guard.h)
TONA_HD inline float clean(float r) { return r != r ? -FLT_MAX : fminf(fmaxf(r, -FLT_MAX), FLT_MAX); }

TONA_HD inline float score(float r) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    return 1.0f / (1.0f + tona::det::exp(-clean(r)));
}

// What the choice ranks by; a bias that makes the sum NaN (inf - inf) ranks last
TONA_HD inline float key(float s, float bias) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float k = s + bias;
    return k != k ? -FLT_MAX : k;
}

// Weight of a chosen expert with score s; sum: the scores of all chosen experts added in rank order
TONA_HD inline float weight(float s, float sum, float scale) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    return s / fmaxf(sum, kMinSum) * scale;
}

}  // namespace sigmoid_route

}  // namespace tona::backend
