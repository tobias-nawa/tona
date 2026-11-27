// Operations of the CPU reference as pure functions on spans and tensors, shared by the oracles of every architecture
// (docs/architecture/model-architectures.md section 3.6). Math in F32, dot products in double, weights dequantized row
// by row: slow and simple, the arithmetic the GPU kernels are checked against. An oracle composes these; it adds no
// math of its own that another architecture could share.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "model/gguf.h"

namespace tona::ref {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }  // as ggml_softplus

// One row (n elements from the start of row `row`) or the whole tensor as F32
std::vector<float> dequant(const TensorInfo& t, int64_t row, int64_t n);
std::vector<float> dequant_all(const TensorInfo& t);

// y = W x for a matrix in ggml order [in, out]; expert picks the slice of an [in, out, E] tensor
void matvec(const TensorInfo& w, const float* x, float* y, int64_t expert = 0);

// x <- x / sqrt(mean(x^2) + eps) * w (w may be null), as ggml_rms_norm followed by ggml_mul
void rms_norm(float* x, int64_t n, float eps, const float* w = nullptr);

// NeoX RoPE on the first n_rot dimensions: pairs (i, i + n_rot/2), theta_i = pos * base^(-2i/n_rot), with the YaRN
// table of backend/rope_yarn.h (TONA_ROPE_YARN) as on the GPU
void rope(float* x, int64_t n_rot, int64_t pos, float base);

enum class OutGate { Sigmoid, Silu };  // the Coder gates the readout with sigmoid(z), Qwen3.5/Ornith with silu(z)

// Gated DeltaNet, one token: per V head the state is decayed, corrected by the delta rule and read out with q, then the
// readout is RMS-normalized per head and gated by sigmoid(z) or silu(z) (out_gate). q and k are already L2-normalized
// per head; V head j uses K head j mod k_heads (the GGUF's head order).
struct DeltaNetInputs {
    const float *q, *k, *v;     // after conv and L2 norm: [k_heads x S], [k_heads x S], [v_heads x S]
    const float* z;             // output gate logits [v_heads x S]
    const float *beta, *alpha;  // raw projections [v_heads]
    const float *dt_bias, *a;   // ssm_dt.bias and ssm_a (decay rate), [v_heads]
    const float* norm_w;        // per-head readout norm [S]
    int64_t k_heads, v_heads, head_dim;
    float eps;
    OutGate out_gate = OutGate::Sigmoid;
};
// state: [v_heads][S (value)][S (key)], updated in place; o: [v_heads x S]
void deltanet_recurrence(const DeltaNetInputs& in, float* state, float* o);

// Softmax attention of one query token over cached cells. K and V hold [position][n_head_kv x head_dim]; query head
// h reads KV head h / (n_head / n_head_kv). Returns [n_head x head_dim].
struct AttendShape {
    int64_t n_head, n_head_kv, head_dim;
};
std::vector<float> attend_cells(const AttendShape& s, const float* q, const std::vector<float>& K,
                                const std::vector<float>& V, const std::vector<int64_t>& cells);

// Router with softmax over all experts: the n_used most probable (ties by lower index), weights renormalized to sum 1
struct Routing {
    std::vector<int> experts;
    std::vector<float> weights;  // per chosen expert, same order
};
Routing route_softmax_topk(const std::vector<float>& logits, int64_t n_used);

// down(silu(gate(x)) * up(x)) for one expert (expert 0 for a plain 2D tensor such as the shared expert)
std::vector<float> swiglu_expert(const TensorInfo& gate, const TensorInfo& up, const TensorInfo& down, const float* x,
                                 int64_t expert);

}  // namespace tona::ref
