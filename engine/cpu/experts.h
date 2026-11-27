// Experts on the CPU (AVX-512, Zen 5): the CPU computes missing experts straight from RAM, in parallel with the GPU,
// instead of pushing their weights (1.66 MB) over PCIe. As on the GPU: x as int8 with a scale per 32 values (same
// rounding as wave_q8), weights unpacked per 32-value sub-block (magnitude u8, sign applied to x) and multiplied
// with vpdpbusd. Expert formats in the Coder: IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (gate/up), IQ4_NL, Q2_0 (down); in Ornith:
// Q4_K (gate/up), Q6_K (down); in Bolt: IQ2_XXS (up and down).
// Bit-identical with the experts on the GPU (check: tona-expert-check).
#pragma once

#include <cstdint>
#include <vector>

namespace tona::cpu {

// Like gpu::Q8Meta: scale and sums of the two 16-value halves of a 32-value block
struct Q8Meta {
    float d;
    int16_t s0, s1;
};

bool expert_type_supported(uint32_t type);
// Bytes per row of n values in the given format
int64_t row_bytes(uint32_t type, int64_t n);

// Threads for experts_ffn: OMP_NUM_THREADS, otherwise the physical cores
int expert_threads();

// x (n values, n a multiple of 32) to int8 and metadata, rounding like wave_q8 on the GPU
void quantize_q8(const float* x, int64_t n, int8_t* q, Q8Meta* m);

// Summation order as row_q8 on the GPU with 8 lanes per row (bit-identical).
// y[b * ys + r] = row r of W * x_b for the rows [r0, r1) and nb vectors x (stride xs values, metadata xs/32);
// only the vectors with bit b in mask
struct MatvecArgs {
    uint32_t type;
    const uint8_t* w;
    int64_t row_bytes, in, r0, r1;
    const int8_t* xq;
    const Q8Meta* xm;
    int64_t xs;
    int nb;
    uint32_t mask;
    float* y;
    int64_t ys;
};
void matvec_q8(const MatvecArgs& args);

// Weights of one expert (pointers into the mapped file). gate = nullptr: an ungated expert (Bolt), up then down with
// the squared ReLU, relu(u)^2, as the activation (t_gate is not read)
struct ExpertW {
    uint32_t t_gate, t_up, t_down;
    const uint8_t *gate, *up, *down;
};

// Scratch memory for experts_ffn, reused between calls.
struct ExpertScratch {
    std::vector<float> g, u;
    std::vector<int8_t> aq;
    std::vector<Q8Meta> am;
    std::vector<int64_t> yfix;
    std::vector<uint8_t> xa;  // metadata of the activations as float fields (internal)
};
// All experts E of one layer for nb tokens: y[b * d + i] = sum over e of w[b * ne + e] * down_e(silu(gate_e x_b) * up_e
// x_b) (ungated: down_e(relu(up_e x_b)^2), the input and the output both d wide, the latent width in Bolt). w = 0:
// token b does not use expert e. x: int8 with metadata, stride d per token. Parallel over the OpenMP threads.
struct FfnArgs {
    const std::vector<ExpertW>& E;
    int64_t d, ff;
    const int8_t* xq;
    const Q8Meta* xm;
    int nb;
    const float* w;
};
void experts_ffn(const FfnArgs& args, float* y, ExpertScratch& s);
// The same, y as a fixed-point sum (det::to_fix per weighted output): bit-identical with the experts on the GPU,
// whichever side computes which expert (moe.sum adds both sums as integers)
void experts_ffn_fix(const FfnArgs& args, int64_t* y, ExpertScratch& s);

}  // namespace tona::cpu
