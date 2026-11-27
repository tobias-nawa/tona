#include "cpu/ref_mamba2.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tona::ref {

void ssm_step(const SsmStepInputs& in, float* state, float* y) {
    const int64_t per_group = in.heads / in.groups;
    for (int64_t h = 0; h < in.heads; h++) {
        const float dt = softplus(in.dt[h]);
        const float decay = std::exp(dt * in.a[h]);
        const float* B = in.B + (h / per_group) * in.state;
        const float* C = in.C + (h / per_group) * in.state;
        for (int64_t i = 0; i < in.head_dim; i++) {
            const int64_t at = h * in.head_dim + i;
            const float x_dt = in.x[at] * dt;
            float* s = state + at * in.state;
            double acc = 0;
            for (int64_t n = 0; n < in.state; n++) {
                s[n] = s[n] * decay + B[n] * x_dt;
                acc += (double)s[n] * C[n];
            }
            y[at] = (float)acc + in.x[at] * in.d[h];
        }
    }
}

namespace {

// y = W x times the tensor's F32 scale (NVFP4 tensors carry one; llama.cpp multiplies after the matmul)
void project(const TensorInfo& w, const TensorInfo* scale, const float* x, float* y) {
    matvec(w, x, y);
    if (!scale) return;
    const float s = dequant_all(*scale)[0];
    for (int64_t o = 0; o < (int64_t)w.dims[1]; o++) y[o] *= s;
}

// Causal depthwise conv over [conv - 1 old inputs; the current one], weight (k, channel) at channel * conv + k, bias,
// SiLU; the state keeps the last conv - 1 inputs. v is the conv input of the token [channels].
void conv_step(const Mamba2Weights& w, int64_t K, float* state, std::vector<float>& v) {
    const int64_t ch = (int64_t)v.size();
    const auto cw = dequant_all(*w.conv1d);
    const auto bias = dequant_all(*w.conv1d_bias);
    std::vector<float> out(ch);
    for (int64_t c = 0; c < ch; c++) {
        float acc = 0;
        for (int64_t k = 0; k < K - 1; k++) acc += cw[c * K + k] * state[k * ch + c];
        acc += cw[c * K + K - 1] * v[c];
        out[c] = silu(acc + bias[c]);
    }
    std::memmove(state, state + ch, (K - 2) * ch * sizeof(float));
    std::memcpy(state + (K - 2) * ch, v.data(), ch * sizeof(float));
    v = std::move(out);
}

}  // namespace

std::vector<float> mamba2_block(const Mamba2Weights& w, const Mamba2Dims& D, float eps, const float* x,
                                const Mamba2State& st) {
    const int64_t inner = (int64_t)D.inner, heads = (int64_t)D.heads, G = (int64_t)D.groups, N = (int64_t)D.state;
    const int64_t conv_dim = (int64_t)D.conv_dim();
    if (heads <= 0 || G <= 0 || heads % G != 0 || inner % heads != 0)
        throw std::runtime_error("ref::mamba2: heads, groups and inner size do not fit");

    // in_proj output: [z (inner) | x B C (conv_dim) | dt (heads)]
    std::vector<float> zxbcdt(D.in_proj_dim());
    project(*w.in_proj, w.in_scale, x, zxbcdt.data());
    std::vector<float> xbc(zxbcdt.begin() + inner, zxbcdt.begin() + inner + conv_dim);
    std::vector<float> dt(zxbcdt.end() - heads, zxbcdt.end());
    const auto dt_bias = dequant_all(*w.dt_bias), A = dequant_all(*w.a), Dskip = dequant_all(*w.dskip);
    for (int64_t h = 0; h < heads; h++) dt[h] += dt_bias[h];

    conv_step(w, (int64_t)D.conv, st.conv, xbc);

    std::vector<float> y(inner);
    const SsmStepInputs in{xbc.data(), xbc.data() + inner, xbc.data() + inner + G * N,
                           dt.data(),  A.data(),           Dskip.data(),
                           heads,      inner / heads,      N,
                           G};
    ssm_step(in, st.ssm, y.data());

    // silu(z) * y, then RMS norm per group of inner / G values with the group's own weights
    const auto norm_w = dequant_all(*w.norm);
    for (int64_t i = 0; i < inner; i++) y[i] *= silu(zxbcdt[i]);
    const int64_t group = inner / G;
    for (int64_t g = 0; g < G; g++) rms_norm(&y[g * group], group, eps, &norm_w[g * group]);

    std::vector<float> out(D.d);
    project(*w.out_proj, w.out_scale, y.data(), out.data());
    return out;
}

}  // namespace tona::ref
