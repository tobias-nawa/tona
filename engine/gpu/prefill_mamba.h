// Prefill kernels of the Mamba-2 layers (Bolt), the counterparts of OP_MAMBA_CONV, OP_MAMBA_SCAN and OP_MAMBA_NORM of
// the megakernel (mega_op_mamba.h): same layout of the in-projection output, same arithmetic per token (mamba_dev.h),
// so a prompt cut anywhere ends in the same state, and a state continued by the decode path is the same too. The
// kernels are in prefill_mamba.hip. Layout of a token: see mega_op_mamba.h.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace tona::gpu {

// Causal conv + bias + SiLU of n tokens from position pos0. Earlier positions come from the ring (position mod
// K - 1 + BMAX, as in decode); the ring is written afterwards (launch_mamba_ring).
struct MambaConvArgs {
    const float* zx;  // [n][in_dim]
    const float* ring;
    const float *w, *bias;  // [conv_dim][K], [conv_dim]
    float* out;             // [n][conv_dim]
    int n;
    int64_t pos0;
    int ch, K, in_dim, off;
    float in_scale;
};
void launch_mamba_conv(const MambaConvArgs& a, hipStream_t st);

// The conv inputs of the last min(n, K - 1 + BMAX) positions of the chunk into the ring
struct MambaRingArgs {
    const float* zx;
    float* ring;
    int n;
    int64_t pos0;
    int ch, K, in_dim, off;
    float in_scale;
};
void launch_mamba_ring(const MambaRingArgs& a, hipStream_t st);

// decay = exp(dt A) and dt = softplus(dt_raw + bias) per token and head, the same function the decode step evaluates
// inline (mb_token). Computed once for all the waves of a head: the scan would repeat the logs and exponentials per row
// group.
struct MambaDtArgs {
    const float *zx, *dt_bias, *A;
    float *decay, *dt;  // [n][heads]
    float* da;          // dt * A (nullptr: not written), for the chunked scan
    int n, heads, in_dim, dt_off;
    float in_scale;
};
void launch_mamba_dt(const MambaDtArgs& a, hipStream_t st);

// Scan over n tokens, one after the other. First commits the m tokens still open from the last decode pass (half `half`
// of the pending buffers). y: [n][inner] with the skip x * D added.
struct MambaScanArgs {
    const float *conv, *decay, *dt, *D;
    float *state, *y;
    const float *pdec, *pxdt, *pb;
    int n, m, half;
    int heads, P, groups, conv_dim, inner;
};
void launch_mamba_scan(const MambaScanArgs& a, hipStream_t st);

// The chunked (SSD) form of the scan on the matrix units, prefill_mamba_ssd.hip: same inputs and outputs as
// launch_mamba_scan (without the commit of open decode tokens, which launch_mamba_scan with n = 0 does first), plus da
// = dt * A per token and head (launch_mamba_dt writes it) and scratch cb of ssd_chunks * groups * 64 * 64 * 3 floats
// (the scores of every (chunk, group), then C and B^T as f16 for the scan). The chunk grid is the absolute position /
// 64: a prompt cut at multiples of 64 gives the same bits as the whole prompt; the state differs from the sequential
// scan's by the rounding of f16 operands.
struct SsdArgs {
    const float *conv, *da, *dt, *D;
    float *state, *y, *cb;
    int64_t pos0;
    int n, nc;  // nc: set by launch_mamba_ssd
    int heads, P, groups, conv_dim, inner;
    _Float16 *c16 = nullptr, *bt16 = nullptr;  // set by launch_mamba_ssd inside cb
};
int ssd_chunks(int64_t pos0, int n);
bool ssd_supported(int heads, int P, int state, int groups);
void launch_mamba_ssd(SsdArgs a, hipStream_t st);

// y * silu(z), RMS norm per group, times out_scale, as f16 for the out-projection GEMM
struct MambaNormArgs {
    const float *y, *zx, *w;
    _Float16* o;
    int n, inner, groups, in_dim;
    float eps, in_scale, out_scale;
};
void launch_mamba_norm(const MambaNormArgs& a, hipStream_t st);

// Shapes the kernels are written for (state 128, rows a multiple of 8, a group a multiple of 32 values); throws
void mamba_check_shapes(int heads, int P, int state, int groups, int conv);

// The state size the kernels are written for (rows of the state: heads * P, columns: MB_STATE)
constexpr int MB_STATE = 128;

}  // namespace tona::gpu
