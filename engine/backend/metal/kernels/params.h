// Parameter structs of the Metal kernels, shared by the kernels (MSL) and the host encoders (C++), so both sides
// always agree on the layout.
#pragma once
#ifdef __METAL_VERSION__
#define TONA_METAL_BEGIN
#define TONA_METAL_END
#define TONA_CONST constant
#else
#include <cstdint>
#define TONA_METAL_BEGIN namespace tona::metal {
#define TONA_METAL_END }
#define TONA_CONST constexpr
#endif

TONA_METAL_BEGIN
#ifndef __METAL_VERSION__
using uint = uint32_t;
#endif

// Decode attention in parts of ATTN_PART cells (attend_part), at most ATTN_PARTS per head, then attend_join
TONA_CONST uint ATTN_PART = 128;
TONA_CONST uint ATTN_PARTS = 32;
TONA_CONST uint ATTN_MAX_CELLS = ATTN_PART * ATTN_PARTS;

struct MvParams {
    uint rows, k, row_bytes;
    uint ebytes;   // bytes per expert; 0 for dense matrices
    uint xstride;  // floats between the inputs of two slots; 0: all slots read the same x
    uint use_ids;  // 1: slot s reads expert ids[s]
    uint xdiv;     // slot s reads input s / xdiv (experts of token b: xdiv = experts per token)
    uint tokens;   // mvm only: tokens of the dispatch, 1 to 8 (columns of its 8x8 tile)
};

struct GemmParams {
    uint rows, k, row_bytes, tokens;
    uint ebytes;  // routed experts: bytes per expert
    uint xdiv;    // routed experts: slot s reads x row s / xdiv
};

struct GroupParams {
    uint n_expert, slots;
    uint split;  // slots per glist entry (mvg threadgroup row); 0: all slots of an expert in one entry
};

struct GatherParams {
    uint n, row_bytes, copies;
};

struct NormParams {
    uint n, in_stride, out_stride, w_stride, use_w;
    uint w_rows;  // weight row of row r: r % w_rows
    float eps, post;
};

struct ScaleParams {
    uint n;
    float s;
};

struct HcParams {
    uint d, hc, nb;
};

struct NormRopeParams {
    uint dim, head_stride, rope_dims;
    uint heads;         // heads per token; token t has its own cos/sin (cs + t * rope_dims)
    uint token_stride;  // floats between the first heads of two tokens
    float eps;
};

struct DnParams {
    uint ch;     // conv channels: q, k (nk heads each) and v (nv heads)
    uint kconv;  // conv taps (4)
    uint nk, nv;
    uint nb;  // tokens of the pass, in order
    float eps;
};

struct PoolParams {
    uint idim, blk, b0, rope_dims;
    float eps;
};

struct ScoreParams {
    uint idim, heads, n_pool;
};

struct TopParams {
    uint n_pool, n_top, blk;
};

struct ScoreTParams {
    uint idim, heads, q_row, score_row, pos0, nb, blk;
};

struct SelectParams {
    uint pos0, blk, n_top, score_row, bm_row;
};

struct AttnTParams {
    uint nh, nkv, hd, q_row, o_row, blk, pos0, nb, bm_row;
    uint t_base;  // first token of this dispatch (attention runs in slices, each its own command buffer)
    float scale;
};

struct AttnParams {
    uint nh, nkv, hd, ncells;
    float scale;
};

struct RouteParams {
    uint n_expert, n_used;
};

struct SumParams {
    uint d, n_used, nb;
};

struct PickParams {
    uint rows, per, groups;  // rows of the head, rows per threadgroup, threadgroups (at most 256)
};

struct PleParams {
    uint d, hc, kconv, dil;  // stream width, streams, conv taps, dilation (n-gram order)
    uint hist, start;        // ring of the last (kconv - 1) * dil normalized inputs; start = oldest
    uint nb;                 // tokens of the pass, in order
    float eps;
};

TONA_METAL_END
