// Building blocks of the op programs that every architecture uses: matvec phases and ops.
#include "backend/env.h"
#include "gpu/mega_build.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

namespace {

// Dense formats whose int8 tasks may run on the wide (WMMA) path
bool wide_type(uint32_t type) {
    return type == 8 || type == 12 || type == 13 || type == 14 || type == 23 || type == 1019 || type == 16 ||
           type == 40;
}

}  // namespace

void ProgramBuilder::mv_begin() { phase0_ = (int)tg_.tasks->size(); }

void ProgramBuilder::mv_end(const char* label) const {
    MegaOp o;
    o.kind = OP_MATVEC, o.a = phase0_, o.b = (int)tg_.tasks->size();
    if (o.b > o.a) tg_.ops->push_back(o), tg_.labels->emplace_back(label);
}

// Distances per token come from the buffers
void ProgramBuilder::add(const Mv& mv) { add(mv, MvExtra{}); }

void ProgramBuilder::add(const Mv& mv, const MvExtra& ex) {
    MegaTask t;
    const bool q8 = g_.int8_ && mv.xq && matvec_q8_supported(mv.type);
    // The tile type has no float path (its rows are not contiguous): a task without int8 x would compute nothing
    if (mv.type == 1019 && !q8) throw std::runtime_error("gpu::Mega: TONA_Q6_K_T matrix without int8 x");
    // Likewise NVFP4 and IQ2_XXS: the float path has no loops for them
    if ((mv.type == 16 || mv.type == 40) && !q8)
        throw std::runtime_error("gpu::Mega: NVFP4 or IQ2_XXS matrix without int8 x");
    t.t = {mv.type, mv.in, mv.rows, mv.w, mv.x, mv.y, q8 ? *mv.xq : Q8Vec{}};
    t.slot_e = ex.slot_e, t.slot_stride = ex.slot_stride, t.act_scale = ex.act_scale;
    t.xb = g_.stride_of(mv.x);
    t.yb = g_.stride_of(mv.y);
    if (q8 && g_.stride_of(mv.xq->q) != t.xb)
        throw std::runtime_error("gpu::Mega: distance per token of x and of int8 x differ");
    // Wide passes: dense int8 tasks of the target model whose x also exists as float. Only matrices with many rows:
    // wide_loop runs 16 rows per wave over the whole row length, with few rows too few waves work (320 x 10240: 20
    // of 512). TONA_WIDE_ROWS = minimum number of rows
    static const int64_t wide_rows = env::integer("TONA_WIDE_ROWS", 4096);
    t.wide = wide_mark_ && q8 && mv.in % 64 == 0 && mv.rows >= wide_rows && wide_type(mv.type);
    tg_.tasks->push_back(t);
}

void ProgramBuilder::addw(const TensorInfo& t, const void* w, const float* x, const Q8Vec* xq, float* y,
                          float act_scale) {
    const int64_t in = (int64_t)t.dims[0];
    add({g_.weight_type(t, w), in, (int64_t)t.elements() / in, w, x, xq, y}, {-1, 0, act_scale});
}

void ProgramBuilder::op(int kind, std::initializer_list<void*> p, std::initializer_list<int> i,
                        std::initializer_list<float> f) const {
    MegaOp o;
    o.kind = kind;
    int k = 0;
    for (void* v : p) o.p[k++] = v;
    k = 0;
    for (int v : i) o.i[k++] = v;
    k = 0;
    for (float v : f) o.f[k++] = v;
    tg_.ops->push_back(o);
    tg_.labels->emplace_back(kind_names[kind]);
}

// One task for UMAX expert places at once (before: 2 per expert; the bookkeeping per task cost more than the work)
void ProgramBuilder::group(const Mv& mv, int64_t slot_bytes, int64_t x_e, int64_t y_e) {
    add(mv, {0, slot_bytes, 0});
    tg_.tasks->back().n_exp = UMAX, tg_.tasks->back().x_e = x_e, tg_.tasks->back().y_e = y_e;
}

PfW ProgramBuilder::pw(const TensorInfo& t, const void* w) const {
    const int64_t in = (int64_t)t.dims[0];
    return PfW{g_.weight_type(t, w), in, (int64_t)t.elements() / in, w};
}

}  // namespace tona::gpu
