// The latent MoE of Bolt in the prefill, around the shared streaming of the experts (prefill_moe_run.cpp): the router
// on the full-width input chooses by sigmoid with a selection bias, the experts work on lat_down(x) in the latent space
// (ungated, squared ReLU), their weighted sum goes through lat_up, and the shared expert (full width, no gate) is
// added. The decode counterpart is moe.cpp and mega_op_moe_latent.h; the oracle is NemotronHOracle::moe.
#include <cstdint>

#include "gpu/expert_parts.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_check.h"
#include "gpu/prefill_moe_latent.h"
#include "gpu/prefill_route_sig.h"
#include "gpu/prefill_run.h"

namespace tona::gpu {

// up, relu(.)^2, down for the experts of one batch: weights are read in their format and unpacked in the LDS like the
// gated path's. Rows of the experts: in values (d or latent), ff in the middle, out values (the latent width again).
void PrefillRun::run_batch_ungated(uint32_t l, const MoeBufs& b, const std::vector<int>& used, const BatchPlan& bp,
                                   const ExpertBatch& eb) {
    const MoeWeights& S = M.m_.moe_weights(l);
    const int in = (int)S.up_exps->dims[0], ff = (int)S.up_exps->dims[1], out = (int)S.down_exps->dims[1];
    const int m = eb.m;
    const int32_t* cnt = M.pf_hostc_;
    const int32_t* off = M.pf_hostc_ + route_off_word(D.max_e);
    const int64_t rbu = row_bytes_of(S.up_exps->type, in), rbd = row_bytes_of(S.down_exps->type, ff);
    GemmQSeg su[MB], sd[MB];
    for (int s = 0; s < m; s++) {
        const int e = used[bp.b0 + s];
        su[s] = {b.xg + (size_t)off[e] * in, eb.wu[s], b.U + (size_t)off[e] * ff, cnt[e], ff, in, in, ff, rbu};
        sd[s] = {b.A16 + (size_t)off[e] * ff, eb.wd[s], b.O + (size_t)off[e] * out, cnt[e], out, ff, ff, out, rbd};
    }
    gemm_q(S.up_exps->type, su, m, st);
    const int e0 = used[bp.b0], e1 = used[bp.b0 + m - 1];
    const int64_t r0 = off[e0], rows = off[e1] + cnt[e1] - r0;
    launch_relu2_16({b.U, b.A16, r0, rows, ff}, st);
    gemm_q(S.down_exps->type, sd, m, st);
}

// Router and the shared expert's first half while the CPU reads the counts; returns the latent input in b.A16 (f16)
// The buffers are the shared MoE phase's: G (float) holds lat_down(x) before it becomes f16 in A16 (rows of the experts
// are written to A16 only after the scatter has read it, in stream order).
void PrefillRun::moe_latent(uint32_t l, int j) {
    const MoeBufs b = moe_buffers();
    const PfMoe& F = M.pf_[M.m_.moe_block(l)].moe;
    // ne from the model, not from cache_[l]: the layered prefill resets the LayerCache of a layer whose
    // slots it borrows (n_used 0), and the scatter of every job past the first chunk launched zero blocks
    const int d = D.d, sff = D.sff, E = (int)M.experts_.experts(l);
    const int ne = (int)M.m_.expert_used(l);
    const int lat = (int)F.lat_down.rows;
    gemmq_f16(F.router, B.mixed16, d, b.rl, E, n);
    HIPCHECK(hipMemsetAsync(b.cnt_d, 0, route_count_words(D.max_e) * sizeof(int32_t), st));
    launch_route_sig({b.rl, F.router_bias, b.sel, b.sw, b.cnt_d, n, E, ne, route_nan_word(D.max_e), F.route_scale}, st);
    HIPCHECK(hipMemcpyAsync(M.pf_hostc_, b.cnt_d, E * sizeof(int32_t), hipMemcpyDeviceToHost, st));
    HIPCHECK(hipMemcpyAsync(M.pf_hostc_ + route_nan_word(D.max_e), b.cnt_d + route_nan_word(D.max_e), sizeof(int32_t),
                            hipMemcpyDeviceToHost, st));
    share_x8(true);
    gemmq(F.sh_up, B.mixed16, d, b.su, sff, n);
    gemmq(F.lat_down, B.mixed16, d, b.G, lat, n);
    share_x8(false);
    launch_relu2_16({b.su, b.sa16, 0, n, sff}, st);
    gemmq(F.sh_down, b.sa16, sff, b.sho, d, n);
    launch_f32_to_f16(b.G, b.A16, (int64_t)n * lat, st);
    const std::vector<int> used = moe_group(l, b, b.A16, lat, ne);
    moe_experts(l, j, b, used);
    if (flags_.check) moe_check(l, b, 2, ne);  // the rowmap must survive the expert GEMMs
    // The sum in the latent space (f16, in xg: the experts are done with it), back to the full width, plus the shared
    // expert
    launch_moe_sum_latent({b.O, b.rowmap, b.sw, b.xg, n, lat, ne}, st);
    gemmq_out(F.lat_up, b.xg, lat, B.y, d, n);
    launch_add_into(B.y, b.sho, (int64_t)n * d, st);
}

}  // namespace tona::gpu
