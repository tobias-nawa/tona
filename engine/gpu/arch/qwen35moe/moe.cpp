// MoE steps of an Ornith layer: router, the union of the experts of all tokens (UMAX places, a mask per token), the
// shared expert with its scalar gate, and the sum.
#include "gpu/arch/qwen35moe/program.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

// Router weights once per layer in VRAM (layer l also needs that of l + 1 for prefetching)
const void* Qwen35MoeProgram::router_weights(uint32_t l) {
    if (!router_dev_[l]) router_dev_[l] = g_.upload(*m_.layers()[l].moe.router);
    return router_dev_[l];
}

void Qwen35MoeProgram::layer_router(uint32_t l, const Qwen35Layer& S, PfLayer& F) {
    const bool pre = g_.cpu_mode_ && g_.prefetch_n_ > 0 && l + 1 < h_.n_layer;
    F.moe.router = pb_.pw(*S.moe.router, router_weights(l));
    pb_.mv_begin();
    pb_.addw(*S.moe.router, router_weights(l), b_.xn, &b_.qxn, b_.router_out);
    if (pre) pb_.addw(*m_.layers()[l + 1].moe.router, router_weights(l + 1), b_.xn, &b_.qxn, b_.pred_out);
    pb_.mv_end("router");
    const LayerCache& C = g_.cache_[l];
    if (g_.cpu_mode_)
        pb_.op(OP_HOST_WAIT,
               {g_.mbox_dev_, b_.router_out, b_.qxn.q, b_.qxn.meta, g_.cpu_io_dev_, g_.cpu_io_dev_ + g_.cpu_xm_off_,
                pre ? b_.pred_out : nullptr},
               {(int)l, dm_.d, (int)g_.experts_.experts(l)});
    else
        pb_.op(OP_HOST_WAIT, {g_.mbox_dev_, b_.router_out}, {(int)l, 0, (int)g_.experts_.experts(l)});
    pb_.program().ops->back().f[3] = HOST_WAIT_MS + 200;  // the other blocks wait with it while the CPU reads
    shared(l);
    pb_.op(OP_EXPERT_COPY, {g_.cpu_mode_ ? g_.fill_dev_ + l * g_.fill_stride_ : g_.staging_dev_, C.slots},
           {(int)C.slot_bytes, g_.cpu_mode_ ? (int)(g_.fill_stride_ / 2 / C.slot_bytes) : 0});
}

// The shared expert's gate and up and its scalar gate do not depend on the routing
void Qwen35MoeProgram::shared_matvecs(const MoeWeights& S, PfLayer& F) {
    pb_.addw(*S.shexp_gate, F.moe.sh_gate.w, b_.xn, &b_.qxn, b_.eg + UMAX * dm_.ff);
    pb_.addw(*S.shexp_up, F.moe.sh_up.w, b_.xn, &b_.qxn, b_.eu + UMAX * dm_.ff);
    pb_.addw(*S.shexp_gate_inp, F.moe.sh_inp.w, b_.xn, &b_.qxn, b_.sh_inp);
}

// With share_wait_ the other blocks compute the shared expert while block 0 waits for the CPU (host.wait without a
// barrier after it); otherwise it sits in moe.gate_up
void Qwen35MoeProgram::shared(uint32_t l) {
    const MoeWeights& S = m_.layers()[l].moe;
    PfLayer& F = g_.pf_[l];
    F.moe.sh_gate = pb_.pw(*S.shexp_gate, g_.upload(*S.shexp_gate));
    F.moe.sh_up = pb_.pw(*S.shexp_up, g_.upload(*S.shexp_up));
    F.moe.sh_inp = pb_.pw(*S.shexp_gate_inp, g_.upload(*S.shexp_gate_inp));
    if (!g_.share_wait_) return;
    pb_.program().ops->back().nobar = 1;  // host.wait
    pb_.mv_begin();
    shared_matvecs(S, F);
    pb_.mv_end("moe.shared");
    pb_.program().ops->back().i[7] = 1;
    pb_.program().ops->back().f[3] = HOST_WAIT_MS + 200;  // the barrier behind it waits for the CPU too
}

void Qwen35MoeProgram::expert_gate_up(uint32_t l, const MoeWeights& S, PfLayer& F) {
    const LayerCache& C = g_.cache_[l];
    const int d = dm_.d, ff = dm_.ff;
    pb_.mv_begin();
    pb_.group({S.gate_exps->type, d, ff, C.slots, b_.xn, &b_.qxn, b_.eg}, (int64_t)C.slot_bytes, 0, ff);
    pb_.group({S.up_exps->type, d, ff, C.slots + C.off_up, b_.xn, &b_.qxn, b_.eu}, (int64_t)C.slot_bytes, 0, ff);
    if (!g_.share_wait_) shared_matvecs(S, F);
    pb_.mv_end("moe.gate_up");
    if (g_.cpu_mode_) pb_.program().ops->back().f[3] = HOST_WAIT_MS;  // reads pinned experts over PCIe: rare outliers
}

void Qwen35MoeProgram::expert_down(const MoeWeights& S, const LayerCache& C, PfLayer& F) {
    const int d = dm_.d, ff = dm_.ff;
    pb_.op(OP_ACT_QUANT, {b_.eg, b_.eu, b_.act, b_.q_act.q, b_.q_act.meta}, {b_.act_n, ff});
    pb_.mv_begin();
    pb_.group({S.down_exps->type, ff, d, C.slots + C.off_down, b_.act, &b_.q_act, b_.ed}, (int64_t)C.slot_bytes, ff, d);
    const Q8Vec xs{b_.q_act.q + UMAX * ff, b_.q_act.meta + UMAX * ff / 32};
    F.moe.sh_down = pb_.pw(*S.shexp_down, g_.upload(*S.shexp_down));
    pb_.addw(*S.shexp_down, F.moe.sh_down.w, b_.act + UMAX * ff, &xs, b_.sh_d);
    pb_.program().tasks->back().wide = 0;  // x only as int8 (moe.act)
    pb_.mv_end("moe.down");
    if (g_.cpu_mode_) pb_.program().ops->back().f[3] = HOST_WAIT_MS;
}

void Qwen35MoeProgram::experts(uint32_t l) {
    const MoeWeights& S = m_.layers()[l].moe;
    const LayerCache& C = g_.cache_[l];
    PfLayer& F = g_.pf_[l];
    expert_gate_up(l, S, F);
    expert_down(S, C, F);
    if (g_.cpu_mode_) {
        pb_.op(OP_HOST_Y, {g_.mbox_dev_}, {(int)l});
        pb_.program().ops->back().f[3] = HOST_WAIT_MS + 200;
    }
    pb_.op(OP_MOE_SUM,
           {b_.ed, b_.sh_d, b_.sh_inp, b_.y, g_.cpu_mode_ ? (void*)(g_.cpu_io_dev_ + g_.cpu_y_off_) : nullptr},
           {UMAX, dm_.d});
}

}  // namespace tona::gpu
