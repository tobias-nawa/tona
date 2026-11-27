// MoE steps of a layer: the union of the experts of all tokens (UMAX places, a mask per token), the shared expert
// and the PLE projection.
#include "backend/env.h"
#include "gpu/arch/qwen4exp/program.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

// The shared expert (gate, up, weight) and PLE do not depend on the routing
void Qwen4ExpProgram::shared_matvecs(const Layer& S, PfLayer& F, bool ple) {
    pb_.addw(*S.shexp_gate, F.moe.sh_gate.w, b_.mixed, &b_.qmix, b_.eg + UMAX * dm_.ff);
    pb_.addw(*S.shexp_up, F.moe.sh_up.w, b_.mixed, &b_.qmix, b_.eu + UMAX * dm_.ff);
    pb_.addw(*S.shexp_gate_inp, F.moe.sh_inp.w, b_.mixed, &b_.qmix, b_.sh_inp);
    if (!ple) return;
    const Layer& P = m_.layers()[h_.ple_layer];
    pb_.addw(*P.ple_key, g_.pf_ple_key_.w, g_.e_ + dm_.d, nullptr, b_.ple_k);
    pb_.addw(*P.ple_value, g_.pf_ple_val_.w, g_.e_ + dm_.d, nullptr, b_.ple_v);
}

// With share_wait_ the other blocks compute the shared expert while block 0 waits for the CPU (host.wait without a
// barrier after it); otherwise it sits in moe.gate_up
void Qwen4ExpProgram::shared(uint32_t l, bool ple) {
    const Layer& S = m_.layers()[l];
    PfLayer& F = g_.pf_[l];
    F.moe.sh_gate = pb_.pw(*S.shexp_gate, g_.upload(*S.shexp_gate));
    F.moe.sh_up = pb_.pw(*S.shexp_up, g_.upload(*S.shexp_up));
    F.moe.sh_inp = pb_.pw(*S.shexp_gate_inp, g_.upload(*S.shexp_gate_inp));
    if (ple) {
        const Layer& P = m_.layers()[h_.ple_layer];
        g_.pf_ple_key_ = pb_.pw(*P.ple_key, g_.upload(*P.ple_key));
        g_.pf_ple_val_ = pb_.pw(*P.ple_value, g_.upload(*P.ple_value));
    }
    if (!g_.share_wait_) return;
    pb_.program().ops->back().nobar = 1;  // host.wait
    pb_.mv_begin();
    shared_matvecs(S, F, ple);
    pb_.mv_end("moe.shared");
    pb_.program().ops->back().i[7] = 1;
    pb_.program().ops->back().f[3] = HOST_WAIT_MS + 200;  // the barrier behind it waits for the CPU too
}

void Qwen4ExpProgram::expert_gate_up(uint32_t l, const Layer& S, PfLayer& F, bool ple, int rep) {
    const LayerCache& C = g_.cache_[l];
    const int d = dm_.d, ff = dm_.ff;
    pb_.mv_begin();
    pb_.group({S.gate_exps->type, d, ff, C.slots, b_.mixed, &b_.qmix, b_.eg}, (int64_t)C.slot_bytes, 0, ff);
    pb_.group({S.up_exps->type, d, ff, C.slots + C.off_up, b_.mixed, &b_.qmix, b_.eu}, (int64_t)C.slot_bytes, 0, ff);
    if (!g_.share_wait_) shared_matvecs(S, F, ple);
    pb_.mv_end(rep ? "moe.gate_up.warm" : "moe.gate_up");
    if (g_.cpu_mode_)
        pb_.program().ops->back().f[3] = HOST_WAIT_MS;  // reads pinned experts over PCIe: rare long outliers
}

void Qwen4ExpProgram::expert_down(const Layer& S, const LayerCache& C, PfLayer& F, int rep) {
    const int d = dm_.d, ff = dm_.ff;
    if (rep == 0) pb_.op(OP_ACT_QUANT, {b_.eg, b_.eu, b_.act, b_.q_act.q, b_.q_act.meta}, {b_.act_n, ff});
    pb_.mv_begin();
    pb_.group({S.down_exps->type, ff, d, C.slots + C.off_down, b_.act, &b_.q_act, b_.ed}, (int64_t)C.slot_bytes, ff, d);
    const Q8Vec xs{b_.q_act.q + UMAX * ff, b_.q_act.meta + UMAX * ff / 32};
    if (rep == 0) F.moe.sh_down = pb_.pw(*S.shexp_down, g_.upload(*S.shexp_down));
    pb_.addw(*S.shexp_down, F.moe.sh_down.w, b_.act + UMAX * ff, &xs, b_.sh_d);
    pb_.program().tasks->back().wide = 0;  // x only as int8 (moe.act)
    pb_.mv_end(rep ? "moe.down.warm" : "moe.down");
    if (g_.cpu_mode_) pb_.program().ops->back().f[3] = HOST_WAIT_MS;
}

void Qwen4ExpProgram::experts(uint32_t l, bool ple) {
    const Layer& S = m_.layers()[l];
    const LayerCache& C = g_.cache_[l];
    PfLayer& F = g_.pf_[l];
    // TONA_WARM_TEST=1: gate_up and down twice each (upper bound for preloading into the Infinity Cache: the second run
    // reads warm); the output stays the same
    static const bool warm = env::flag("TONA_WARM_TEST");
    const int reps = warm ? 2 : 1;
    for (int rep = 0; rep < reps; rep++) expert_gate_up(l, S, F, ple, rep);
    for (int rep = 0; rep < reps; rep++) expert_down(S, C, F, rep);
    if (g_.cpu_mode_) {
        pb_.op(OP_HOST_Y, {g_.mbox_dev_}, {(int)l});
        pb_.program().ops->back().f[3] = HOST_WAIT_MS + 200;
    }
    pb_.op(OP_MOE_SUM,
           {b_.ed, b_.sh_d, b_.sh_inp, b_.y, g_.cpu_mode_ ? (void*)(g_.cpu_io_dev_ + g_.cpu_y_off_) : nullptr},
           {UMAX, dm_.d});
}

}  // namespace tona::gpu
