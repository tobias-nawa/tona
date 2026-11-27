// The latent MoE block of Bolt in the decode program. One block, in the order of the ops:
//   mv router + lat_down   (both read the block's normed input; the router on the full width)
//   quant.lat              latent input to int8 for the expert matvecs
//   host.wait              the CPU ranks the router logits (sigmoid policy of the layer) and picks the union; in CPU
//   mode
//                          it also receives the int8 latent input and computes the misses it keeps (squared-ReLU
//                          experts)
//   mv shared up           (while block 0 waits: share_wait_) the shared expert's first half, full width, no gate
//   moe.copy               misses into their slots
//   mv experts up          ungated, in the latent space, UMAX places at once
//   moe.relu2              relu(u)^2 of the places and the shared expert, int8
//   mv experts down + shared down
//   host.y                 CPU mode: wait for the CPU's experts
//   moe.sumlat             fixed-point sum of the weighted outputs (GPU and CPU) in the latent space, float and int8
//   mv lat_up              back to the full width
//   moe.add                y = up + shared
// The reference is NemotronHOracle::moe (cpu/ref_nemotron_h.cpp), llama.cpp b11822 src/models/nemotron-h-moe.cpp.
#include "gpu/arch/nemotron_h/moe.h"

#include <algorithm>
#include <format>
#include <stdexcept>

#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

LatentGeom latent_geom(const NemotronHMoeModel& m, uint32_t block) {
    const NemotronLayer& L = m.layers()[block];
    LatentGeom G{};
    G.d = (int)m.hp().d, G.lat = (int)(m.hp().latent ? m.hp().latent : m.hp().d);
    G.experts = (int)L.n_expert, G.k = (int)L.n_used, G.ff = (int)L.expert_ff, G.sff = (int)m.hp().shexp_ff;
    G.act_n = UMAX * G.ff + G.sff;
    return G;
}

LatentMax latent_max(const NemotronHMoeModel& m) {
    LatentMax x;
    for (uint32_t i = 0; i < m.count(BoltBlockKind::Moe); i++) {
        const LatentGeom G = latent_geom(m, m.moe_layer(i));
        x.lat = std::max(x.lat, G.lat), x.ff = std::max(x.ff, G.ff), x.sff = std::max(x.sff, G.sff);
        x.k = std::max(x.k, G.k), x.act_n = std::max(x.act_n, G.act_n);
    }
    return x;
}

void latent_check_shapes(const NemotronHMoeModel& m) {
    const LatentMax x = latent_max(m);
    const bool ok = x.lat % 128 == 0 && x.ff % 32 == 0 && x.sff % 32 == 0 && m.hp().d % 128 == 0 &&
                    x.k * BMAX <= UMAX && m.moe_shape().n_expert <= (uint32_t)MEGA_MAX_EXPERTS_WIDE;
    if (!ok) throw std::runtime_error("gpu::Mega: shapes of the latent MoE do not match the fixed kernel assumptions");
}

LatentMoe::LatentMoe(ProgramBuilder& pb, const NemotronHMoeModel& m)
    : pb_(pb), g_(pb.mega()), m_(m), mx_(latent_max(m)) {}

// Buffers of all blocks are sized for the widest one; the ops of a block use the distance between the tokens of the
// buffer, not its own width (OP_ACT_RELU2 takes it as i[2])
void LatentMoe::alloc_buffers() {
    const int d = (int)m_.hp().d;
    b_.router_out = g_.alloc_b(MEGA_MAX_EXPERTS_WIDE);
    b_.xl = g_.alloc_b(mx_.lat);
    b_.qxl = g_.alloc_q8_b(mx_.lat);
    b_.eu = g_.alloc_b(mx_.act_n);
    b_.act = g_.alloc_b(mx_.act_n);
    b_.q_act = g_.alloc_q8_b(mx_.act_n);
    b_.ed = g_.alloc_b((size_t)UMAX * mx_.lat);
    b_.sh_d = g_.alloc_b(d);
    b_.yl = g_.alloc_b(mx_.lat);
    b_.q_yl = g_.alloc_q8_b(mx_.lat);
    b_.y_up = g_.alloc_b(d);
    router_dev_.assign(m_.count(BoltBlockKind::Moe), nullptr);
}

// The selection bias is read by the host every pass. The file holds it as F32, but its mapping is released after the
// upload (it is no expert tensor), so the policy points into a copy that lives as long as the backend.
void LatentMoe::route_policy(uint32_t i, const NemotronLayer& L) {
    const TensorInfo& t = *L.moe.router_bias;
    if (t.type != 0 || !t.data || t.elements() != L.n_expert)
        throw std::runtime_error(std::format("gpu::Mega: {} is not {} F32 values in the file", t.name, L.n_expert));
    const float* src = reinterpret_cast<const float*>(t.data);
    g_.route_bias_.push_back(std::make_unique<std::vector<float>>(src, src + L.n_expert));
    g_.cache_[i].route = {backend::Gate::Sigmoid, m_.hp().expert_scale, g_.route_bias_.back()->data()};
}

const void* LatentMoe::router_weights(const TensorInfo& t, uint32_t i) {
    if (!router_dev_[i]) router_dev_[i] = g_.upload(t);
    return router_dev_[i];
}

// Router and the latent projection read the same input; the CPU needs the router logits and the experts the int8 latent
void LatentMoe::router_and_latent(uint32_t i, const LatentGeom& G, const MoeWeights& S, const Io& io) {
    PfMoe& F = g_.pf_[m_.moe_layer(i)].moe;
    const void* w_down = g_.upload(*S.latent_down);
    F.router = pb_.pw(*S.router, router_weights(*S.router, i));
    F.lat_down = pb_.pw(*S.latent_down, w_down);
    pb_.mv_begin();
    pb_.addw(*S.router, router_weights(*S.router, i), io.xn, io.qxn, b_.router_out);
    pb_.addw(*S.latent_down, w_down, io.xn, io.qxn, b_.xl);
    pb_.mv_end("router");
    pb_.op(OP_QUANT_LAT, {b_.xl, b_.qxl.q, b_.qxl.meta}, {G.lat});
    if (g_.cpu_mode_)
        pb_.op(OP_HOST_WAIT,
               {g_.mbox_dev_, b_.router_out, b_.qxl.q, b_.qxl.meta, g_.cpu_io_dev_, g_.cpu_io_dev_ + g_.cpu_xm_off_,
                nullptr},
               {(int)i, G.lat, G.experts});
    else
        pb_.op(OP_HOST_WAIT, {g_.mbox_dev_, b_.router_out}, {(int)i, 0, G.experts});
    pb_.program().ops->back().f[3] = HOST_WAIT_MS + 200;  // the other blocks wait with it while the CPU reads
}

// The shared expert's first half does not depend on the routing: with share_wait_ it runs while block 0 waits for the
// CPU (host.wait without a barrier after it), else it joins the experts' up phase
void LatentMoe::shared_up(uint32_t i, const MoeWeights& S, const LatentGeom& G, const Io& io) {
    PfMoe& F = g_.pf_[m_.moe_layer(i)].moe;
    F.sh_up = pb_.pw(*S.shexp_up, g_.upload(*S.shexp_up));
    pb_.addw(*S.shexp_up, F.sh_up.w, io.xn, io.qxn, b_.eu + (size_t)UMAX * G.ff);
}

void LatentMoe::experts(uint32_t i, const LatentGeom& G, const MoeWeights& S, const Io& io) {
    const LayerCache& C = g_.cache_[i];
    PfMoe& F = g_.pf_[m_.moe_layer(i)].moe;
    const int64_t sb = (int64_t)C.slot_bytes;
    if (g_.share_wait_) {
        pb_.program().ops->back().nobar = 1;  // host.wait
        pb_.mv_begin();
        shared_up(i, S, G, io);
        pb_.mv_end("moe.shared");
        pb_.program().ops->back().i[7] = 1;
        pb_.program().ops->back().f[3] = HOST_WAIT_MS + 200;  // the barrier behind it waits for the CPU too
    }
    pb_.op(OP_EXPERT_COPY, {g_.cpu_mode_ ? g_.fill_dev_ + i * g_.fill_stride_ : g_.staging_dev_, C.slots},
           {(int)C.slot_bytes, g_.cpu_mode_ ? (int)(g_.fill_stride_ / 2 / C.slot_bytes) : 0});
    pb_.mv_begin();
    pb_.group({S.up_exps->type, G.lat, G.ff, C.slots + C.off_up, b_.xl, &b_.qxl, b_.eu}, sb, 0, G.ff);
    if (!g_.share_wait_) shared_up(i, S, G, io);
    pb_.mv_end("moe.up");
    if (g_.cpu_mode_) pb_.program().ops->back().f[3] = HOST_WAIT_MS;  // reads pinned experts over PCIe: rare outliers
    pb_.op(OP_ACT_RELU2, {b_.eu, b_.act, b_.q_act.q, b_.q_act.meta}, {G.act_n, G.ff, mx_.act_n});
    pb_.mv_begin();
    pb_.group({S.down_exps->type, G.ff, G.lat, C.slots + C.off_down, b_.act, &b_.q_act, b_.ed}, sb, G.ff, G.lat);
    const Q8Vec xs{b_.q_act.q + (size_t)UMAX * G.ff, b_.q_act.meta + (size_t)UMAX * G.ff / 32};
    F.sh_down = pb_.pw(*S.shexp_down, g_.upload(*S.shexp_down));
    pb_.addw(*S.shexp_down, F.sh_down.w, b_.act + (size_t)UMAX * G.ff, &xs, b_.sh_d);
    pb_.program().tasks->back().wide = 0;  // x only as int8 (moe.relu2)
    pb_.mv_end("moe.down");
    if (g_.cpu_mode_) {
        pb_.program().ops->back().f[3] = HOST_WAIT_MS;
        pb_.op(OP_HOST_Y, {g_.mbox_dev_}, {(int)i});
        pb_.program().ops->back().f[3] = HOST_WAIT_MS + 200;
    }
    pb_.op(OP_MOE_SUM_LAT,
           {b_.ed, b_.yl, b_.q_yl.q, b_.q_yl.meta, g_.cpu_mode_ ? (void*)(g_.cpu_io_dev_ + g_.cpu_y_off_) : nullptr},
           {UMAX, G.lat});
    const void* w_up = g_.upload(*S.latent_up);
    F.lat_up = pb_.pw(*S.latent_up, w_up);
    pb_.mv_begin();
    pb_.addw(*S.latent_up, w_up, b_.yl, &b_.q_yl, b_.y_up);
    pb_.mv_end("moe.latup");
    pb_.op(OP_MOE_ADD, {b_.y_up, b_.sh_d, io.y}, {G.d});
}

// What the prefill needs besides the weights: the selection bias on the device and the scale of the weights
void LatentMoe::record_prefill(uint32_t i, const NemotronLayer& L) {
    PfMoe& F = g_.pf_[m_.moe_layer(i)].moe;
    F.router_bias = g_.upload_f32(*L.moe.router_bias);
    F.route_scale = m_.hp().expert_scale;
}

void LatentMoe::build(uint32_t i, const Io& io) {
    const uint32_t block = m_.moe_layer(i);
    const NemotronLayer& L = m_.layers()[block];
    const LatentGeom G = latent_geom(m_, block);
    route_policy(i, L);
    record_prefill(i, L);
    router_and_latent(i, G, L.moe, io);
    experts(i, G, L.moe, io);
}

}  // namespace tona::gpu
