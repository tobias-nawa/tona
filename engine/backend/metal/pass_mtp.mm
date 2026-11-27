// The MTP head as a pass (engine/gpu/mega_build_mtp.cpp, same order): per (entry, stream) x = [enorm(e) | hnorm(h)],
// eh_proj into the head's own residual streams, then a QSA block with its own KV and an MoE like a layer of the
// target, its own gated read and the truncated LM head of the target. The head runs as a decode pass of up to BMAX
// entries; it swaps its residual streams in, so R of the target (its h for the next fill) stays untouched.
#include <algorithm>
#include <utility>

#include "backend/metal/kernels/params.h"
#include "backend/metal/pass.h"

namespace tona::metal {

namespace {

MTLSize n1(uint64_t n) { return MTLSizeMake(n, 1, 1); }

}  // namespace

void Pass::mtp_init(const MtpHead& m) {
    const uint64_t hcd = (uint64_t)h_.n_hc * h_.d * std::max(BMAX, mtp_part());
    if (m.n_used != h_.n_expert_used) throw std::runtime_error("Metal: the MTP head routes to another expert count");
    if (m.rope_base != h_.rope_base) throw std::runtime_error("Metal: the MTP head has another RoPE base");
    mb_.e = dev_.floats(hcd);
    mb_.hin = dev_.floats(hcd);
    mb_.x = dev_.floats(2 * hcd);
    mb_.R = dev_.floats(hcd);
}

void Pass::mtp(id<MTLComputeCommandEncoder> enc, SeqState& st, const MtpPassIn& in) {
    const MtpHead& M = *mtp_;
    const Layer& S = M.L;
    const uint32_t D = h_.d, hc = h_.n_hc, rows = hc * (uint32_t)in.nb;
    if (in.nb < 1 || in.nb > (in.prefill ? mtp_part() : BMAX) || in.pos + in.nb > max_ctx_ ||
        (in.prefill && !in.kv_only))
        throw std::logic_error("Metal: MTP pass out of range");
    gemm_ = in.prefill;
    write_positions(in.pos, in.nb);
    Dispatch d{enc, dev_};
    std::swap(s_.R, mb_.R);
    // The norm of an h of zeros is zeros: an entry without a known h gets [enorm(e) | 0], as in the reference
    d.run("rms_rows", {mb_.e, mb_.x, w_.vec(M.enorm)}, NormParams{D, D, 2 * D, 0, 1, 1, h_.rms_eps, 1.0f}, n1(rows),
          n1(256));
    d.run("rms_rows", {mb_.hin, mb_.x.at((uint64_t)D * 4), w_.vec(M.hnorm)},
          NormParams{D, D, 2 * D, D, 1, hc, h_.rms_eps, 1.0f}, n1(rows), n1(256));
    mv(d, w_.mat(M.eh_proj), mb_.x, s_.R, (int)rows);
    hc_mix(d, S.gr_attn, true, in.nb);
    if (in.kv_only) {
        qsa_kv(d, S, mb_.slot, st, in.pos, in.nb);
        std::swap(s_.R, mb_.R);
        return;
    }
    const std::vector<int32_t> none;
    qsa(d, S, mb_.slot, st, PassIn{nullptr, in.nb, in.pos, &none, Buf{}, false, false});
    hc_combine(d, in.nb);
    hc_mix(d, S.gr_ffn, true, in.nb);
    moe(d, S, in.nb);
    hc_combine(d, in.nb);
    hc_mix(d, M.gr_head, false, in.nb);
    mv(d, mb_.head, s_.mixed.at((uint64_t)(in.nb - 1) * D * 4), mb_.logits, 1);
    if (mb_.pick_out.b) {
        const uint32_t rows = mb_.head.rows, per = std::max<uint32_t>(1024, (rows + 255) / 256);
        const PickParams p{rows, per, (rows + per - 1) / per};
        d.run("mtp_pick", {mb_.logits, mb_.rows, mb_.pick_part, mb_.pick_out, mb_.pick_done}, p, n1(p.groups), n1(256));
    }
    std::swap(s_.R, mb_.R);
}

}  // namespace tona::metal
