#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <numeric>
#include <stdexcept>

#include "backend/env.h"
#include "backend/nan_guard.h"
#include "gpu/forward.h"
#include "gpu/forward_util.h"
#include "gpu/matvec.h"
#include "gpu/ops.h"
#include "model/ple.h"
#include "model/quant.h"

namespace tona::gpu {

namespace {

int64_t row_bytes(uint32_t type, int64_t n) {
    const GgmlTypeInfo* ti = ggml_type_info(type);
    return n / ti->block_elems * ti->block_bytes;
}

std::vector<float> dequant_host(const TensorInfo& t, int64_t row = 0, int64_t n = -1) {
    if (n < 0) n = (int64_t)t.elements();
    std::vector<float> out(n);
    dequantize_row(t.type, t.data + row * row_bytes(t.type, n), out.data(), n);
    return out;
}

}  // namespace

float* Forward::alloc(size_t floats) {
    void* p;
    HIPCHECK(hipMalloc(&p, floats * sizeof(float)));
    HIPCHECK(hipMemset(p, 0, floats * sizeof(float)));
    allocs_.push_back(p);
    vram_ += floats * sizeof(float);
    return static_cast<float*>(p);
}

const void* Forward::upload(const TensorInfo& t) {
    void* p;
    HIPCHECK(hipMalloc(&p, t.bytes + kMatvecReadPad));
    HIPCHECK(hipMemcpy(p, t.data, t.bytes, hipMemcpyHostToDevice));
    allocs_.push_back(p);
    vram_ += t.bytes;
    return p;
}

const float* Forward::upload_f32(const TensorInfo& t) {
    const auto v = dequant_host(t);
    float* p = alloc(v.size());
    HIPCHECK(hipMemcpy(p, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice));
    return p;
}

Forward::GR Forward::upload_gr(const GatedResidual& g) {
    return GR{upload(*g.down), upload(*g.up), g.inject ? upload(*g.inject) : nullptr, upload_f32(*g.norm)};
}

void Forward::init_qsa(LayerDev& D, const Layer& S) {
    D.q = upload(*S.q), D.k = upload(*S.k), D.v = upload(*S.v), D.o = upload(*S.o);
    D.idx_q = upload(*S.idx_q), D.idx_k = upload(*S.idx_k);
    D.q_norm = upload_f32(*S.q_norm), D.k_norm = upload_f32(*S.k_norm);
    D.idx_q_norm = upload_f32(*S.idx_q_norm), D.idx_k_norm = upload_f32(*S.idx_k_norm);
    D.K = alloc((size_t)max_ctx_ * h_.n_head_kv * h_.head_dim);
    D.V = alloc((size_t)max_ctx_ * h_.n_head_kv * h_.head_dim);
    D.idx_raw = alloc((size_t)max_ctx_ * h_.idx_dim);
    D.idx_pool = alloc((size_t)(max_ctx_ / h_.idx_block + 1) * h_.idx_dim);
}

void Forward::init_deltanet(LayerDev& D, const Layer& S) {
    D.qkv = upload(*S.qkv), D.gate = upload(*S.gate), D.beta = upload(*S.beta), D.alpha = upload(*S.alpha);
    D.ssm_out = upload(*S.ssm_out);
    D.conv = upload_f32(*S.conv1d), D.dt = upload_f32(*S.dt_bias), D.A = upload_f32(*S.a);
    D.ssm_norm = upload_f32(*S.ssm_norm);
    D.dn_state = alloc((size_t)h_.dn_v_heads * h_.dn_head_dim * h_.dn_head_dim);
    D.dn_conv = alloc((size_t)(h_.dn_conv - 1) * (2 * h_.dn_k_dim() + h_.dn_v_dim()));
}

void Forward::init_ple(const Layer& S) {
    const int64_t hc_dim = (int64_t)h_.n_hc * h_.d;
    ple_key_ = upload(*S.ple_key), ple_val_ = upload(*S.ple_value);
    ple_wk_ = upload_f32(*S.ple_norm_key), ple_wq_ = upload_f32(*S.ple_norm_query);
    ple_wc_ = upload_f32(*S.ple_norm_conv), ple_conv_ = upload_f32(*S.ple_conv1d);
    ple_hist_ = alloc((size_t)(h_.ple_conv - 1) * h_.ple_ngram * hc_dim);
}

// Per slot gate | up | down of one expert, each part aligned to 256 bytes
void Forward::init_expert_cache(LayerDev& D, const Layer& S) {
    const int64_t d = h_.d, ff = h_.expert_ff;
    const size_t g = (size_t)(ff * row_bytes(S.gate_exps->type, d)), u = (size_t)(ff * row_bytes(S.up_exps->type, d));
    const size_t dn = (size_t)(d * row_bytes(S.down_exps->type, ff));
    D.off_up = (g + 255) / 256 * 256;
    D.off_down = D.off_up + (u + 255) / 256 * 256;
    D.slot_bytes = D.off_down + (dn + 255) / 256 * 256;
    void* p;
    HIPCHECK(hipMalloc(&p, D.slot_bytes * slots_ + kMatvecReadPad));
    allocs_.push_back(p);
    vram_ += D.slot_bytes * slots_;
    D.slots = static_cast<uint8_t*>(p);
    for (int s = slots_ - 1; s >= 0; s--) D.free_slots.push_back(s);
}

void Forward::init_layer(uint32_t l) {
    const Layer& S = m_.layers()[l];
    LayerDev& D = L_[l];
    D.gr_attn = upload_gr(S.gr_attn);
    D.gr_ffn = upload_gr(S.gr_ffn);
    if (h_.is_qsa(l))
        init_qsa(D, S);
    else
        init_deltanet(D, S);
    D.router = upload(*S.router);
    D.sh_gate = upload(*S.shexp_gate), D.sh_up = upload(*S.shexp_up), D.sh_down = upload(*S.shexp_down);
    D.sh_inp = upload(*S.shexp_gate_inp);
    if (l == h_.ple_layer) init_ple(S);
    init_expert_cache(D, S);
}

void Forward::init_buffers() {
    const int64_t d = h_.d, hc_dim = (int64_t)h_.n_hc * d;
    const size_t big = std::max<size_t>(hc_dim, 2 * (size_t)h_.n_head * h_.head_dim);
    R_ = alloc(hc_dim), xn_ = alloc(hc_dim), lo_ = alloc(h_.hc_rank), gate_ = alloc(hc_dim), mixed_ = alloc(d);
    inject_ = alloc(h_.n_hc), y_ = alloc(d), e_ = alloc(d);
    t1_ = alloc(big), t2_ = alloc(big), t3_ = alloc(big), t4_ = alloc(big), t5_ = alloc(big), t6_ = alloc(2 * hc_dim);
    logits_ = alloc(h_.vocab);
    void* p;
    HIPCHECK(hipMalloc(&p, 4096 * sizeof(int32_t)));
    allocs_.push_back(p);
    cells_ = static_cast<int32_t*>(p);
    logits_host_.resize(h_.vocab);
    use_q8_ = env::flag("TONA_GPU_INT8");
    auto q8buf = [&](Q8Vec& v, size_t n) {
        v.q = reinterpret_cast<int8_t*>(alloc(n / 4));
        v.meta = reinterpret_cast<Q8Meta*>(alloc(n / 32 * sizeof(Q8Meta) / 4));
    };
    q8buf(qmix_, d);
    q8buf(qtmp_, big);
}

Forward::Forward(const Qwen4ExpModel& m, int expert_slots, int64_t max_ctx)
    : m_(m), h_(m.hp()), max_ctx_(max_ctx), slots_(std::min<int>(expert_slots, m.hp().n_expert)) {
    ops::set_yarn(rope::yarn_active(h_.rope_base));
    ops::check_yarn(rope::yarn_active(h_.rope_base));
    L_.resize(h_.n_layer);
    for (uint32_t l = 0; l < h_.n_layer; l++) init_layer(l);
    head_ = upload_gr(m_.gr_head());
    output_ = upload(*m_.output());
    init_buffers();
    // Room for the misses of one layer: at most n_expert_used experts
    size_t max_slot = 0;
    for (const auto& D : L_) max_slot = std::max(max_slot, D.slot_bytes);
    staging_bytes_ = max_slot * h_.n_expert_used;
    HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&staging_), staging_bytes_, hipHostMallocDefault));
    HIPCHECK(hipStreamCreateWithFlags(&copy_stream_, hipStreamNonBlocking));
    HIPCHECK(hipEventCreateWithFlags(&copied_, hipEventDisableTiming));
    router_host_.resize(h_.n_expert);
}

Forward::~Forward() {
    if (copy_stream_) (void)hipStreamSynchronize(copy_stream_);
    for (void* p : allocs_) (void)hipFree(p);
    if (staging_) (void)hipHostFree(staging_);
    if (copied_) (void)hipEventDestroy(copied_);
    if (copy_stream_) (void)hipStreamDestroy(copy_stream_);
}

void Forward::mv(const TensorInfo& t, const void* w, const float* x, float* y, const Q8Vec* xq) {
    const int64_t in = (int64_t)t.dims[0];
    run_matvec(MatvecTask{t.type, in, (int64_t)t.elements() / in, w, x, y}, xq);
}

const Q8Vec* Forward::quant(const float* x, int64_t n, const Q8Vec& out) const {
    if (!use_q8_) return nullptr;
    quantize_q8(x, n, out);
    return &out;
}

void Forward::gr_mix(const GR& g, const TensorInfo& down, const TensorInfo& up, const TensorInfo* inject, float* mixed,
                     float* inject_out) {
    const int hc = (int)h_.n_hc, d = (int)h_.d;
    ops::rms_norm_groups({R_, g.norm, xn_, hc, d, h_.rms_eps});
    mv(down, g.down, xn_, lo_);
    ops::silu_scale(lo_, (int)h_.hc_rank, 1.0f / hc);
    mv(up, g.up, lo_, gate_);
    ops::hc_mix(xn_, gate_, mixed, hc, d);
    if (inject) mv(*inject, g.inject, xn_, inject_out);
}

void Forward::deltanet(uint32_t l, float* y) {
    const Layer& S = m_.layers()[l];
    LayerDev& D = L_[l];
    const int kd = (int)h_.dn_k_dim(), vd = (int)h_.dn_v_dim(), ch = 2 * kd + vd, Sd = (int)h_.dn_head_dim;
    const Q8Vec* xq = quant(mixed_, h_.d, qmix_);
    mv(*S.qkv, D.qkv, mixed_, t1_, xq);
    mv(*S.gate, D.gate, mixed_, t2_, xq);
    mv(*S.beta, D.beta, mixed_, t3_, xq);
    mv(*S.alpha, D.alpha, mixed_, t4_, xq);
    ops::dn_conv({t1_, D.dn_conv, D.conv, t5_, ch, (int)h_.dn_conv, pos_});
    ops::l2norm_heads(t5_, (int)h_.dn_k_heads, Sd, h_.rms_eps);
    ops::l2norm_heads(t5_ + kd, (int)h_.dn_k_heads, Sd, h_.rms_eps);
    ops::dn_recur({t5_, t5_ + kd, t5_ + 2 * kd, t3_, t4_, D.dt, D.A, D.ssm_norm, t2_, D.dn_state, t6_,
                   (int)h_.dn_k_heads, (int)h_.dn_v_heads, Sd, h_.rms_eps});
    mv(*S.ssm_out, D.ssm_out, t6_, y, quant(t6_, vd, qtmp_));
}

void Forward::qsa(uint32_t l, float* y) {
    const Layer& S = m_.layers()[l];
    LayerDev& D = L_[l];
    const int nh = (int)h_.n_head, nkv = (int)h_.n_head_kv, hd = (int)h_.head_dim, idim = (int)h_.idx_dim;
    const int blk = (int)h_.idx_block, n_rot = (int)h_.rope_dims;
    if (pos_ >= max_ctx_) throw std::runtime_error("gpu::Forward: context full");

    const Q8Vec* xq = quant(mixed_, h_.d, qmix_);
    mv(*S.idx_k, D.idx_k, mixed_, D.idx_raw + pos_ * idim, xq);
    if ((pos_ + 1) % blk == 0)
        ops::idx_pool({D.idx_raw, pos_ + 1 - blk, blk, idim, D.idx_k_norm, n_rot, h_.rope_base, h_.rms_eps,
                       D.idx_pool + (pos_ / blk) * idim});
    const QsaCells qc = qsa_cells(pos_, blk, h_.idx_top_k);
    const int64_t n_pool = qc.n_pool, n_top = qc.top_blocks;
    const int32_t* cells = nullptr;
    int n_cells = (int)(pos_ + 1);
    if (n_pool > n_top) {
        // Selection needed: fetch the scores and take the top k on the CPU as cpu/ref.cpp does
        mv(*S.idx_q, D.idx_q, mixed_, t1_, xq);
        ops::head_norm_rope({t1_, D.idx_q_norm, (int)h_.idx_heads, idim, n_rot, pos_, h_.rope_base, h_.rms_eps});
        ops::idx_score({t1_, D.idx_pool, t6_, (int)h_.idx_heads, idim, n_pool});
        n_cells = select_cells(n_pool, n_top);
        cells = cells_;
    }

    mv(*S.q, D.q, mixed_, t2_, xq);
    ops::qsa_split(t2_, t3_, t4_, nh, hd);
    ops::head_norm_rope({t3_, D.q_norm, nh, hd, n_rot, pos_, h_.rope_base, h_.rms_eps});
    float* Kp = D.K + pos_ * nkv * hd;
    mv(*S.k, D.k, mixed_, Kp, xq);
    mv(*S.v, D.v, mixed_, D.V + pos_ * nkv * hd, xq);
    ops::head_norm_rope({Kp, D.k_norm, nkv, hd, n_rot, pos_, h_.rope_base, h_.rms_eps});
    ops::attention({t3_, D.K, D.V, cells, n_cells, t5_, nh, nkv, hd, 1.0f / std::sqrt((float)hd)});
    ops::mul_sigmoid(t5_, t4_, nh * hd);
    mv(*S.o, D.o, t5_, y, quant(t5_, nh * hd, qtmp_));
}

// Top n_top blocks by score (ties to the lower block) plus the cells after the last full block, uploaded to cells_;
// returns the number of cells. The scores are in t6_.
int Forward::select_cells(int64_t n_pool, int64_t n_top) {
    const int blk = (int)h_.idx_block;
    scores_host_.resize(n_pool);
    HIPCHECK(hipMemcpy(scores_host_.data(), t6_, n_pool * sizeof(float), hipMemcpyDeviceToHost));
    tona::nan::report("block scores (Forward)", tona::nan::clean(scores_host_.data(), (size_t)n_pool),
                      std::format("position {}", pos_));
    std::vector<int64_t> idx(n_pool);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + n_top, idx.end(), [&](int64_t a, int64_t b) {
        return scores_host_[a] > scores_host_[b] || (scores_host_[a] == scores_host_[b] && a < b);
    });
    std::vector<int32_t> c;
    for (int64_t t = 0; t < n_top; t++)
        for (int k = 0; k < blk; k++) c.push_back((int32_t)(idx[t] * blk + k));
    for (int64_t p = n_pool * blk; p <= pos_; p++) c.push_back((int32_t)p);
    HIPCHECK(hipMemcpy(cells_, c.data(), c.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    return (int)c.size();
}

void Forward::ple() {
    const int64_t d = h_.d;
    std::vector<int64_t> rows(h_.ple_heads);
    tona::ple::rows_at(h_, tokens_.data(), (int64_t)tokens_.size() - 1,
                       rows.data());  // tokens_ already holds the token
    std::vector<float> e(d);
    for (uint32_t head = 0; head < h_.ple_heads; head++) {
        const auto r = dequant_host(*m_.ple_embd(), rows[head], h_.ple_dim);
        std::copy(r.begin(), r.end(), e.begin() + head * h_.ple_dim);
    }
    HIPCHECK(hipMemcpy(e_, e.data(), d * sizeof(float), hipMemcpyHostToDevice));
    const Layer& S = m_.layers()[h_.ple_layer];
    mv(*S.ple_key, ple_key_, e_, t1_);
    mv(*S.ple_value, ple_val_, e_, t2_);
    ops::ple_apply({R_, t1_, t2_, ple_wk_, ple_wq_, ple_wc_, ple_conv_, ple_hist_, t6_, (int)h_.n_hc, (int)d,
                    (int)h_.ple_conv, (int)h_.ple_ngram, pos_, h_.rms_eps});
}

const std::vector<float>& Forward::step(int32_t token) {
    if (token < 0 || token >= (int32_t)h_.vocab) throw std::runtime_error("gpu::step: token outside the vocabulary");
    const int hc = (int)h_.n_hc, d = (int)h_.d;
    tokens_.push_back(token);
    const auto emb = dequant_host(*m_.tok_embd(), token, d);
    HIPCHECK(hipMemcpy(e_, emb.data(), d * sizeof(float), hipMemcpyHostToDevice));
    ops::broadcast_rows(R_, e_, hc, d);

    for (uint32_t l = 0; l < h_.n_layer; l++) {
        const Layer& S = m_.layers()[l];
        if (l == h_.ple_layer) ple();
        gr_mix(L_[l].gr_attn, *S.gr_attn.down, *S.gr_attn.up, S.gr_attn.inject, mixed_, inject_);
        if (h_.is_qsa(l))
            qsa(l, y_);
        else
            deltanet(l, y_);
        ops::hc_combine(R_, y_, inject_, hc, d);
        gr_mix(L_[l].gr_ffn, *S.gr_ffn.down, *S.gr_ffn.up, S.gr_ffn.inject, mixed_, inject_);
        moe(l, y_);
        ops::hc_combine(R_, y_, inject_, hc, d);
    }
    gr_mix(head_, *m_.gr_head().down, *m_.gr_head().up, nullptr, mixed_, nullptr);
    mv(*m_.output(), output_, mixed_, logits_, quant(mixed_, d, qmix_));
    HIPCHECK(hipMemcpy(logits_host_.data(), logits_, logits_host_.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipGetLastError());
    pos_++;
    return logits_host_;
}

}  // namespace tona::gpu
