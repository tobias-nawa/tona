// The drafter (DFlash2, cpu/dflash) as a second program for the same kernel. Feeding (steps with cond 1, over the m
// accepted tokens): features -> fc -> norm -> K, V per layer. Then the block at pos.
#include <cmath>
#include <string_view>

#include "backend/env.h"
#include "gpu/arch/qwen4exp/program.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

namespace {

const TensorInfo& draft_tensor(const Gguf& G, const std::string& name) {
    const TensorInfo* p = G.find(name);
    if (!p) throw std::runtime_error("gpu::Mega: drafter tensor missing: " + name);
    return *p;
}

const TensorInfo& block_tensor(const Gguf& G, int l, const char* name) {
    return draft_tensor(G, std::format("blk.{}.{}", l, name));
}

}  // namespace

Qwen4ExpProgram::DraftDims Qwen4ExpProgram::draft_dims(const Gguf& G) {
    DraftDims D{};
    D.nh = (int)G.get_uint("dflash.attention.head_count"), D.nkv = (int)G.get_uint("dflash.attention.head_count_kv");
    D.hd = (int)G.get_uint("dflash.attention.key_length"), D.ff = (int)G.get_uint("dflash.feed_forward_length");
    D.layers = (int)G.get_uint("dflash.block_count"), D.group = (int)G.get_uint("dflash.conv_group_size");
    D.swa = G.has("dflash.attention.sliding_window") ? (int)G.get_uint("dflash.attention.sliding_window") : 0;
    g_.dswa_ = D.swa;
    g_.drank_ = (int)G.get_uint("dflash.selector_rank");
    D.eps = (float)G.get_float("dflash.attention.layer_norm_rms_epsilon");
    D.base = (float)G.get_float("dflash.rope.freq_base");
    if (D.hd != 128 || G.get_uint("dflash.conv_kernel_size") != 2 ||
        (int)G.get_uint("dflash.embedding_length") != dm_.d ||
        (D.swa > 0 ? D.swa + MEGA_BMAX : g_.max_ctx_) > MAX_CELLS - 512)
        throw std::runtime_error("gpu::Mega: the drafter does not match the fixed kernel assumptions");
    D.qd = D.nh * D.hd, D.kvd = D.nkv * D.hd, D.p4 = 4 * (dm_.d / D.group);
    // KV of the drafter as a ring: it sees only the window (2,048) plus the block; with 2 * BMAX buffers the block
    // overwrites no position that is still in the window
    D.ring = D.swa > 0 ? (int)std::min<int64_t>(D.swa + 2 * MEGA_BMAX, g_.max_ctx_) : (int)g_.max_ctx_;
    // TONA_ATTN=old: old drafter attention; split-K needs head dim 128
    const char* mode = env::text("TONA_ATTN");
    D.attn_old = (mode && std::string_view(mode) == "old") || D.hd != 128;
    return D;
}

void Qwen4ExpProgram::build_drafter() {
    const Gguf& G = *g_.drafter_;
    // Drafter: k_wide like the target (TONA_WIDE_DRAFT=0: off); block 8 takes 6.30 instead of 7.96 ms. Its drafts then
    // differ slightly from the CPU drafter, the target checks them anyway
    pb_.select_program({&g_.dops_, &g_.dtasks_, &g_.dlabels_}, env::integer("TONA_WIDE_DRAFT", 1) != 0);
    const DraftDims D = draft_dims(G);
    dattn_part_ = D.attn_old ? nullptr : g_.alloc((size_t)MEGA_BMAX * D.nh * ATT_SPL * (D.hd + 2));
    DraftLayers L;
    draft_inject(G, D, L);
    draft_block_buffers(D);
    for (int l = 0; l < D.layers; l++) draft_layer(G, D, L, l);
    draft_head(G);
    draft_finish();
}

void Qwen4ExpProgram::draft_inject(const Gguf& G, const DraftDims& D, DraftLayers& L) {
    const int d = dm_.d, nfeat = (int)g_.feat_layers_.size() * d;
    auto cond = [&] { pb_.program().ops->back().cond = 1; };
    Q8Vec qfeat = g_.alloc_q8_b(nfeat);
    float* dg = g_.alloc_b(d);
    float* dgn = g_.alloc_b(d);
    Q8Vec qdgn = g_.alloc_q8_b(d);
    pb_.op(OP_QUANT, {g_.feat_, qfeat.q, qfeat.meta}, {nfeat});
    cond();
    pb_.mv_begin();
    pb_.addw(draft_tensor(G, "fc.weight"), g_.upload(draft_tensor(G, "fc.weight")), g_.feat_, &qfeat, dg);
    pb_.mv_end("d.fc");
    cond();
    pb_.op(OP_NORM, {dg, (void*)g_.upload_f32(draft_tensor(G, "enc.output_norm.weight")), dgn, qdgn.q, qdgn.meta}, {d},
           {D.eps});
    cond();
    L.Kc.resize(D.layers), L.Vc.resize(D.layers), L.kin.resize(D.layers), L.vin.resize(D.layers);
    L.wk.resize(D.layers), L.wv.resize(D.layers), L.knorm.resize(D.layers);
    pb_.mv_begin();
    for (int l = 0; l < D.layers; l++) {
        // K/V in half when the split attention reads (half the bytes when reading and in the partials)
        const size_t kvb = (size_t)D.ring * D.kvd * (D.attn_old ? sizeof(float) : sizeof(_Float16));
        L.Kc[l] = g_.alloc(kvb / sizeof(float)), L.Vc[l] = g_.alloc(kvb / sizeof(float));
        g_.dring_bufs_.emplace_back(L.Kc[l], kvb);
        g_.dring_bufs_.emplace_back(L.Vc[l], kvb);
        L.kin[l] = g_.alloc_b(D.kvd), L.vin[l] = g_.alloc_b(D.kvd);
        L.wk[l] = g_.upload(block_tensor(G, l, "attn_k.weight")),
        L.wv[l] = g_.upload(block_tensor(G, l, "attn_v.weight"));
        L.knorm[l] = g_.upload_f32(block_tensor(G, l, "attn_k_norm.weight"));
        pb_.addw(block_tensor(G, l, "attn_k.weight"), L.wk[l], dgn, &qdgn, L.kin[l]);
        pb_.addw(block_tensor(G, l, "attn_v.weight"), L.wv[l], dgn, &qdgn, L.vin[l]);
    }
    pb_.mv_end("d.kv");
    cond();
    for (int l = 0; l < D.layers; l++) {
        pb_.op(OP_DQK, {nullptr, nullptr, L.kin[l], L.vin[l], (void*)L.knorm[l], L.Kc[l], L.Vc[l]},
               {D.nh, D.nkv, 1, D.ring, D.attn_old ? 0 : 1}, {D.eps, D.base});
        cond();
    }
    g_.dinject_end_ = (int)pb_.program().ops->size();
}

// Block: x = embeddings, then the residual
void Qwen4ExpProgram::draft_block_buffers(const DraftDims& D) {
    const int d = dm_.d;
    DraftBufs& B = db_;
    g_.e2_ = g_.alloc_b(d);
    B.x = g_.e2_;
    B.hn = g_.alloc_b(d);
    B.qhn = g_.alloc_q8_b(d);
    B.dyn = g_.alloc_b(D.p4);
    B.h2 = g_.alloc_b(d);
    B.qh2 = g_.alloc_q8_b(d);
    B.qb = g_.alloc_b(D.qd), B.kb = g_.alloc_b(D.kvd), B.vb = g_.alloc_b(D.kvd), B.ob = g_.alloc_b(D.qd);
    B.ab = g_.alloc_b(d);
    B.qob = g_.alloc_q8_b(D.qd);
    B.fg = g_.alloc_b(D.ff), B.fu = g_.alloc_b(D.ff), B.fa = g_.alloc_b(D.ff);
    B.qfa = g_.alloc_q8_b(D.ff);
}

void Qwen4ExpProgram::draft_layer(const Gguf& G, const DraftDims& D, const DraftLayers& L, int l) {
    const float* abase = g_.upload_f32(block_tensor(G, l, "attn_conv_base"));
    const float* fbase = g_.upload_f32(block_tensor(G, l, "ffn_conv_base"));
    draft_attention_half(G, D, L, l, abase);
    draft_ffn_half(G, D, l, fbase);
}

void Qwen4ExpProgram::draft_attention_half(const Gguf& G, const DraftDims& D, const DraftLayers& L, int l,
                                           const float* abase) {
    const int d = dm_.d;
    const DraftBufs& B = db_;
    auto bt = [&](const char* name) -> const TensorInfo& { return block_tensor(G, l, name); };
    pb_.op(OP_NORM, {B.x, (void*)g_.upload_f32(bt("attn_norm.weight")), B.hn, B.qhn.q, B.qhn.meta}, {d}, {D.eps});
    pb_.mv_begin();
    pb_.addw(bt("attn_conv_proj.weight"), g_.upload(bt("attn_conv_proj.weight")), B.hn, &B.qhn, B.dyn);
    pb_.mv_end("d.conv_proj");
    pb_.op(OP_DCONV, {B.hn, B.dyn, (void*)abase, B.h2, B.qh2.q, B.qh2.meta}, {d, D.group, 0, 0});
    pb_.mv_begin();
    pb_.addw(bt("attn_q.weight"), g_.upload(bt("attn_q.weight")), B.h2, &B.qh2, B.qb);
    pb_.addw(bt("attn_k.weight"), L.wk[l], B.h2, &B.qh2, B.kb);
    pb_.addw(bt("attn_v.weight"), L.wv[l], B.h2, &B.qh2, B.vb);
    pb_.mv_end("d.qkv");
    pb_.op(OP_DQK,
           {B.qb, (void*)g_.upload_f32(bt("attn_q_norm.weight")), B.kb, B.vb, (void*)L.knorm[l], L.Kc[l], L.Vc[l]},
           {D.nh, D.nkv, 0, D.ring, D.attn_old ? 0 : 1}, {D.eps, D.base});
    draft_attention_op(D, L, l);
    pb_.mv_begin();
    pb_.addw(bt("attn_output.weight"), g_.upload(bt("attn_output.weight")), B.ob, &B.qob, B.ab);
    pb_.mv_end("d.o");
    pb_.op(OP_DCONV, {B.ab, B.dyn, (void*)abase, B.x, nullptr, nullptr}, {d, D.group, 1, 1});
}

void Qwen4ExpProgram::draft_attention_op(const DraftDims& D, const DraftLayers& L, int l) {
    const DraftBufs& B = db_;
    const float scale = 1.0f / std::sqrt((float)D.hd);
    if (!dattn_part_) {
        pb_.op(OP_DATTN, {B.qb, L.Kc[l], L.Vc[l], B.ob, B.qob.q, B.qob.meta}, {D.nh, D.nkv, D.hd, D.swa, D.ring},
               {scale});
        return;
    }
    // WMMA with 4 heads per group (x 4 tokens = 16 rows); TONA_ATTN=wave or TONA_DATTN=split: a wave per (token, head,
    // chunk)
    static const bool split = [] {
        const char* v = env::text("TONA_DATTN");
        return v && std::string_view(v) == "split";
    }();
    const bool wmma = !attn_wave_ && !split && D.nh / D.nkv * MEGA_BMAX <= 16;
    pb_.op(wmma ? OP_DATTN_WMMA : OP_DATTN_SPLIT, {B.qb, L.Kc[l], L.Vc[l], B.ob, B.qob.q, B.qob.meta, dattn_part_},
           {D.nh, D.nkv, D.hd, D.swa, D.ring}, {scale});
    pb_.op(OP_DATTN_MERGE, {B.qb, L.Kc[l], L.Vc[l], B.ob, B.qob.q, B.qob.meta, dattn_part_},
           {D.nh, D.nkv, D.hd, D.swa, D.ring, (int)wmma});
}

void Qwen4ExpProgram::draft_ffn_half(const Gguf& G, const DraftDims& D, int l, const float* fbase) {
    const int d = dm_.d;
    const DraftBufs& B = db_;
    auto bt = [&](const char* name) -> const TensorInfo& { return block_tensor(G, l, name); };
    pb_.op(OP_NORM, {B.x, (void*)g_.upload_f32(bt("ffn_norm.weight")), B.hn, B.qhn.q, B.qhn.meta}, {d}, {D.eps});
    pb_.mv_begin();
    pb_.addw(bt("ffn_conv_proj.weight"), g_.upload(bt("ffn_conv_proj.weight")), B.hn, &B.qhn, B.dyn);
    pb_.mv_end("d.conv_proj");
    pb_.op(OP_DCONV, {B.hn, B.dyn, (void*)fbase, B.h2, B.qh2.q, B.qh2.meta}, {d, D.group, 0, 0});
    pb_.mv_begin();
    pb_.addw(bt("ffn_gate.weight"), g_.upload(bt("ffn_gate.weight")), B.h2, &B.qh2, B.fg);
    pb_.addw(bt("ffn_up.weight"), g_.upload(bt("ffn_up.weight")), B.h2, &B.qh2, B.fu);
    pb_.mv_end("d.gate_up");
    pb_.op(OP_SWIGLU, {B.fg, B.fu, B.fa, B.qfa.q, B.qfa.meta}, {D.ff});
    pb_.mv_begin();
    pb_.addw(bt("ffn_down.weight"), g_.upload(bt("ffn_down.weight")), B.fa, &B.qfa, B.ab);
    pb_.mv_end("d.down");
    pb_.op(OP_DCONV, {B.ab, B.dyn, (void*)fbase, B.x, nullptr, nullptr}, {d, D.group, 1, 1});
}

void Qwen4ExpProgram::draft_head(const Gguf& G) {
    const DraftBufs& B = db_;
    pb_.op(OP_NORM, {B.x, (void*)g_.upload_f32(draft_tensor(G, "output_norm.weight")), B.hn, B.qhn.q, B.qhn.meta},
           {dm_.d}, {(float)G.get_float("dflash.attention.layer_norm_rms_epsilon")});
    g_.dlog_ = g_.alloc_b(h_.vocab);
    g_.dgate_ = g_.alloc_b(g_.drank_);
    pb_.mv_begin();
    pb_.addw(*m_.output(), g_.head_w_, B.hn, &B.qhn, g_.dlog_);
    pb_.addw(draft_tensor(G, "selector_hidden.weight"), g_.upload(draft_tensor(G, "selector_hidden.weight")), B.hn,
             &B.qhn, g_.dgate_);
    pb_.mv_end("d.head");
}

// Upload and the host buffers of the drafter; the slots and the other programs follow in ProgramBuilder::run
void Qwen4ExpProgram::draft_finish() {
    g_.upload_prog(g_.dops_, g_.dtasks_, g_.d_dops_, g_.d_dtasks_);
    g_.dlog_host_.resize((size_t)(MEGA_BMAX - 1) * h_.vocab);
    g_.dgate_host_.resize((size_t)(MEGA_BMAX - 1) * g_.drank_);
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.in2_host_), (size_t)MEGA_BMAX * dm_.d * sizeof(float),
                                hipHostMallocDefault));
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.dparams_host_), sizeof(MegaParams), hipHostMallocDefault));
    std::memset(g_.dparams_host_, 0, sizeof(MegaParams));
    // Drafter: its own threshold (TONA_WIDE_DRAFT_MIN); its rounding need not be the same over the width (the target
    // checks)
    g_.dparams_host_->wide_min = (int32_t)env::integer("TONA_WIDE_DRAFT_MIN", g_.wide_min_);
    HIPCHECK(trace::dev_malloc(reinterpret_cast<void**>(&g_.d_dparams_), sizeof(MegaParams)));
    HIPCHECK(hipMemset(g_.d_dparams_, 0, sizeof(MegaParams)));
}

}  // namespace tona::gpu
