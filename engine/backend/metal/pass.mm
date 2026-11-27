#include "backend/metal/pass.h"

#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

#include "backend/metal/gpu_gaps.h"
#include "backend/metal/host_math.h"
#include "backend/metal/kernels/params.h"
#include "backend/ple_prefetch.h"

namespace tona::metal {

namespace {

MTLSize n1(uint64_t n) { return MTLSizeMake(n, 1, 1); }
uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

void check_shapes(const Hparams& h) {
    if (h.dn_head_dim != 128) throw std::runtime_error("Metal: DeltaNet head dim must be 128");
    if (h.n_expert > 1024) throw std::runtime_error("Metal: more than 1024 experts");
    if (h.idx_dim > 512 || h.rope_dims > h.idx_dim) throw std::runtime_error("Metal: indexer dim above 512");
    if ((uint64_t)h.ple_heads * h.ple_dim > h.d) throw std::runtime_error("Metal: n-gram embedding wider than d");
}

}  // namespace

Pass::Pass(Device& dev, Weights& w, const Qwen4ExpModel& m, const MtpHead* mtp, int64_t max_ctx, int cap)
    : dev_(dev), w_(w), m_(m), h_(m.hp()), mtp_(mtp), max_ctx_(max_ctx), cap_(std::max(cap, BMAX)) {
    const Hparams& h = h_;
    check_shapes(h);
    ple_ = w.mat(m.ple_embd());
    const uint64_t d = h.d, hc = h.n_hc, vd = h.dn_v_dim(), ch = 2 * (uint64_t)h.dn_k_dim() + vd;
    const uint64_t nq = (uint64_t)h.n_head * h.head_dim, used = h.n_expert_used;
    cell_cap_ = h.idx_top_k / h.idx_block * h.idx_block + h.idx_block;
    pool_cap_ = (uint32_t)ceil_div(max_ctx / h.idx_block + 1, 32) * 32;
    if (cell_cap_ > ATTN_MAX_CELLS) throw std::runtime_error("Metal: more indexer cells than attend_part() holds");
    if (h.head_dim % 128) throw std::runtime_error("Metal: attend_part needs head_dim a multiple of 128");
    attn_part_ = dev.floats((uint64_t)h.n_head * ATTN_PARTS * (h.head_dim + 2));
    // Default 2: 27 to 28 tok/s against 23 without and 24 with =1 (2026-11-15-mac-coder-arena.md); 0 turns it off
    const int64_t ahead = w.arena() ? env::integer("TONA_METAL_ARENA_AHEAD", 2) : 0;
    if (ahead > 0) {
        ahead_k_ = (uint32_t)std::clamp<int64_t>(env::integer("TONA_METAL_ARENA_AHEAD_K", (int64_t)used), 1, 32);
        ahead_logits_ = dev.floats((uint64_t)BMAX * h.n_expert), ahead_ids_ = dev.floats((uint64_t)BMAX * ahead_k_);
        ahead_ew_ = dev.floats((uint64_t)BMAX * ahead_k_), ahead_nan_ = dev.floats(1);
        if (ahead == 2)
            ahead_hc_ = {dev.floats((uint64_t)BMAX * hc * d), dev.floats((uint64_t)BMAX * h.hc_rank),
                         dev.floats((uint64_t)BMAX * hc * d), dev.floats((uint64_t)BMAX * d)};
    }
    auto per_token = [&](uint64_t n) { return dev.floats(n * cap_); };
    s_ = {per_token(hc * d),
          per_token(hc * d),
          per_token(h.hc_rank),
          per_token(hc * d),
          per_token(d),
          per_token(hc),
          per_token(d),
          per_token(ch),
          per_token(vd),
          per_token(h.dn_v_heads),
          per_token(h.dn_v_heads),
          per_token(ch),
          per_token(std::max(vd, nq)),
          per_token(2 * nq),
          per_token((uint64_t)h.idx_heads * h.idx_dim),
          dev.floats(max_ctx / h.idx_block + 1),
          per_token(cell_cap_),
          per_token(std::max(h.n_expert, mtp ? mtp->n_expert : 0)),
          per_token(used),
          per_token(used),
          per_token(used * h.expert_ff),
          per_token(used * h.expert_ff),
          per_token(used * d),
          per_token(h.shexp_ff),
          per_token(h.shexp_ff),
          per_token(d),
          per_token(1),
          per_token(d),
          per_token(hc * d),
          per_token(d),
          per_token(h.rope_dims),
          per_token(h.rope_dims),
          per_token(1),
          per_token(h.ple_heads),
          dev.floats(1),
          dev.floats(std::max(h.n_expert, mtp ? mtp->n_expert : 0) + 1),
          per_token(used),
          per_token(pool_cap_),
          per_token(pool_cap_ / 32),
          per_token((uint64_t)h.ple_heads * ceil_div(ple_.row_bytes, 4)),
          per_token(used)};
    int n_dn = 0, n_qsa = 0;
    for (uint32_t l = 0; l < h.n_layer; l++) slot_of_layer_.push_back(h.is_qsa(l) ? n_qsa++ : n_dn++);
    mb_.slot = n_qsa;
    if (mtp) mtp_init(*mtp);
}

// Per token: RoPE table, the tail of its indexer cells; per completed block: its RoPE table
void Pass::write_positions(int64_t pos, int nb) {
    const Hparams& h = h_;
    blocks_.clear();
    for (int t = 0; t < nb; t++) {
        const int64_t p = pos + t;
        const auto cs = rope_table(h.rope_dims, p, h.rope_base);
        std::copy(cs.begin(), cs.end(), s_.cs_pos.f32() + (size_t)t * h.rope_dims);
        if ((p + 1) % h.idx_block == 0) {
            const auto cb = rope_table(h.rope_dims, p + 1 - h.idx_block, h.rope_base);
            std::copy(cb.begin(), cb.end(), s_.cs_blk.f32() + blocks_.size() * h.rope_dims);
            blocks_.push_back(p + 1 - h.idx_block);
        }
        const QsaCells c = qsa_cells(p, h.idx_block, h.idx_top_k);
        int32_t* cells = s_.cells.i32() + (uint64_t)t * cell_cap_;
        for (int64_t i = 0; i < c.n_tail; i++) cells[c.top_blocks * c.blk + i] = (int32_t)(c.tail0 + i);
    }
    s_.nan.i32()[0] = 0;
}

// Per token: its id and n-gram rows, then the positions
void Pass::write_inputs(const PassIn& in) {
    const Hparams& h = h_;
    const double t0 = GpuGaps::enabled() ? GpuGaps::now() : 0;
    const size_t hist0 = in.history->size() - in.nb;
    backend::PleStats::Copy timed(ple_stats_, in.prefill, in.nb);
    // all rows first, so that the read-ahead can ask for their pages together; row j of token t is row t * heads + j
    std::vector<int32_t> rows((size_t)in.nb * h.ple_heads);
    for (int t = 0; t < in.nb; t++) {
        s_.tok.i32()[t] = in.tokens[t];
        tona::ple::rows_at(h, in.history->data(), (int64_t)hist0 + t, rows.data() + (size_t)t * h.ple_heads);
    }
    if (backend::ple_prefetch::enabled())
        timed.failed(backend::ple_prefetch::prefetch(ple_.host, ple_.row_bytes, rows.data(), rows.size()));
    for (size_t at = 0; at < rows.size(); at++) {
        std::memcpy((uint8_t*)s_.ple_stage.b.contents + at * ple_.row_bytes,
                    ple_.host + (uint64_t)rows[at] * ple_.row_bytes, ple_.row_bytes);
        s_.ple_rows.i32()[at] = (int32_t)at;
    }
    const double t1 = GpuGaps::enabled() ? GpuGaps::now() : 0;
    write_positions(in.pos, in.nb);
    if (GpuGaps::enabled()) rows_s_ = t1 - t0, positions_s_ = GpuGaps::now() - t1;
}

void Pass::prefetch_rows(const int32_t* seq, int64_t from, int64_t to) {
    if (!backend::ple_prefetch::ahead() || to <= from) return;
    std::vector<int32_t> rows((size_t)(to - from) * h_.ple_heads);
    for (int64_t i = from; i < to; i++) tona::ple::rows_at(h_, seq, i, rows.data() + (size_t)(i - from) * h_.ple_heads);
    ple_stats_.refused(backend::ple_prefetch::prefetch(ple_.host, ple_.row_bytes, rows.data(), rows.size()));
}

void Pass::begin(id<MTLComputeCommandEncoder> enc, const PassIn& in) {
    if (in.nb < 1 || in.nb > (in.prefill ? cap_ : BMAX)) throw std::logic_error("Metal: pass width out of range");
    if (in.pos + in.nb > max_ctx_) throw std::runtime_error("Metal: context full");
    gemm_ = in.prefill;
    write_inputs(in);
    Dispatch d{enc, dev_};
    const Mat te = w_.mat(m_.tok_embd());
    d.run("gather_" + te.kernel_type, {te.w, s_.tok, s_.R}, GatherParams{h_.d, te.row_bytes, h_.n_hc},
          MTLSizeMake(ceil_div(h_.d / 8, 64), in.nb, 1), MTLSizeMake(64, 1, 1));
}

void Pass::layer(id<MTLComputeCommandEncoder> enc, SeqState& st, const PassIn& in, uint32_t l) {
    Dispatch d{enc, dev_};
    const Layer& L = m_.layers()[l];
    const int nb = in.nb;
    if (l == h_.ple_layer) part(d, "ple"), ple(d, L, st.live, nb);
    part(d, "gated_read");
    hc_mix(d, L.gr_attn, true, nb);
    if (h_.is_qsa(l))
        qsa(d, L, slot_of_layer_[l], st, in);
    else
        deltanet(d, L, slot_of_layer_[l], st, nb);
    part(d, "gated_read");
    hc_combine(d, nb);
    hc_mix(d, L.gr_ffn, true, nb);
    moe(d, L, nb);
    part(d, "gated_read");
    hc_combine(d, nb);
}

void Pass::finish(id<MTLComputeCommandEncoder> enc, SeqState& st, const PassIn& in) {
    st.live.start = (st.live.start + in.nb) % ((h_.ple_conv - 1) * h_.ple_ngram);
    if (!in.logits.b && !in.mix_head) return;
    Dispatch d{enc, dev_};
    part(d, "head");
    hc_mix(d, m_.gr_head(), false, in.nb);
    if (in.logits.b) mv(d, w_.mat(m_.output()), s_.mixed, in.logits, in.nb);
}

void Pass::head(id<MTLComputeCommandEncoder> enc, int t0, int n, const Buf& logits) {
    Dispatch d{enc, dev_};
    mv(d, w_.mat(m_.output()), s_.mixed.at((uint64_t)t0 * h_.d * 4), logits, n);
}

void Pass::encode(id<MTLComputeCommandEncoder> enc, SeqState& st, const PassIn& in) {
    begin(enc, in);
    for (uint32_t l = 0; l < h_.n_layer; l++) layer(enc, st, in, l);
    finish(enc, st, in);
}

// Dense matrix over nb tokens: in a prefill one GEMM (tiles of 64 rows × 64 tokens), otherwise at most MV_BMAX tokens
// per mat-vec dispatch (5 with TONA_METAL_MV5). Narrow matrices (< 2048 rows) split K over 8 simdgroups,
// wider ones spread rows over 4 simdgroups (research/measurements/2026-11-11-mac-densek.md). The shape depends on the
// matrix only, never on nb: that keeps every width bit-identical.
void Pass::mv(Dispatch& d, const Mat& w, const Buf& x, const Buf& y, int nb) {
    if (gemm_) {
        if (w.k % 32) throw std::runtime_error("Metal: GEMM needs K a multiple of 32");
        if (w.rows < 16) {  // a simdgroup per row (kernels/gemm.metal, gemv_narrow)
            d.run("gemv_narrow_" + w.kernel_type, {w.w, x, y}, GemmParams{w.rows, w.k, w.row_bytes, (uint32_t)nb, 0, 1},
                  n1(nb), n1(32ull * w.rows));
            return;
        }
        d.run("gemm_" + w.kernel_type, {w.w, x, y}, GemmParams{w.rows, w.k, w.row_bytes, (uint32_t)nb, 0, 1},
              MTLSizeMake(ceil_div(w.rows, 64), ceil_div(nb, 64), 1), n1(256));
        return;
    }
    const uint32_t R = w.rows % 4 == 0 ? 4 : 1;
    MvParams p{w.rows, w.k, w.row_bytes, 0, 0, 0, 1, 0};
    for (int t0 = 0, B = 0; t0 < nb; t0 += B) {
        const Buf xt = x.at((uint64_t)t0 * w.k * 4), yt = y.at((uint64_t)t0 * w.rows * 4);
        if (use_mma(w, nb - t0)) {
            p.tokens = B = std::min(mma_tokens_, nb - t0);
            d.run("mvm_" + w.kernel_type, {w.w, xt, yt, yt}, p, MTLSizeMake(w.rows / 8, 1, 1), n1(256));
            continue;
        }
        B = mv_chunk(nb - t0);
        const std::string name = "_" + w.kernel_type + "_r" + std::to_string(R) + "_b" + std::to_string(B);
        if (w.rows < 2048)
            d.run("mvs" + name, {w.w, xt, yt, yt}, p, MTLSizeMake(w.rows / R, 1, 1), n1(256));
        else
            d.run("mv" + name, {w.w, xt, yt, yt}, p, MTLSizeMake(ceil_div(w.rows, 4ull * R), 1, 1), n1(128));
    }
}

// Tokens of the next f32 mat-vec dispatch with n left: chunks of 4, with TONA_METAL_MV5 as few chunks of at most 5 as
// fit, evenly split (llama.cpp mul_mv_ext: 6 as 3+3, 7 as 4+3, 8 as 4+4)
int Pass::mv_chunk(int n) const {
    if (!mv5_) return std::min(MV_BMAX, n);
    const int chunks = (n + 4) / 5;
    return (n + chunks - 1) / chunks;
}

// The half MMA kernel: at least 4 tokens left, a wide matrix of whole 8-row tiles, a type it has an instance for
bool Pass::use_mma(const Mat& w, int nb) const {
    static const std::set<std::string> types = {"F16", "BF16", "Q80", "Q4K", "Q5K", "Q6K", "Q6KT", "IQ4XS", "IQ4NL"};
    // not the LM head: its 151k rows ran 2.0 ms slower per pass (research/measurements/2026-11-12-mac-q6k-tile.md)
    return mma_ && nb >= 4 && w.rows >= 2048 && w.rows < h_.vocab && w.rows % 8 == 0 && types.contains(w.kernel_type);
}

void Pass::mv_experts(Dispatch& d, const Mat& w, const Buf& x, const Buf& y, const Slots& sl) {
    if (gemm_) {  // a tile column per expert; an expert has at most one slot per token
        if (w.k % 32) throw std::runtime_error("Metal: GEMM needs K a multiple of 32");
        const GemmParams p{w.rows, w.k, w.row_bytes, sl.n, (uint32_t)w.expert_bytes, sl.xdiv};
        const MTLSize grid = MTLSizeMake(ceil_div(w.rows, 64), ceil_div(sl.n / h_.n_expert_used, 64), w.experts);
        if (sl.tab.b)
            d.run("gemmxt_" + w.kernel_type, {x, y, s_.goff, s_.perm, sl.tab}, p, grid, n1(256));
        else
            d.run("gemmx_" + w.kernel_type, {w.w, x, y, s_.goff, s_.perm}, p, grid, n1(256));
        return;
    }
    // Grouped by expert: a threadgroup row per expert the pass uses, at most one per slot. One token: a row per slot
    // with mv, the same arithmetic as mvg with one slot, so widths stay bit-identical
    const uint32_t R = w.rows % 4 == 0 ? 4 : 1;
    const MvParams p{w.rows, w.k, w.row_bytes, (uint32_t)w.expert_bytes, sl.xstride, 1, sl.xdiv, 0};
    const MTLSize grid = MTLSizeMake(ceil_div(w.rows, 4ull * R), sl.n, 1);
    const std::string tr = w.kernel_type + "_r" + std::to_string(R);
    if (!sl.grouped) {
        if (sl.tab.b)
            d.run("mvt_" + tr, {x, y, sl.ids, sl.tab}, p, grid, n1(128));
        else
            d.run("mv_" + tr + "_b1", {w.w, x, y, sl.ids}, p, grid, n1(128));
        return;
    }
    MvParams pg = p;
    pg.tokens = mvg_split_;  // slots per glist entry
    if (sl.tab.b) {          // one instance of at most 4 slots per weight pass (moe_table.metal)
        d.run("mvgt_" + tr, {x, y, s_.goff, s_.perm, s_.glist, sl.tab}, pg, grid, n1(128));
        return;
    }
    const std::string kind = mvg_split_ == 1 || mvg_split_ == 2 ? "mvg2_" : "mvg_";
    d.run(kind + tr, {w.w, x, y, s_.goff, s_.perm, s_.glist}, pg, grid, n1(128));
}

// RMS norm of the rows of R into xn; row r uses weight row r % w_rows
void Pass::norm_rows(Dispatch& d, const Buf& w, uint32_t rows, uint32_t w_rows, const Buf& xn) {
    const uint32_t n = h_.d;
    d.run("rms_rows", {s_.R, xn, w}, NormParams{n, n, n, n, 1, w_rows, h_.rms_eps, 1.0f}, n1(rows), n1(256));
}

// Gated residual read (ref.cpp hc_mix): norm per stream, low-rank gate, mean of the gated streams
void Pass::hc_mix(Dispatch& d, const GatedResidual& gr, bool inject, int nb) {
    hc_mix_to(d, gr, nb, {s_.xn, s_.lo, s_.gate, s_.mixed});
    if (inject) mv(d, w_.mat(gr.inject), s_.xn, s_.inject, nb);
}

void Pass::hc_mix_to(Dispatch& d, const GatedResidual& gr, int nb, const HcOut& o) {
    const uint32_t D = h_.d, hc = h_.n_hc, rank = h_.hc_rank * nb;
    norm_rows(d, w_.vec(gr.norm), hc * nb, hc, o.xn);
    mv(d, w_.mat(gr.down), o.xn, o.lo, nb);
    d.run("silu_scale", {o.lo}, ScaleParams{rank, 1.0f / (float)hc}, n1(ceil_div(rank, 256)), n1(256));
    mv(d, w_.mat(gr.up), o.lo, o.gate, nb);
    d.run("hc_mixed", {o.xn, o.gate, o.mixed}, HcParams{D, hc, (uint32_t)nb}, n1(ceil_div((uint64_t)D * nb, 256)),
          n1(256));
}

void Pass::hc_combine(Dispatch& d, int nb) {
    d.run("hc_combine", {s_.R, s_.y, s_.inject}, HcParams{h_.d, h_.n_hc, (uint32_t)nb},
          n1(ceil_div((uint64_t)h_.d * h_.n_hc * nb, 256)), n1(256));
}

}  // namespace tona::metal
