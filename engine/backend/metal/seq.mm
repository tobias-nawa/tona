#include "backend/metal/seq.h"

#include <cstring>
#include <stdexcept>

namespace tona::metal {

namespace {

uint64_t ch_of(const Hparams& h) { return 2 * (uint64_t)h.dn_k_dim() + h.dn_v_dim(); }

Recurrent make_recurrent(Device& dev, const Hparams& h) {
    Recurrent r;
    for (uint32_t l = 0; l < h.n_layer; l++) {
        if (h.is_qsa(l)) continue;
        r.dn_state.push_back(dev.floats((uint64_t)h.dn_v_heads * 128 * 128));
        r.dn_win.push_back(dev.floats((uint64_t)(h.dn_conv - 1) * ch_of(h)));
    }
    r.ple_ring = dev.floats((uint64_t)(h.ple_conv - 1) * h.ple_ngram * h.n_hc * h.d);
    return r;
}

// Every buffer of a Recurrent in one order, for copies and the stash layout
std::vector<Buf> buffers(const Recurrent& r) {
    std::vector<Buf> all(r.dn_state);
    all.insert(all.end(), r.dn_win.begin(), r.dn_win.end());
    all.push_back(r.ple_ring);
    return all;
}

void zero(const Buf& b) { std::memset(b.b.contents, 0, b.b.length); }

}  // namespace

SeqState make_seq(Device& dev, const Hparams& h, int64_t max_ctx, int extra_qsa) {
    SeqState s{make_recurrent(dev, h), make_recurrent(dev, h), make_recurrent(dev, h), {}, {}, {}, {}, {}};
    for (size_t i = 0; i < s.live.dn_state.size(); i++) {
        s.keep.qkv.push_back(dev.floats((uint64_t)backend::BMAX * ch_of(h)));
        s.keep.alpha.push_back(dev.floats((uint64_t)backend::BMAX * h.dn_v_heads));
        s.keep.beta.push_back(dev.floats((uint64_t)backend::BMAX * h.dn_v_heads));
    }
    const uint64_t nkv = (uint64_t)h.n_head_kv * h.head_dim;
    for (uint32_t l = 0; l < h.n_layer + (uint32_t)extra_qsa; l++) {
        if (l < h.n_layer && !h.is_qsa(l)) continue;
        // Slack for the prefill tiles: attend_t reads K and V in windows of 8 cells, idx_score_t pool rows in tiles of
        // 32 blocks; what lies past the end is masked, but must be inside the buffer
        s.kv_k.push_back(dev.floats((max_ctx + 8) * nkv));
        s.kv_v.push_back(dev.floats((max_ctx + 8) * nkv));
        s.idx_raw.push_back(dev.floats(max_ctx * h.idx_dim));
        s.idx_pool.push_back(dev.floats((max_ctx / h.idx_block + 32) * h.idx_dim));
    }
    return s;
}

SeqLayout seq_layout(const Hparams& h) {
    SeqLayout l;
    uint32_t n_dn = 0;
    for (uint32_t i = 0; i < h.n_layer; i++) h.is_qsa(i) ? l.n_qsa++ : n_dn++;
    l.recurrent = n_dn * ((uint64_t)h.dn_v_heads * 128 * 128 + (uint64_t)(h.dn_conv - 1) * ch_of(h)) * 4 +
                  (uint64_t)(h.ple_conv - 1) * h.ple_ngram * h.n_hc * h.d * 4;
    l.kv = (uint64_t)h.n_head_kv * h.head_dim * 4;
    l.raw = (uint64_t)h.idx_dim * 4;
    l.pool = (uint64_t)h.idx_dim * 4;
    l.idx_block = h.idx_block;
    return l;
}

void clear(SeqState& s) {
    for (Recurrent* r : {&s.live, &s.backup, &s.saved}) {
        for (const Buf& b : buffers(*r)) zero(b);
        r->start = 0;
    }
    for (auto* v : {&s.kv_k, &s.kv_v, &s.idx_raw, &s.idx_pool})
        for (const Buf& b : *v) zero(b);
}

void copy(id<MTLCommandBuffer> cb, const Recurrent& from, Recurrent& to) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    const std::vector<Buf> a = buffers(from), b = buffers(to);
    for (size_t i = 0; i < a.size(); i++)
        [blit copyFromBuffer:a[i].b sourceOffset:0 toBuffer:b[i].b destinationOffset:0 size:a[i].b.length];
    [blit endEncoding];
    to.start = from.start;
}

void restore_after(id<MTLCommandBuffer> cb, SeqState& s, int m, int nb, uint32_t hist) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    const auto all = [&](const std::vector<Buf>& from, const std::vector<Buf>& to) {
        for (size_t i = 0; i < from.size(); i++)
            [blit copyFromBuffer:from[i].b sourceOffset:0 toBuffer:to[i].b destinationOffset:0 size:from[i].b.length];
    };
    all(s.backup.dn_state, s.live.dn_state);
    all(s.backup.dn_win, s.live.dn_win);
    // Ring slot (start + t) % hist holds token t's entry (nb <= hist): the slots of the rejected tokens back
    const uint64_t slot = s.backup.ple_ring.b.length / hist;
    for (int t = m; t < nb; t++) {
        const uint64_t off = (uint64_t)((s.backup.start + t) % hist) * slot;
        [blit copyFromBuffer:s.backup.ple_ring.b
                 sourceOffset:off
                     toBuffer:s.live.ple_ring.b
            destinationOffset:off
                         size:slot];
    }
    [blit endEncoding];
    s.live.start = (s.backup.start + m) % hist;
}

void stash(const SeqState& s, const SeqLayout& l, int64_t pos, backend::HostSnap& out) {
    out.state.resize(l.recurrent + 4);
    uint8_t* p = out.state.data();
    for (const Buf& b : buffers(s.saved)) {
        std::memcpy(p, b.b.contents, b.b.length);
        p += b.b.length;
    }
    std::memcpy(p, &s.saved.start, 4);
    const uint64_t kvb = pos * l.kv, rawb = pos * l.raw, poolb = l.pool_rows(pos) * l.pool;
    out.kv.resize(l.n_qsa * (2 * kvb + rawb));
    out.pool.resize(l.n_qsa * poolb);
    uint8_t* k = out.kv.data();
    for (uint32_t i = 0; i < l.n_qsa; i++) {
        std::memcpy(k, s.kv_k[i].b.contents, kvb), k += kvb;
        std::memcpy(k, s.kv_v[i].b.contents, kvb), k += kvb;
        std::memcpy(k, s.idx_raw[i].b.contents, rawb), k += rawb;
        std::memcpy(out.pool.data() + i * poolb, s.idx_pool[i].b.contents, poolb);
    }
    out.dring.clear();
}

void unstash(SeqState& s, const SeqLayout& l, const backend::HostSnap& in) {
    const uint8_t* p = in.state.data();
    for (const Buf& b : buffers(s.saved)) {
        std::memcpy(b.b.contents, p, b.b.length);
        p += b.b.length;
    }
    std::memcpy(&s.saved.start, p, 4);
    const uint64_t kvb = in.pos * l.kv, rawb = in.pos * l.raw, poolb = l.pool_rows(in.pos) * l.pool;
    const uint8_t* k = in.kv.data();
    for (uint32_t i = 0; i < l.n_qsa; i++) {
        std::memcpy(s.kv_k[i].b.contents, k, kvb), k += kvb;
        std::memcpy(s.kv_v[i].b.contents, k, kvb), k += kvb;
        std::memcpy(s.idx_raw[i].b.contents, k, rawb), k += rawb;
        std::memcpy(s.idx_pool[i].b.contents, in.pool.data() + i * poolb, poolb);
    }
}

bool snap_fits(const SeqLayout& l, const backend::HostSnap& in, int64_t max_ctx) {
    if (!backend::snap_pos_ok(in, max_ctx)) return false;
    if (in.state.size() != l.recurrent + 4 || !in.dring.empty()) return false;
    return backend::snap_kv_pool_ok(in, in.pos * l.kv_row(), l.n_qsa * l.pool_rows(in.pos) * l.pool);
}

}  // namespace tona::metal
