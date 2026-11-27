#include "gpu/mega_dump.h"

#include <algorithm>
#include <format>
#include <stdexcept>

#include "gpu/prefill_run.h"

namespace tona::gpu {

ProgramDump::ProgramDump(const Mega& g, const char* path) : g_(g), f_(std::fopen(path, "w")) {
    if (!f_) throw std::runtime_error(std::format("gpu::Mega: cannot write {}", path));
}

ProgramDump::~ProgramDump() {
    if (f_) std::fclose(f_);
}

// Regions are the allocations and the mapped host buffers; the allocation order, not the address, names them
void ProgramDump::refresh() {
    regions_.clear();
    auto add = [&](const void* p, std::string name) {
        if (p) regions_.push_back({reinterpret_cast<uintptr_t>(p), std::move(name)});
    };
    for (size_t i = 0; i < g_.allocs_.size(); i++) add(g_.allocs_[i], std::format("dev{}", i));
    for (size_t i = 0; i < g_.pin_allocs_.size(); i++) add(g_.pin_allocs_[i], std::format("pin{}", i));
    for (size_t i = 0; i < g_.seq_bufs_.size(); i++) {
        add(g_.seq_bufs_[i].dev[0], std::format("seq{}", i));
        add(g_.seq_bufs_[i].host[0], std::format("seqh{}", i));
    }
    add(g_.mbox_dev_, "mbox");
    add(g_.staging_dev_, "staging");
    add(g_.fill_dev_, "fill");
    add(g_.cpu_io_dev_, "cpu_io");
    add(g_.d_ops_, "d_ops"), add(g_.d_tasks_, "d_tasks"), add(g_.d_xops_, "d_xops"), add(g_.d_xmops_, "d_xmops");
    std::stable_sort(regions_.begin(), regions_.end(),
                     [](const Region& a, const Region& b) { return a.base < b.base; });
}

std::string ProgramDump::ptr(const void* p) const {
    if (!p) return "0";
    const auto a = reinterpret_cast<uintptr_t>(p);
    auto it =
        std::upper_bound(regions_.begin(), regions_.end(), a, [](uintptr_t v, const Region& r) { return v < r.base; });
    if (it == regions_.begin()) return std::format("?{:#x}", a);
    --it;
    return std::format("{}+{:#x}", it->name, a - it->base);
}

void ProgramDump::ops(std::string_view name, const std::vector<MegaOp>& ops, const std::vector<std::string>& labels) {
    refresh();
    std::fprintf(f_, "program %.*s %zu\n", (int)name.size(), name.data(), ops.size());
    for (size_t k = 0; k < ops.size(); k++) {
        const MegaOp& o = ops[k];
        std::fprintf(f_, "O%zu %s kind=%d cond=%d a=%d b=%d nobar=%d seq=%d i=", k,
                     k < labels.size() ? labels[k].c_str() : "-", o.kind, o.cond, o.a, o.b, o.nobar, o.seq);
        for (int v : o.i) std::fprintf(f_, "%d,", v);
        std::fprintf(f_, " f=");
        for (float v : o.f) std::fprintf(f_, "%a,", (double)v);
        std::fprintf(f_, " p=");
        for (const void* v : o.p) std::fprintf(f_, "%s,", ptr(v).c_str());
        std::fputc('\n', f_);
    }
}

void ProgramDump::table(std::string_view name, const void* data, size_t bytes) {
    std::fprintf(f_, "table %.*s %zu\n", (int)name.size(), name.data(), bytes);
    const auto* w = static_cast<const uint32_t*>(data);
    for (size_t i = 0; i < bytes / 4; i++) std::fprintf(f_, "%x%c", w[i], i % 16 == 15 ? '\n' : ' ');
    std::fputc('\n', f_);
}

void ProgramDump::pf_weight(const char* what, const PfW& w) {
    std::fprintf(f_, "  %s type=%u in=%lld rows=%lld w=%s\n", what, w.type, (long long)w.in, (long long)w.rows,
                 ptr(w.w).c_str());
}

void ProgramDump::patches(const char* name, const std::vector<XPatch>& v) {
    std::fprintf(f_, "patches %s %zu\n", name, v.size());
    for (const auto& x : v)
        std::fprintf(f_, "  op=%d j=%d seq=%d base=%s row=%lld\n", x.op, x.j, x.seq, ptr(x.base).c_str(),
                     (long long)x.row);
}

PfDumpSink ProgramDump::sink() {
    return {[this](const char* what, const PfW& w) { pf_weight(what, w); },
            [this](const char* what, const void* p) { std::fprintf(f_, "  %s=%s\n", what, ptr(p).c_str()); }};
}

void ProgramDump::pf_layers() {
    for (size_t l = 0; l < g_.pf_.size(); l++) {
        const PfLayer& F = g_.pf_[l];
        std::fprintf(f_, "pf layer %zu\n", l);
        prefill_sequence(g_.m_).dump_layer(g_.pf_arch_, l, sink());
        pf_weight("qkv", F.dn.qkv), pf_weight("gate", F.dn.gate), pf_weight("beta", F.dn.beta),
            pf_weight("alpha", F.dn.alpha);
        pf_weight("ssm_out", F.dn.ssm_out), pf_weight("q", F.attn.q), pf_weight("k", F.attn.k),
            pf_weight("v", F.attn.v);
        pf_weight("o", F.attn.o), pf_weight("idx_q", F.attn.idx.idx_q), pf_weight("idx_k", F.attn.idx.idx_k),
            pf_weight("router", F.moe.router);
        pf_weight("sh_gate", F.moe.sh_gate), pf_weight("sh_up", F.moe.sh_up), pf_weight("sh_down", F.moe.sh_down);
        pf_weight("sh_inp", F.moe.sh_inp);
        std::fprintf(f_, "  conv_w=%s dt_bias=%s A=%s ssm_norm=%s state=%s ring=%s pend=%s,%s,%s\n",
                     ptr(F.dn.conv_w).c_str(), ptr(F.dn.dt_bias).c_str(), ptr(F.dn.A).c_str(),
                     ptr(F.dn.ssm_norm).c_str(), ptr(F.dn.state).c_str(), ptr(F.dn.ring).c_str(),
                     ptr(F.dn.pend_d).c_str(), ptr(F.dn.pend_a).c_str(), ptr(F.dn.pend_k).c_str());
        std::fprintf(f_, "  norms=%s,%s,%s,%s kv=%s raw_ring=%s pool=%s\n", ptr(F.attn.q_norm).c_str(),
                     ptr(F.attn.k_norm).c_str(), ptr(F.attn.idx.idx_q_norm).c_str(), ptr(F.attn.idx.idx_k_norm).c_str(),
                     ptr(F.attn.kv).c_str(), ptr(F.attn.idx.raw_ring).c_str(), ptr(F.attn.idx.pool).c_str());
    }
}

void ProgramDump::state() {
    refresh();
    const Mega& g = g_;
    std::fprintf(
        f_, "state blocks=%d,%d,%d wide_min=%d seqs=%d slots=%d cpu_mode=%d share_wait=%d prefetch=%d fill=%d,%d,%zu\n",
        g.kblocks_[0], g.kblocks_[1], g.kblocks_[2], g.wide_min_, g.n_seqs_, g.slots_, (int)g.cpu_mode_,
        (int)g.share_wait_, g.prefetch_n_, g.fill_n_, g.fill_cap_, g.fill_stride_);
    std::fprintf(f_,
                 "state kv_row=%lld sstride=%lld dswa=%d drank=%d dinject_end=%d kv_bytes=%zu kv_host=%d vram=%zu\n",
                 (long long)g.kv_row_, (long long)g.sstride_, g.dswa_, g.drank_, g.dinject_end_, g.kv_bytes_,
                 (int)g.kv_host_, g.vram_);
    std::fprintf(f_, "state prefill chunk=%d lchunk=%d arena=%zu snap=%s\n", g.pf_chunk_, g.pf_lchunk_,
                 g.pf_arena_bytes_, ptr(g.snap_).c_str());
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : g.mtp_vocab_) h = (h ^ (uint32_t)t) * 1099511628211ull;
    std::fprintf(f_, "state mtp vocab=%zu hash=%llx nstat=%zu rb=%lld q4=%d ex=%u,%lld,%lld,%lld\n",
                 g.mtp_vocab_.size(), (unsigned long long)h, g.mtp_vocab_.n_static(), (long long)g.mtp_rb_,
                 (int)g.mtp_head_q4_, g.mtp_ex_type_, (long long)g.mtp_ex_up_, (long long)g.mtp_ex_down_,
                 (long long)g.mtp_ex_bytes_);
    std::fprintf(f_, "state params budget=%d yield=%d nobar_all=%d wide_min=%d\n", g.params_host_->budget_ms,
                 g.params_host_->yield_ms, g.params_host_->nobar_all, g.params_host_->wide_min);
    for (size_t l = 0; l < g.cache_.size(); l++)
        std::fprintf(f_, "cache %zu slot_bytes=%zu up=%zu down=%zu slots=%s\n", l, g.cache_[l].slot_bytes,
                     g.cache_[l].off_up, g.cache_[l].off_down, ptr(g.cache_[l].slots).c_str());
    pf_layers();
    prefill_sequence(g.m_).dump_head(g.pf_arch_, sink());
    pf_weight("out", g.pf_out_), pf_weight("ple_key", g.pf_ple_key_), pf_weight("ple_val", g.pf_ple_val_);
    prefill_sequence(g.m_).dump_mtp(g.pf_arch_, sink());
    ops("xops", g.xops_, g.xlabels_);
    ops("xmops", g.xmops_, g.xmlabels_);
    patches("xpatch", g.xpatch_);
    patches("xmpatch", g.xmpatch_);
    std::fflush(f_);
}

}  // namespace tona::gpu
