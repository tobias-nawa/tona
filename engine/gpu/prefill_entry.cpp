// Mega::prefill: checks, chunk plan, the embedding of the next chunk on a CPU thread, and the loop over the groups.
// Also the CPU side embedding (embed_tokens) and the injection of the drafter's features.
#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <thread>

#include "backend/env.h"
#include "backend/host_trace.h"
#include "backend/omp_guard.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_extra.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_run.h"
#include "model/progress.h"
#include "model/quant.h"

namespace tona::gpu {

namespace {

// Embeds the next chunk on a thread while the GPU runs the current one (a 2,048-token chunk takes ~105 ms on the CPU,
// in which the GPU would wait). Calls come in the order of the chunks, also layer-major: layers 0-1 run per chunk.
class EmbedAhead {
public:
    using Embed = std::function<void(size_t off, int n, float* out)>;
    EmbedAhead(Embed embed, int n, int chunk, int width, float* host)
        : embed_(std::move(embed)), n_(n), chunk_(chunk), width_(width), host_(host) {}
    EmbedAhead(const EmbedAhead&) = delete;
    EmbedAhead& operator=(const EmbedAhead&) = delete;
    ~EmbedAhead() { join(); }

    void join() {
        if (thread_.joinable()) thread_.join();
    }

    // Embedding of tokens at ... at + m - 1 (host memory); starts the following chunk
    const float* get(int at, int m) {
        join();
        // an error of the look-ahead (e.g. I/O of the PLE table) is rethrown here: in the thread it would terminate the
        // process and kill the whole server
        if (err_) std::rethrow_exception(std::exchange(err_, nullptr));
        if (ahead_at_ == at)
            slot_ ^= 1;
        else
            embed_((size_t)at, m, host_ + (size_t)slot_ * half());
        ahead_at_ = -1;
        const float* out = host_ + (size_t)slot_ * half();
        start(at + m);
        return out;
    }

private:
    size_t half() const { return (size_t)chunk_ * 2 * width_; }
    void start(int at) {
        if (at >= n_) return;
        ahead_at_ = at;
        thread_ = std::thread([this, at, k = slot_ ^ 1] {
            try {
                embed_((size_t)at, std::min(chunk_, n_ - at), host_ + (size_t)k * half());
            } catch (...) {
                err_ = std::current_exception();
            }
        });
    }

    Embed embed_;
    int n_, chunk_;
    int width_;  // d: the embedding and PLE vector of a token take 2 * d floats
    float* host_;
    std::thread thread_;
    std::exception_ptr err_;
    int ahead_at_ = -1, slot_ = 0;
};

}  // namespace

bool PrefillRun::begin(Mega& m, const int32_t* tokens, int n) {
    if (m.pf_chunk_ <= 0) throw std::runtime_error("gpu::Mega: prefill disabled (TONA_PREFILL_CHUNK=0)");
    if (m.awaiting_commit_) throw std::runtime_error("gpu::Mega: prefill without commit of the previous pass");
    for (int s = 0; s < m.n_seqs_; s++)
        if (s != m.cur_slot_) m.seq_host_[s].trunk_ok = false;  // the buffers of the target now belong to this prefill
    if (n < 1) return false;
    m.prefetch_drain();  // the prefill reads the expert slots
    if (m.pos_ + n > m.max_ctx_) throw std::runtime_error("gpu::Mega: context full");
    for (int i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= (int32_t)m.h_.vocab)
            throw std::runtime_error("gpu::Mega: token outside the vocabulary");
    return true;
}

// Chunk of a prompt that needs only one: its size, rounded up to the 128 tokens of the GEMM tiles
int PrefillRun::short_chunk(const Mega& m, int n) {
    if (!m.extras_on_ || n > m.pf_chunk_) return m.pf_chunk_;
    return std::clamp((n + 127) / 128 * 128, 128, m.pf_chunk_);
}

// The chunk-by-chunk prefill occupies the start of the workspace: the extra slots there give way (host bookkeeping
// only, k_mega is not running and prefetch_drain has waited for the copies)
void PrefillRun::extras_out(Mega& m, const PfPlan& plan, int n) {
    if (!m.extras_on_ || plan.layered) return;
    const int taken = evict_extras(m.cache_, (m.prefill_arena_bytes(plan.chunk) + 511) / 256 * 256);
    if (tona::env::flag("TONA_PF_DEBUG"))
        std::fprintf(stderr, "Prefill %d tokens: %d extra slots given up\n", n, taken);
}

// The stream is idle (run() synchronized): the workspace is slots again, empty. After an exception they stay out of use
// until the next prefill gets here.
void PrefillRun::extras_back(Mega& m) {
    if (m.extras_on_) restore_extras(m.cache_);
}

// Every group of the prompt fits into the borrowed slots with chunk and group of plan p
bool PrefillRun::plan_fits(const Mega& m, int n, int64_t win0, const PfPlan& p) {
    for (int i = 0; i < n; i += p.group)
        if (m.pf_layered_need(std::min(p.group, n - i), p.chunk, m.pos_ + i, win0, p.stage, p.wcache) > m.pf_loan_cap())
            return false;
    return true;
}

// One group of exactly two chunks: equal halves instead of a full chunk and a small rest (7,721 tokens at chunk 6144:
// 6144 + 1577 lost 1.4 % prefill against 3968 x 2, and the smaller workspace borrows 17 instead of 23 layers). With
// three or more chunks equal ones are 0.2-0.6 % slower than full ones (research/measurements/2026-11-02-kv-staging.md).
// A smaller chunk needs no more room, but the fit is checked.
PfPlan PrefillRun::balanced_plan(const Mega& m, int n, int64_t win0, PfPlan p) {
    const int k = (n + p.chunk - 1) / p.chunk;
    const int c = std::min(p.chunk, (n + k - 1) / k + 127) / 128 * 128;
    if (!p.layered || n > p.group || k != 2 || c == p.chunk ||
        !plan_fits(m, n, win0, {c, p.group, true, p.stage, p.wcache}))
        return p;
    p.chunk = c;
    return p;
}

// Prompt length above which a prompt goes layer by layer: TONA_PF_LAYERED_MIN, else the architecture's own minimum
// (short_layered_min, 0: none), else more than one chunk
static int layered_from(int chunk, int smin, int env_min) {
    return env_min >= 0 ? env_min : smin > 0 ? smin - 1 : chunk;
}

// First chunk size the layered search tries. An architecture that opts in (short_layered_min) takes prompts of up to
// one chunk layer by layer with a chunk as large as the prompt: the workspace, and with it the borrowed slots to give
// back, stay small. Longer prompts keep the full layered chunk (a chunk of the prompt's size at 3.9K to 5.8K tokens:
// prefill -0.4 %, research/measurements/2026-11-20-short-layered.md)
static int first_layered_chunk(int chunk, int lchunk, int n, int smin) {
    if (smin > 0 && n <= chunk) return std::min(lchunk, (n + 127) / 128 * 128);
    return lchunk;
}

// Largest chunk and group that fit into the borrowed slots (chunks from pf_lchunk_ halved down to pf_chunk_, groups
// from TONA_PF_GROUP halved down to two chunks); otherwise chunk by chunk. TONA_PF_LAYERED_MIN: layer by layer from
// that many tokens on (default: more than one chunk).
PfPlan PrefillRun::choose_plan_for(const Mega& m, int n, int64_t win0, bool wc) {
    static const bool layered_env = tona::env::integer("TONA_PF_LAYERED", 1) != 0;
    static const int lmin = (int)tona::env::integer("TONA_PF_LAYERED_MIN", -1);
    static const int gmax = (int)tona::env::integer("TONA_PF_GROUP", 32768);
    // One chunk of a short prompt needs only as much workspace as it has tokens: the rest of the workspace stays extra
    // slots (mega_extra.h). Without extra slots the chunk is the full one, as before.
    const int c1 = short_chunk(m, n);
    const PfPlan chunked{c1, c1, false};
    const int smin = prefill_sequence(m.m_).short_layered_min();
    if (!layered_env || n <= layered_from(m.pf_chunk_, smin, lmin) || !m.pf_layered_ok()) return chunked;
    const int cfirst = first_layered_chunk(m.pf_chunk_, m.pf_lchunk_, n, smin);
    const int clast = std::min(cfirst, m.pf_chunk_);
    auto fits = [&](int c, int g, bool stage) { return plan_fits(m, n, win0, {c, g, true, stage, wc}); };
    auto search = [&](bool stage) {
        for (int c = cfirst;; c = std::max(clast, c - 1024)) {
            for (int g = std::max(2 * c, std::min(gmax, n + c - 1) / c * c); g >= 2 * c; g = g / 2 / c * c)
                if (fits(c, g, stage)) return PfPlan{c, g, true, stage, wc};
            if (c == clast) break;
        }
        return chunked;
    };
    const PfPlan plain = search(false);
    // The staged KV takes room from the chunk and the group (250K prompt: chunk 5120 -> 4096 and still +27 % prefill,
    // research/measurements/2026-10-30-kv-placement.md); staging pays as long as the chunk loses at most half
    if (plain.layered && m.pf_stage_on(m.pos_ + n)) {
        const PfPlan staged = search(true);
        if (staged.layered && 2 * staged.chunk >= plain.chunk) return balanced_plan(m, n, win0, staged);
    }
    return balanced_plan(m, n, win0, plain);
}

// The weight cache takes room from the chunk and the group. It is used only where the plan stays the one without it: a
// smaller chunk costs more than the unpacking it saves (30K tokens: chunk 5120 -> 4096 took the whole gain).
PfPlan PrefillRun::choose_plan(const Mega& m, int n, int64_t win0) {
    const PfPlan plain = choose_plan_for(m, n, win0, false);
    if (!plain.layered || !pf_flags().wcache || m.pf_wcache_bytes() == 0) return plain;
    PfPlan with = choose_plan_for(m, n, win0, true);
    if (with.layered && with.chunk == plain.chunk && with.group == plain.group && with.stage == plain.stage)
        return with;
    return plain;
}

// The prompt group by group (a group is one layer-major sweep); set_base tells the embedding where the group starts
void PrefillRun::run_groups(Mega& m, const PfJob& all, const std::function<void(int)>& set_base) {
    std::vector<float> part;
    for (int i = 0; i < all.n; i += all.plan.group) {
        const int len = std::min(all.plan.group, all.n - i);
        // Between two groups no loan is out and no kernel runs: the safe point where a long prefill (1M tokens) can
        // give VRAM back (vram_tick itself checks the stream and the loans)
        if (i > 0) m.vram_tick();
        set_base(i);
        m.pf_prog_base_ = i;
        PrefillRun(m, {all.tokens + i, len, i + len < all.n ? all.tokens[i + len] : -1, all.nll ? &part : nullptr,
                       all.win0, all.emb, all.plan})
            .run();
        if (all.nll)
            for (int k = 0; k < len && i + k < all.n - 1; k++) (*all.nll)[i + k] = part[k];
    }
}

void Mega::prefill(const int32_t* tokens, int n, std::vector<float>* nll) {
    vram_tick();
    if (!PrefillRun::begin(*this, tokens, n)) return;
    if (nll) nll->assign(n > 1 ? n - 1 : 0, 0.0f);
    // Context of the drafter: the last dswa positions of the prompt (the window of its attention)
    const int64_t end = pos_ + n;
    const int64_t win0 = drafter_ ? (dswa_ > 0 ? std::max<int64_t>(pos_, end - dswa_) : pos_) : end;
    // History for the n-grams of the PLE, then the prompt
    const size_t ng = (size_t)h_.ple_ngram;
    std::vector<int32_t> seq(tokens_.end() - std::min(tokens_.size(), ng), tokens_.end());
    const size_t s0 = seq.size();
    seq.insert(seq.end(), tokens, tokens + n);
    const PfPlan plan = PrefillRun::choose_plan(*this, n, win0);
    pf_last_chunk_ = plan.chunk, pf_last_group_ = plan.layered ? plan.group : 0;
    if (tona::env::flag("TONA_PF_DEBUG"))
        std::fprintf(stderr, "Prefill %d tokens: %s, chunk %d, group %d, loan limit %.0f MB%s\n", n,
                     plan.layered ? "layer by layer" : "chunk by chunk", plan.chunk, plan.layered ? plan.group : 0,
                     pf_loan_cap() / 1e6, plan.wcache ? ", weight cache" : "");
    EmbedAhead ahead([&](size_t off, int m, float* out) { embed_tokens(seq.data(), s0 + off, m, out); }, n, plan.chunk,
                     (int)h_.d, pf_host_);
    int base = 0;  // first token of the running group
    const std::function<const float*(int, int)> emb = [&](int j0, int m) { return ahead.get(base + j0, m); };
    progress::prefill(0, n);
    pf_pass_ = 0;
    PrefillRun::extras_out(*this, plan, n);
    try {
        PrefillRun::run_groups(*this, {tokens, n, -1, nll, win0, &emb, plan}, [&](int at) { base = at; });
    } catch (...) {
        ahead.join();
        try {
            pf_return_loans();
        } catch (const std::exception& e) {  // the original exception counts
            std::fprintf(stderr, "%s\n", e.what());
        }
        throw;
    }
    ahead.join();
    PrefillRun::extras_back(*this);
}

void Mega::inject_draft(const float* feat_rows, int64_t p0, int m) {
    const size_t nf = feat_layers_.size() * h_.d;
    HIPCHECK(hipMemcpyAsync(feat_, feat_rows, (size_t)m * nf * sizeof(float), hipMemcpyDeviceToDevice, nullptr));
    dparams_host_->pos = p0 + m;
    dparams_host_->nb = 0;
    dparams_host_->commit = m;
    launch_program({m, dinject_end_, d_dops_, d_dtasks_, d_dparams_, nullptr}, 0, dparams_host_);
    HIPCHECK(hipStreamSynchronize(nullptr));  // dparams_host_ is rewritten for the next chunk
    check_aborts();
}

namespace {

// Table rows of the PLE n-grams of tokens off ... off + n - 1: heads rows per token
std::vector<int64_t> ple_rows(const NgramInput& in, uint32_t heads, const int32_t* seq, size_t off, int n) {
    std::vector<int64_t> rows((size_t)n * heads);
    for (int b = 0; b < n; b++) in.rows_at(seq, (int64_t)(off + b), &rows[(size_t)b * heads]);
    return rows;
}

}  // namespace

// Rows of the n-gram table gathered through io_uring (buffered, POSIX_FADV_RANDOM): 32,768 random rows per chunk of
// 2,048 tokens take 13 ms from the page cache and 105 ms from the NVMe. Through the mapping it was 4 and ~400 ms
// (tona-ple-io), and 600-1,700 ms cold next to the GPU in the prefill: every page fault blocks its thread.
// The tokens are independent, so even the 4 to 8 of a decode pass go to the threads (from 2 on): 0.30 ms of the serial
// loop per cycle, research/measurements/2026-11-17-b1-cycle-breakdown.md.
void Mega::embed_tokens(const int32_t* seq, size_t off, int n, float* out) const {
    const int64_t d = h_.d, ng = h_.ple_ngram, pd = h_.ple_dim;
    const uint32_t heads = h_.ple_heads;
    const TensorInfo& et = *m_.embedding_table();
    const int64_t erb = row_bytes_of(et.type, d);
    if (!ple_table_) {                 // one plain embedding row per token
        backend::OmpErrors plain_err;  // dequantize_row throws for a type it does not know
#pragma omp parallel for schedule(static) if (n >= 2)
        for (int b = 0; b < n; b++)
            plain_err.run(
                [&] { dequantize_row(et.type, et.data + (int64_t)seq[off + b] * erb, out + (int64_t)b * d, d); });
        plain_err.rethrow();
        return;
    }
    const TensorInfo& pt = *ple_table_;
    const int64_t prb = row_bytes_of(pt.type, pd);
    if (ng < 2 || ng > 8) throw std::runtime_error("gpu::Mega: PLE n-grams above 8");
    std::call_once(ple_io_once_, [&] {
        if (const char* e = tona::env::text("TONA_PLE_IO"); e && std::string(e) == "mmap") return;
        try {
            ple_io_ =
                std::make_unique<DirectRows>(m_.gguf().file_path(pt.file), m_.gguf().file_offset(pt), (uint32_t)prb);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "PLE table through the mapping (io_uring: %s)\n", e.what());
        }
    });
    backend::HostTrace::mark("emb_start");
    const std::vector<int64_t> rows = ple_rows(*m_.ngram_input(), heads, seq, off, n);
    backend::HostTrace::mark("emb_rows");
    std::vector<uint8_t> raw;
    if (ple_io_) {
        raw.resize(rows.size() * prb);
        ple_io_->read(rows.data(), rows.size(), raw.data());
    }
    backend::RegionTrace rt("emb_enter", "emb_worker_start", "emb_first_done", "emb_exit", n);
    rt.enter();
    backend::OmpErrors err;  // dequantize_row throws for a type it does not know
#pragma omp parallel for schedule(static) if (n >= 2)
    for (int b = 0; b < n; b++) {
        rt.row_begin(b);
        err.run([&] {
            float* in = out + (int64_t)b * 2 * d;
            dequantize_row(et.type, et.data + (int64_t)seq[off + b] * erb, in, d);
            for (uint32_t h = 0; h < heads; h++) {
                const size_t i = (size_t)b * heads + h;
                const uint8_t* src = ple_io_ ? raw.data() + i * prb : pt.data + rows[i] * prb;
                dequantize_row(pt.type, src, in + d + h * pd, pd);
            }
        });
        rt.row_end(b);
    }
    rt.leave();
    err.rethrow();
}

}  // namespace tona::gpu
