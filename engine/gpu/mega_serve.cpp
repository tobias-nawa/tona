// Running passes: forward and forward_multi, serving the mailbox while the kernel waits at the routers, yields, aborts
// and their diagnostics.
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>

#include "backend/env.h"
#include "backend/host_trace.h"
#include "backend/nan_guard.h"
#include "gpu/evict_counter.h"
#include "gpu/mega.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_tap.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

std::vector<std::pair<std::string, double>> Mega::profile(bool reset) {
    std::vector<std::pair<std::string, double>> out;
    if (!prof_tokens_) return out;
    std::unordered_map<std::string, double> by;
    std::unordered_map<std::string, int> n;
    for (size_t o = 0; o < ops_.size(); o++) n[labels_[o]]++;
    // TONA_PROF_LAYERS: one row per step (label#k, k-th step of this label), and "nu#l", "gd#l": mean experts of
    // layer l computed by the GPU and, of them, read over PCIe, times 1000
    static const bool layers = env::flag("TONA_PROF_LAYERS");
    std::unordered_map<std::string, int> seen;
    for (size_t o = 0; o < ops_.size(); o++)
        by[layers ? std::format("{}#{}", labels_[o], seen[labels_[o]]++)
                  : std::format("{} ({}x)", labels_[o], n[labels_[o]])] += prof_us_[o] / prof_tokens_;
    auto& nu = prof_gpu_experts();
    for (size_t l = 0; layers && l < nu.size(); l++)
        by[std::format("{}#{}", l % 2 ? "gd" : "nu", l / 2)] = 1e3 * nu[l] / prof_tokens_;
    out.assign(by.begin(), by.end());
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    if (reset) std::fill(prof_us_.begin(), prof_us_.end(), 0.0), prof_tokens_ = 0, prof_gpu_experts().clear();
    return out;
}

ChainDiag Mega::diag() {
    unsigned long long acc[3];
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(acc, static_cast<uint8_t*>(d_bar_) + 32, sizeof acc, hipMemcpyDeviceToHost));
    const double us = 1e3 / wall_khz_;
    return {(int64_t)acc[0], (double)acc[1] * us, (double)acc[2] * us};
}

// Heartbeat for scripts/gpu-watch.sh: TONA_HEARTBEAT=file, one line per layer (phase, position, layer)
void Mega::beat(const char* phase, int layer) {
    static const char* path = env::text("TONA_HEARTBEAT");
    if (!path) return;
    if (hb_fd_ < 0) hb_fd_ = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (hb_fd_ < 0) return;
    const long ms =
        (long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    char buf[96];
    const int n =
        std::snprintf(buf, sizeof buf, "%ld %s pos=%lld layer=%d%40s\n", ms, phase, (long long)pos_, layer, "");
    (void)!pwrite(hb_fd_, buf, (size_t)std::min<int>(n, 95), 0);
}

// Without CPU mode the union must fit into the slots; with it the CPU or the GPU direct computes the rest
void Mega::check_slots_fit(int nb) const {
    if (!cpu_mode_ && nb * (int)h_.n_expert_used > slots_ && !fake_experts())
        throw std::runtime_error(std::format("gpu::Mega: {} slots too few for {} tokens per pass", slots_, nb));
}

void Mega::check_tokens(const int32_t* tokens, int nb) const {
    for (int b = 0; b < nb; b++)
        if (tokens[b] < 0 || tokens[b] >= (int32_t)h_.vocab)
            throw std::runtime_error("gpu::Mega: token outside the vocabulary");
}

// Embedding and PLE vector per token on the CPU (rows depend only on the tokens); the n-gram context of token b
// reaches into the earlier tokens of the pass
void Mega::stage_embeddings(const int32_t* tokens, int nb) {
    const int64_t n = h_.ple_ngram;
    std::vector<int32_t> seq(tokens_.end() - std::min<size_t>(tokens_.size(), n), tokens_.end());
    const size_t s0 = seq.size();
    seq.insert(seq.end(), tokens, tokens + nb);
    embed_tokens(seq.data(), s0, nb, in_host_);
    HIPCHECK(hipMemcpyAsync(e_, in_host_, (size_t)nb * h_.in_row() * sizeof(float), hipMemcpyHostToDevice, nullptr));
}

// Logits (and features) to the host after the kernel is done; true if a yield after the last router made it run on
float* Mega::fetch_logits(int nb) {
    float* dst = nb == 1 ? logits1_.data() : logits_host_.data();
    auto fetch = [&] {
        HIPCHECK(hipMemcpyAsync(dst, logits_, (size_t)nb * h_.vocab * sizeof(float), hipMemcpyDeviceToHost, nullptr));
        if (feat_)
            HIPCHECK(hipMemcpyAsync(feat_host_.data(), feat_, (size_t)nb * feat_layers_.size() * h_.d * sizeof(float),
                                    hipMemcpyDeviceToHost, nullptr));
    };
    fetch();
    beat("forward-end", -1);
    evict_sync(mbox_host_);
    if (finish_yields()) {
        fetch();
        evict_sync(mbox_host_);
    }
    return dst;
}

// TONA_MEGA_PROF_EVERY=n (with TONA_MEGA_PROF): the profile of the last n passes to stderr, by kind of step, with the
// mean pass width; the server has no other way to show it (tona-gpu-gen is Coder-only)
static void report_profile(const std::vector<std::pair<std::string, double>>& rows, int64_t n, double width) {
    // TONA_PROF_LAYERS: the rows nu#l and gd#l are expert counts times 1000, not times; they stay out of the total
    // and every row is printed (a per-layer step is far below 1 % of a pass)
    static const bool layers = env::flag("TONA_PROF_LAYERS");
    auto is_count = [](const std::string& name) { return name.starts_with("nu#") || name.starts_with("gd#"); };
    double total = 0;
    for (const auto& [name, us] : rows)
        if (!is_count(name)) total += us;
    std::fprintf(stderr, "MEGA_PROF %lld passes, mean width %.2f: %.2f ms per pass\n", (long long)n, width,
                 total / 1e3);
    for (const auto& [name, us] : rows)
        if (layers || us > 0.01 * total)
            std::fprintf(stderr, "MEGA_PROF   %-28s %8.0f us %5.1f %%\n", name.c_str(), us, 100 * us / total);
}

// TONA_MEGA_PROF: time per step of the pass
void Mega::record_step_times(int nb) {
    if (prof_us_.empty()) return;
    std::vector<unsigned long long> t(2 * ops_.size());
    HIPCHECK(hipMemcpy(t.data(), d_prof_, t.size() * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    const double us = 1e3 / wall_khz_;
    unsigned long long prev = t[0];
    for (size_t o = 0; o < ops_.size(); o++) {
        prof_us_[o] += (double)(t[2 * o + 1] - prev) * us;
        prev = t[2 * o + 1];
    }
    prof_tokens_++;
    prof_width_sum_ += nb;
    static const int64_t every = env::integer("TONA_MEGA_PROF_EVERY", 0);
    if (every > 0 && prof_tokens_ >= every) {
        report_profile(profile(), prof_tokens_, (double)prof_width_sum_ / (double)prof_tokens_);
        profile(true);
        prof_width_sum_ = 0;
    }
    log_slow_steps(t);
}

// TONA_BAR_PROF: arrival of the blocks at the barriers of the pass
void Mega::write_barrier_trace(int nb) {
    if (!arr_fp_) return;
    const int nblk = kblocks_[kix(nb)];
    std::vector<unsigned long long> t(2 * ops_.size()), a(ops_.size() * nblk);
    HIPCHECK(hipMemcpy(t.data(), d_prof_, t.size() * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(a.data(), d_arr_, a.size() * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    const int32_t hd[4] = {(int32_t)ops_.size(), nblk, nb, wall_khz_};
    std::fwrite(hd, sizeof hd, 1, arr_fp_);
    std::fwrite(t.data(), sizeof(t[0]), t.size(), arr_fp_);
    std::fwrite(a.data(), sizeof(a[0]), a.size(), arr_fp_);
    std::fflush(arr_fp_);
}

// Bookkeeping of the sequence before a pass of nb tokens at pos_
void Mega::begin_pass(const int32_t* tokens, int nb) {
    params_host_->pos = pos_;
    params_host_->seq = (int64_t)seq_;
    params_host_->nb = nb;
    pass_nb_ = nb;
    mtrk_.on_target_pass(pos_, tokens[0]);
    fw_pos_ = pos_, fw_m_ = 0, trunk_ok_ = true;
    params_host_->commit = pending_commit_;
    params_host_->parity = parity_;
    params_host_->nseq = 0, params_host_->arr = d_arr_;
    for (int s = 0; s < n_seqs_; s++)
        if (s != cur_slot_) seq_host_[s].trunk_ok = false;  // h_trunk_ now belongs to this pass
    fw_row0_ = 0;
    multi_pass_ = false;
}

const float* Mega::forward(const int32_t* tokens, int nb) {
    if (nb < 1 || nb > MEGA_BMAX) throw std::runtime_error(std::format("gpu::Mega: {} tokens per pass", nb));
    if (awaiting_commit_) throw std::runtime_error("gpu::Mega: forward without commit of the previous pass");
    if (pos_ + nb > max_ctx_) throw std::runtime_error("gpu::Mega: context full");
    vram_tick();
    check_slots_fit(nb);
    batch_.assign(tokens, tokens + nb);
    check_tokens(tokens, nb);
    backend::HostTrace::mark("fw_begin", nb);
    stage_embeddings(tokens, nb);
    begin_pass(tokens, nb);
    backend::HostTrace::mark("fw_staged");
    // one kernel for the whole pass; the CPU serves the mailbox at every router
    launch(nb);
    backend::HostTrace::mark("fw_launched");
    serve_pass(nb);
    backend::HostTrace::mark("fw_served");
    const float* dst = fetch_logits(nb);
    backend::HostTrace::mark("fw_fetched");
    HIPCHECK(hipGetLastError());
    record_step_times(nb);
    write_barrier_trace(nb);
    if (tap_) tap_->dump(nb, pos_);
    check_aborts();
    for (LayerCache& C : cache_) C.installed.clear();  // arrived: a later recover does not discard them
    awaiting_commit_ = true;
    return dst;
}

// Serve the mailbox of the running pass: the selection at every router, in CPU mode the CPU's experts
void Mega::serve_pass(int nb) {
    for (uint32_t l = 0; l < h_.n_layer; l++) {
        const uint32_t want = (uint32_t)(seq_ + l + 1);
        const auto t0 = std::chrono::steady_clock::now();
        for (uint64_t spin = 0; __atomic_load_n(&mbox_host_->gpu_seq, __ATOMIC_ACQUIRE) != want; spin++) {
            __builtin_ia32_pause();
            if (resume_if_yielded(seq_)) continue;
            evict_serve(mbox_host_);
            if (__atomic_load_n(&mbox_host_->aborted, __ATOMIC_ACQUIRE) == (uint32_t)(seq_ + 63)) {
                beat("gpu-aborted", (int)l);
                (void)hipDeviceSynchronize();  // the kernel ends without the CPU (all blocks see bar[2])
                check_aborts();
                throw std::runtime_error(
                    std::format("gpu::Mega: pass aborted before router {} at position {}{}", l, pos_, abort_info()));
            }
            if ((spin & 0xffff) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
                // scripts/gpu-watch.sh reacts to this text
                beat("waiting-for-gpu", (int)l);
                (void)hipDeviceSynchronize();
                throw std::runtime_error(
                    std::format("gpu::Mega: no message from router {} at position {}{}", l, pos_, abort_info()));
            }
        }
        stats_.wait_s += seconds_since<std::chrono::steady_clock>(t0);
        backend::HostTrace::mark("router_arrived", (int)l);
        // NaN router logits arrive clamped (op_host_wait); the count tells that something before the router is wrong
        //. The mailbox is written before gpu_seq, which the loop above has acquired.
        if (mbox_host_->nan_router || mbox_host_->nan_pred) {
            const std::string where = std::format("layer {}, position {}", l, pos_);
            tona::nan::report("router logits", mbox_host_->nan_router, where);
            tona::nan::report("predicted router logits", mbox_host_->nan_pred, where);
        }
        beat("forward", (int)l);
        const auto t1 = std::chrono::steady_clock::now();
        const CpuStall stall;
        select(l, nb);
        __atomic_store_n(&mbox_host_->cpu_seq, want, __ATOMIC_RELEASE);
        backend::HostTrace::mark("selected", (int)l);
        stats_.select_s += seconds_since<std::chrono::steady_clock>(t1);
        if (cpu_mode_) cpu_experts(l, nb);
        backend::HostTrace::mark("cpu_experts_done", (int)l);
        stall.report("router", (int)l, pos_);
    }
    seq_ += 64;
}

// After the sync of the pass: if the kernel yielded after the last router (the CPU finished just after), run the
// rest; true then, and the copies are to be fetched again. The CPU has served all layers
bool Mega::finish_yields() {
    bool any = false;
    while (resume_if_yielded(seq_ - 64)) {  // serve_pass has already advanced seq_
        any = true;
        evict_sync(mbox_host_);
    }
    return any;
}

// Arguments of forward_multi; returns the total number of tokens
int Mega::check_multi(const int* slots, const int* n, int k, const int32_t* tok) const {
    int nb = 0;
    for (int i = 0; i < k; i++) {
        if (n[i] < 1) throw std::runtime_error("gpu::Mega: sequence without tokens in the pass");
        for (int j = 0; j < i; j++)
            if (slots[j] == slots[i]) throw std::runtime_error("gpu::Mega: slot twice in the pass");
        nb += n[i];
    }
    if (nb > MEGA_BMAX) throw std::runtime_error(std::format("gpu::Mega: {} tokens per pass", nb));
    check_slots_fit(nb);
    check_tokens(tok, nb);
    return nb;
}

// Rows, positions and embeddings of every sequence of the pass into the parameters and the host buffer
void Mega::stage_multi(const int* slots, const int* n, int k, const int32_t* tok) {
    const int64_t ng = h_.ple_ngram;
    MegaParams& P = *params_host_;
    std::memset(P.snb, 0, sizeof P.snb);
    for (int i = 0, b0 = 0; i < k; b0 += n[i], i++) {
        use_slot(slots[i]);
        if (awaiting_commit_) throw std::runtime_error("gpu::Mega: forward without commit of the previous pass");
        if (pos_ + n[i] > max_ctx_) throw std::runtime_error("gpu::Mega: context full");
        std::vector<int32_t> hist(tokens_.end() - std::min<size_t>(tokens_.size(), ng), tokens_.end());
        const size_t s0 = hist.size();
        hist.insert(hist.end(), tok + b0, tok + b0 + n[i]);
        embed_tokens(hist.data(), s0, n[i], in_host_ + (int64_t)b0 * h_.in_row());
        batch_.assign(tok + b0, tok + b0 + n[i]);
        pass_nb_ = n[i];
        mtrk_.on_target_pass(pos_, batch_[0]);
        fw_pos_ = pos_, fw_m_ = 0, fw_row0_ = b0, trunk_ok_ = true;
        const int s = cur_slot_;
        P.sb0[s] = b0, P.snb[s] = n[i], P.scommit[s] = pending_commit_, P.sparity[s] = parity_, P.spos[s] = pos_;
    }
    for (int s = 0; s < n_seqs_; s++)
        if (P.snb[s] == 0) (s == cur_slot_ ? trunk_ok_ : seq_host_[s].trunk_ok) = false;
}

void Mega::fetch_multi_logits(int nb) {
    auto fetch = [&] {
        HIPCHECK(hipMemcpyAsync(logits_host_.data(), logits_, (size_t)nb * h_.vocab * sizeof(float),
                                hipMemcpyDeviceToHost, nullptr));
    };
    fetch();
    beat("forward-end", -1);
    evict_sync(mbox_host_);
    if (!finish_yields()) return;
    fetch();
    evict_sync(mbox_host_);
}

const float* Mega::forward_multi(const int* slots, const int* n, int k, const int32_t* tok) {
    if (k < 1 || k > n_seqs_) throw std::runtime_error(std::format("gpu::Mega: {} sequences in one pass", k));
    if (k == 1) {
        use_slot(slots[0]);
        return forward(tok, n[0]);
    }
    const int nb = check_multi(slots, n, k, tok);
    vram_tick();
    MegaParams& P = *params_host_;
    stage_multi(slots, n, k, tok);
    HIPCHECK(hipMemcpyAsync(e_, in_host_, (size_t)nb * h_.in_row() * sizeof(float), hipMemcpyHostToDevice, nullptr));
    for (const XPatch& x : xpatch_) xops_host_[x.op].p[x.j] = x.base + (int64_t)P.sb0[x.seq] * x.row;
    HIPCHECK(hipMemcpyAsync(d_xops_, xops_host_, xops_.size() * sizeof(MegaOp), hipMemcpyHostToDevice, nullptr));
    P.pos = P.spos[slots[0]], P.seq = (int64_t)seq_, P.nb = nb, P.nseq = k, P.commit = 0, P.parity = 0;
    P.arr = nullptr;  // TONA_BAR_PROF measures only the program of one sequence
    launch_program({nb, (int)xops_.size(), d_xops_, d_tasks_, d_params_, nullptr}, 0, params_host_);
    multi_pass_ = true;
    serve_pass(nb);
    fetch_multi_logits(nb);
    HIPCHECK(hipGetLastError());
    check_aborts();
    for (LayerCache& C : cache_) C.installed.clear();
    for (int i = 0; i < k; i++) {
        use_slot(slots[i]);
        awaiting_commit_ = true;
    }
    stats_.multi_passes++;
    return logits_host_.data();
}

void Mega::check_aborts() {
    unsigned long long aborts;
    HIPCHECK(hipMemcpy(&aborts, static_cast<uint8_t*>(d_bar_) + 32, sizeof aborts, hipMemcpyDeviceToHost));
    if ((int64_t)aborts > aborts_seen_)
        throw std::runtime_error(
            std::format("gpu::Mega: barrier aborted (time limit) at position {}{}", pos_, abort_info()));
}

// TONA_MEGA_PROF: longest step of the aborted pass (the timestamps up to k are from this pass)
std::string Mega::slowest_step_info(unsigned k) const {
    if (!d_prof_ || k < 1 || labels_.size() < ops_.size()) return "";
    std::vector<unsigned long long> t(2 * ops_.size());
    if (hipMemcpy(t.data(), d_prof_, t.size() * sizeof(unsigned long long), hipMemcpyDeviceToHost) != hipSuccess)
        return "";
    const double ms = 1.0 / wall_khz_;
    size_t worst = 0;
    double wmax = 0;
    for (size_t i = 1; i < k && i < ops_.size(); i++) {
        const double dt = (double)(long long)(t[2 * i + 1] - t[2 * i - 1]) * ms;
        if (dt > wmax) wmax = dt, worst = i;
    }
    return std::format(", longest step {} {} {:.1f} ms, pass up to there {:.1f} ms", worst, labels_[worst], wmax,
                       (double)(long long)(t[2 * (k - 1) + 1] - t[0]) * ms);
}

// Stragglers: blocks that were not at the furthest step (0xffffffff: never started)
std::string Mega::straggler_info(const std::vector<std::string>& labels) const {
    std::vector<unsigned> pr(256);
    if (hipMemcpy(pr.data(), static_cast<uint8_t*>(d_bar_) + 64, pr.size() * sizeof(unsigned), hipMemcpyDeviceToHost) !=
        hipSuccess)
        return "";
    const int nblk = kblocks_[1];
    unsigned top = 0;
    for (int b = 0; b < nblk; b++)
        if (pr[b] != 0xffffffffu) top = std::max(top, pr[b]);
    std::string lag;
    for (int b = 0; b < nblk; b++) {
        if (pr[b] == top) continue;
        if (pr[b] == 0xffffffffu)
            lag += std::format(" B{}:never started", b);
        else
            lag += std::format(" B{}:{} {}", b, pr[b], pr[b] < labels.size() ? labels[pr[b]] : "");
    }
    return std::format(", blocks at step {}", top) + (lag.empty() ? ", all equal" : ", stragglers" + lag);
}

std::string Mega::abort_info() const {
    unsigned o = 0;
    if (hipMemcpy(&o, static_cast<uint8_t*>(d_bar_) + 12, sizeof o, hipMemcpyDeviceToHost) != hipSuccess || o == 0)
        return ", no step recorded" + cause_info();
    const auto& labels = last_drafter_ ? dlabels_ : last_multi_ ? xlabels_ : labels_;
    const auto& ops = last_drafter_ ? dops_ : last_multi_ ? xops_ : ops_;
    const unsigned k = o - 1;
    if (k >= ops.size()) return std::format(" (step {})", k);
    const std::string slow =
        (&ops == &ops_ ? slowest_step_info(k) + arrival_info(k) : "") + straggler_info(labels) + cause_info();
    return std::format(" (step {}: {} {}{})", k, kind_names[ops[k].kind], k < labels.size() ? labels[k] : "", slow);
}

void Mega::recover() {
    (void)hipDeviceSynchronize();
    // Prefetched experts of the aborted pass may not have arrived completely: take them back like the refilled ones
    auto drop = [&](LayerCache& C, int e) {
        auto it = C.where.find(e);
        if (it == C.where.end()) return;
        C.free_slots.push_back(it->second.first);
        C.erase(it);
    };
    if (pre_st_) {
        std::fill(pre_busy_.begin(), pre_busy_.end(), 0);
        for (size_t l = 0; l < pre_got_.size(); l++) {
            for (int e : pre_got_[l]) drop(cache_[l], e);
            pre_got_[l].clear();
        }
    }
    (void)hipGetLastError();
    unsigned long long aborts = 0;
    (void)hipMemcpy(&aborts, static_cast<uint8_t*>(d_bar_) + 32, sizeof aborts, hipMemcpyDeviceToHost);
    aborts_seen_ = (int64_t)aborts;
    // Experts this pass was only about to copy into their slots: may never have arrived
    for (LayerCache& C : cache_) {
        for (int e : C.installed) drop(C, e);
        C.installed.clear();
    }
    seq_ += 64;  // do not reuse the sequence numbers of the aborted pass
    awaiting_commit_ = false;
    snap_valid_ = false;
    reset();
}

void Mega::commit(int m) {
    if (!awaiting_commit_ || m < 1 || m > pass_nb_)
        throw std::runtime_error(std::format("gpu::Mega: commit({}) after a pass of {} tokens", m, batch_.size()));
    tokens_.insert(tokens_.end(), batch_.begin(), batch_.begin() + m);
    pos_ += m;
    fw_m_ = m;
    pending_commit_ = m;
    parity_ ^= 1;
    awaiting_commit_ = false;
}

const std::vector<float>& Mega::step(int32_t token) {
    forward(&token, 1);
    commit(1);
    step_out_.assign(logits1_.begin(), logits1_.end());  // copy on the CPU; the GPU writes only pinned memory
    return step_out_;
}

void Mega::check_draft_args(int m, const float* emb, int nb) const {
    if (!drafter_) throw std::runtime_error("gpu::Mega: no drafter");
    if (awaiting_commit_) throw std::runtime_error("gpu::Mega: draft before commit");
    if (m < 0 || m > (int)batch_.size() || m > pos_ || (emb && (nb < 2 || nb > MEGA_BMAX)) || pos_ + nb > max_ctx_)
        throw std::runtime_error(std::format("gpu::Mega: draft({}, {}) at position {}", m, nb, pos_));
}

// Logits and gate of the positions 1 ... nb - 1 of the block
void Mega::fetch_draft(int nb) {
    HIPCHECK(hipMemcpyAsync(dlog_host_.data(), dlog_ + h_.vocab, (size_t)(nb - 1) * h_.vocab * sizeof(float),
                            hipMemcpyDeviceToHost, nullptr));
    HIPCHECK(hipMemcpyAsync(dgate_host_.data(), dgate_ + drank_, (size_t)(nb - 1) * drank_ * sizeof(float),
                            hipMemcpyDeviceToHost, nullptr));
}

void Mega::draft(int m, const float* emb, int nb) {
    check_draft_args(m, emb, nb);
    if (!emb) nb = 0;
    if (m == 0 && nb == 0) return;
    dparams_host_->pos = pos_;
    dparams_host_->nb = nb;
    dparams_host_->commit = m;
    const int64_t d = h_.d;
    if (nb) {
        std::memcpy(in2_host_, emb, (size_t)nb * d * sizeof(float));
        HIPCHECK(hipMemcpyAsync(e2_, in2_host_, (size_t)nb * d * sizeof(float), hipMemcpyHostToDevice, nullptr));
    }
    launch_program({std::max(m, nb), nb ? (int)dops_.size() : dinject_end_, d_dops_, d_dtasks_, d_dparams_, nullptr},
                   m ? 0 : dinject_end_, dparams_host_);
    if (nb) fetch_draft(nb);
    beat("draft-end", -1);
    HIPCHECK(hipStreamSynchronize(nullptr));
    HIPCHECK(hipGetLastError());
    check_aborts();
}

}  // namespace tona::gpu
