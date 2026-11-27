// The Backend of the Metal path: passes of up to BMAX tokens, TONA_SEQS slots, save points and stashes, prefill in
// chunks, the MTP head. What it cannot do yet throws, as the interface allows
// (docs/backends/README.md#interface-contract): DFlash.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "backend/env.h"
#include "backend/host_logits.h"
#include "backend/metal/device.h"
#include "backend/metal/metal.h"
#include "backend/metal/runner.h"
#include "backend/metal/weights.h"

namespace tona::metal {

namespace {

[[noreturn]] void unsupported(const char* what) {
    throw std::runtime_error(std::string("Metal backend: ") + what + " is not supported yet");
}

std::string device_name(id<MTLDevice> dev) {
    std::string n = "metal-";
    const char* name = dev.name.UTF8String;
    for (const char* p = name ? name : ""; *p; p++) n += *p == ' ' ? '-' : (char)std::tolower((unsigned char)*p);
    return n;
}

// TONA_SEQS slots (1 ... SMAX, default 1: every slot holds a KV cache for max_ctx)
int seqs_from_env() { return (int)std::clamp<int64_t>(env::integer("TONA_SEQS", 1), 1, backend::SMAX); }

// TONA_PREFILL_CHUNK tokens per prefill pass, a multiple of 32 (whole tiles of the scratch), 0 ... 4096; 0: prefill
// through forward. Default 2048: 190 tok/s against 135 with 512 at a 2048-token prompt, for 0.9 GB more scratch
// (research/measurements/2026-11-11-mac-prefill-gemm64.md).
int chunk_from_env() { return (int)std::clamp<int64_t>(env::integer("TONA_PREFILL_CHUNK", 2048) / 32 * 32, 0, 4096); }

// Pass costs on the M1 Max with the Coder (variant DM) and the head cut to 164 experts, 512 tokens of context, with
// the routed experts grouped by expert (research/measurements/2026-11-12-mac-moe-grouped.md; step and fill from
// 2026-11-12-mac-verify-profile.md): every token past the first adds ~16 ms here against ~6 on the reference, so the
// reference's costs push the width far past its best. The fill grows with the width (2.8 to
// 7.2 ms); 4.0 is its value at width 3. Across conversations the pass of N tokens costs about the same as one of
// width N (no measurement of forward_multi yet). With TONA_METAL_MMA every chunk of 4 tokens (MV_BMAX) costs 6.7 to
// 7.2 ms less, W8's two chunks 11.7 ms (research/measurements/2026-11-12-mac-mma-pass-costs.md). DM-T (Q6_K layer
// matrices as TONA_Q6_K_T) with MMA: 1.5 to 6.2 ms below DM at every width (2026-11-12-mac-dmt-pass-costs.md).
// TONA_METAL_MMA_WIDE (one MMA dispatch for up to 8 tokens): W5 to W8 12.9 to 21.2 ms below the 4-token chunks
// (2026-11-13-mac-mma-wide-pass-costs.md). TONA_METAL_MV5 (dense f32 mat-vecs of up to 5 tokens): W5 4.4 to 4.8 ms,
// W6 0.7 to 1.3 ms below that, the other widths unchanged (2026-11-13-mac-mv8.md).
backend::PassCosts pass_costs_of(const double (&ms)[backend::BMAX + 1]) {
    backend::PassCosts p;
    std::copy(ms, ms + backend::BMAX + 1, p.pass);
    std::copy(ms, ms + backend::BMAX + 1, p.multi);
    p.step = 2.13, p.fill = 4.0;
    return p;
}

const backend::PassCosts& metal_pass_costs(bool tiled) {
    static const backend::PassCosts f32 = pass_costs_of({0, 44.5, 61.2, 78.5, 94.2, 127.4, 141.6, 159.4, 173.1});
    static const backend::PassCosts mma = pass_costs_of({0, 44.5, 61.2, 78.5, 87.0, 120.5, 134.9, 152.6, 161.4});
    static const backend::PassCosts mma_tiled = pass_costs_of({0, 42.4, 59.7, 75.0, 85.0, 117.0, 131.8, 146.4, 157.0});
    static const backend::PassCosts wide = pass_costs_of({0, 44.5, 61.2, 78.5, 87.0, 106.3, 118.3, 131.4, 144.4});
    static const backend::PassCosts wide_tiled = pass_costs_of({0, 42.4, 59.7, 75.0, 85.0, 104.1, 115.7, 127.4, 139.8});
    static const backend::PassCosts mv5 = pass_costs_of({0, 44.5, 61.2, 78.5, 87.0, 101.5, 117.6, 131.4, 144.4});
    static const backend::PassCosts mv5_tiled = pass_costs_of({0, 42.4, 59.7, 75.0, 85.0, 99.7, 114.4, 127.4, 139.8});
    if (!mma_from_env()) return f32;
    if (mma_tokens_from_env() > 4 && mv5_from_env()) return tiled ? mv5_tiled : mv5;
    if (mma_tokens_from_env() > 4) return tiled ? wide_tiled : wide;
    return tiled ? mma_tiled : mma;
}

// Any matrix in a tile type (model/gguf.cpp allows TONA_Q6_K_T for dense layer matrices only)
bool has_tiles(const Qwen4ExpModel& m) {
    const auto& t = m.gguf().tensors();
    return std::any_of(t.begin(), t.end(), [](const TensorInfo& i) { return kernel_type(i.type) == "Q6KT"; });
}

class MetalBackend final : public backend::Backend {
public:
    MetalBackend(const Qwen4ExpModel& m, const backend::CreateOptions& o)
        : w_(weights(dev_, m, o.mtp)),
          run_(dev_, w_, m, RunnerOptions{o.max_ctx, seqs_from_env(), chunk_from_env(), o.mtp}),
          max_ctx_(o.max_ctx),
          vocab_(m.hp().vocab),
          tiled_(has_tiles(m)) {
        if (o.drafter || !o.feat_layers.empty()) unsupported("the DFlash drafter");
        if (Arena* a = w_.arena()) a->load(dev_, w_.mapped_bytes(), w_.mapped_bytes() + dev_.allocated());
        info_.name = device_name(dev_.dev());
        info_.vram_bytes = w_.mapped_bytes() + dev_.allocated();
        info_.kv_bytes = run_.seqs() * (3 * run_.layout().recurrent + max_ctx_ * run_.layout().kv_row());
    }

    backend::Info info() const override { return info_; }
    const backend::Stats& stats() const override { return stats_; }

    const float* forward(const int32_t* tokens, int nb) override {
        const int slot = run_.slot();
        return forward_multi(&slot, &nb, 1, tokens);
    }
    const float* forward_multi(const int* slots, const int* n, int k, const int32_t* tok) override {
        const float* lg = run_.forward(slots, n, k, tok, true);
        stats_.launches += k;
        stats_.nan_rows = run_.router_nans(), stats_.nan_reported = run_.nan_reported();
        return lg;
    }
    void commit(int m) override { run_.commit(m); }

    // Prefill kernels in chunks; with TONA_PREFILL_CHUNK=0 decode passes of up to BMAX tokens, all accepted, so no
    // backup of the recurrent state
    void prefill(const int32_t* tokens, int n, std::vector<float>* nll) override {
        if (run_.chunk() > 0) return run_.prefill(tokens, n, nll);
        if (nll) nll->assign(n > 1 ? n - 1 : 0, 0.0f);
        const int slot = run_.slot();
        for (int i = 0; i < n; i += BMAX) {
            const int nb = std::min(BMAX, n - i);
            const float* lg = run_.forward(&slot, &nb, 1, tokens + i, false);
            for (int t = 0; nll && t < nb && i + t + 1 < n; t++)
                (*nll)[i + t] = neg_log_p(lg + (size_t)t * vocab_, tokens[i + t + 1]);
            run_.commit(nb);
        }
    }
    int prefill_chunk() const override { return run_.chunk(); }
    void reset() override { run_.reset(); }
    // A failed pass may have written part of the state: start over (the generator re-sends the prompt)
    void recover() override { run_.reset(); }
    int64_t pos() const override { return run_.cur().pos; }
    int64_t max_ctx() const override { return max_ctx_; }
    const std::vector<int32_t>& tokens() const override { return run_.cur().tokens; }

    void save_state() override { run_.save_state(); }
    void load_state() override { run_.load_state(); }
    bool has_state() const override { return run_.cur().snap; }
    const std::vector<int32_t>& snap_tokens() const override { return run_.cur().snap_tokens; }
    const std::vector<int32_t>& snap_tokens_of(int s) const override { return run_.at(s).snap_tokens; }
    void stash(backend::HostSnap& s) const override { run_.stash(s); }
    size_t stash_bytes() const override { return run_.stash_bytes(); }
    void unstash(const backend::HostSnap& s) override { run_.unstash(s); }
    bool snap_fits(const backend::HostSnap& s) const override { return run_.snap_fits(s); }
    size_t state_bytes() const override { return run_.layout().recurrent + 4; }
    size_t dring_bytes() const override { return 0; }
    size_t kv_row_bytes() const override { return run_.layout().kv_row(); }

    int seqs() const override { return run_.seqs(); }
    int slot() const override { return run_.slot(); }
    void use_slot(int s) override { run_.use(s); }

    void draft(int, const float*, int) override { unsupported("the DFlash drafter"); }
    const float* draft_logits() const override { return nullptr; }
    const float* draft_gate() const override { return nullptr; }
    bool has_mtp() const override { return run_.has_mtp(); }
    const backend::PassCosts* pass_costs() const override { return &metal_pass_costs(tiled_); }
    backend::MtpPick mtp_fill(int32_t next) override { return counted(run_.mtp_fill(next)); }
    backend::MtpPick mtp_step(int32_t tok) override { return counted(run_.mtp_step(tok)); }
    // One slot after the other (the reference runs them in one pass)
    void mtp_step_multi(const int* slots, const int32_t* toks, int k, backend::MtpPick* out) override {
        const int was = run_.slot();
        for (int i = 0; i < k; i++) run_.use(slots[i]), out[i] = mtp_step(toks[i]);
        run_.use(was);
    }
    // The logits are in shared memory after every pass
    void mtp_keep_logits(bool) override {}
    const float* mtp_logits() const override { return run_.has_mtp() ? run_.mtp_logits() : nullptr; }
    size_t mtp_rows() const override { return run_.has_mtp() ? run_.mtp_vocab().rows() : 0; }
    int32_t mtp_token(size_t r) const override { return run_.has_mtp() ? run_.mtp_vocab().token(r) : -1; }

private:
    static Weights weights(Device& dev, const Qwen4ExpModel& m, const MtpHead* mtp) {
        Weights w(dev, m);
        // Embedding and LM head of the head's file are the target's (MtpHead): 1.35 GB of the Q8_0 file left unmapped
        if (mtp) w.add(mtp->gguf, {"token_embd.weight", "output.weight"});
        const char* pin = std::getenv("TONA_METAL_PIN");
        if (pin && *pin) std::fprintf(stderr, "Metal: %s pinned %.2f GB\n", pin, (double)w.pin(pin) / 1e9);
        dev.keepalive_reads(w.buffers());
        return w;
    }
    backend::MtpPick counted(backend::MtpPick p) {
        stats_.mtp_passes = run_.mtp_counts().passes, stats_.mtp_stale = run_.mtp_counts().stale;
        stats_.mtp_dyn = run_.mtp_vocab().loaded();
        stats_.nan_rows = run_.router_nans(), stats_.nan_reported = run_.nan_reported();
        return p;
    }

    float neg_log_p(const float* lg, int32_t next) const { return (float)backend::neg_log_p(lg, vocab_, next); }

    Device dev_;
    Weights w_;
    Runner run_;
    int64_t max_ctx_;
    uint32_t vocab_;
    bool tiled_;  // the file holds TONA_Q6_K_T matrices (its own pass costs)
    backend::Info info_;
    backend::Stats stats_;
};

}  // namespace

bool probe(std::string& why) {
    @autoreleasepool {
        if (MTLCreateSystemDefaultDevice()) return true;
    }
    why = "no Metal device";
    return false;
}

std::set<uint32_t> weight_types() {
    std::set<uint32_t> out;
    for (const auto& [type, name] : kernel_types()) out.insert(type);
    return out;
}

std::unique_ptr<backend::Backend> create(const Qwen4ExpModel& m, const backend::CreateOptions& o) {
    return std::make_unique<MetalBackend>(m, o);
}

}  // namespace tona::metal
