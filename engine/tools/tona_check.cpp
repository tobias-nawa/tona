// Checks a backend against the CPU reference (cpu/ref.cpp, f32) step by step: every prompt token and then greedy
// continuation from the backend's own top-1, comparing the logits of each step (largest absolute difference, relative
// to the largest reference logit, and top-1). Backend-neutral: takes any backend the program links (--backend).
// With --widths SEED, the same tokens then run again in passes of random width (1 ... BMAX) with a random number of
// them accepted (commit(m < nb) rolls the rest back), a save point / load and a stash round trip on the way, and on
// up to two slots at once when the backend has them (TONA_SEQS); every logit row must equal the width-1 run bitwise
// (the contract of TONA_DETERMINISTIC).
// With --prefill SEED (backends with prefill_chunk() > 0): the first two thirds of the tokens through the prefill in
// one call and in random pieces must leave byte-equal states (stash; docs/use/prompt-cache.md), and the decode steps
// after the prefill are compared with the width-1 run (top-1, relative deviation, -log p of each prefilled token). With
// --ref N only the first N steps run on the CPU reference, the rest follow the backend's own top-1: on a real model the
// reference takes seconds per token, while the widths and prefill checks need long traces. Usage: tona-check MODEL.gguf
// "PROMPT" [--backend NAME|auto] [-n TOKENS] [--ctx N] [--widths SEED] [--prefill SEED]
//                   [--ref N] [--mtp MTP.gguf (--dump OUT | --golden FILE) [-n N] [--mtp-steps N]
//                   [--mtp-prompt forward|fill|prefill]]
// With --mtp the MTP head is checked instead (tools/tona_check_mtp.h): --dump writes a golden file of its picks and
// logits, --golden replays one, also one made by another backend, and compares.
// TONA_REF_TRACE_OUT=file (Ornith) writes the oracle's per-layer residual for tools/tap_compare.py (next to the GPU's
// TONA_TAP_OUT).
// Exit code 1 if a step's top-1 differs, a row of the widths run is not bit-identical or a cut changes the state.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/builtin.h"
#include "backend/host_logits.h"
#include "backend/registry.h"
#include "cpu/ref.h"
#include "model/model_registry.h"
#include "model/qwen4exp.h"
#include "model/tokenizer.h"
#include "tools/tona_check_mtp.h"

namespace {

struct Options {
    std::string model, prompt, backend = "auto";
    int n_gen = 8;
    int64_t ctx = 4096;
    int64_t widths_seed = -1;   // < 0: no widths run
    int64_t prefill_seed = -1;  // < 0: no prefill run
    int64_t ref_steps = -1;     // < 0: every step against the CPU reference
    std::string mtp = {}, dump = {}, golden = {};
    int mtp_steps = 3;
    std::string mtp_prompt = "forward";
};

Options parse(int argc, char** argv) {
    if (argc < 3)
        throw std::invalid_argument("usage: tona-check MODEL.gguf \"PROMPT\" [--backend NAME] [-n N] [--ctx N]");
    Options o{argv[1], argv[2]};
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--backend")
            o.backend = v;
        else if (a == "-n")
            o.n_gen = std::stoi(v);
        else if (a == "--ctx")
            o.ctx = std::stoll(v);
        else if (a == "--widths")
            o.widths_seed = std::stoll(v);
        else if (a == "--prefill")
            o.prefill_seed = std::stoll(v);
        else if (a == "--ref")
            o.ref_steps = std::stoll(v);
        else if (a == "--mtp")
            o.mtp = v;
        else if (a == "--dump")
            o.dump = v;
        else if (a == "--golden")
            o.golden = v;
        else if (a == "--mtp-steps")
            o.mtp_steps = std::stoi(v);
        else if (a == "--mtp-prompt")
            o.mtp_prompt = v;
        else
            throw std::invalid_argument("unknown option " + a);
    }
    return o;
}

int argmax(const float* v, size_t n) { return (int)(std::max_element(v, v + n) - v); }

struct Tally {
    int steps = 0, same_top1 = 0;
    double worst = 0;
};

// Compares one step and prints it; returns the backend's top-1
int compare(const float* be, const std::vector<float>& ref, int32_t token, Tally& t) {
    double max_diff = 0, max_ref = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        max_diff = std::max(max_diff, (double)std::fabs(be[i] - ref[i]));
        max_ref = std::max(max_ref, (double)std::fabs(ref[i]));
    }
    const int tb = argmax(be, ref.size()), tr = argmax(ref.data(), ref.size());
    const double rel = max_diff / std::max(max_ref, 1e-30);
    std::fprintf(stderr, "  step %d (token %d): max |d| %.4f (relative %.2e), top-1 backend %d CPU %d%s\n", t.steps,
                 token, max_diff, rel, tb, tr, tb == tr ? "" : "  <-- differs");
    t.steps++;
    t.same_top1 += tb == tr;
    t.worst = std::max(t.worst, rel);
    return tb;
}

// The width-1 run: every token and the backend's logits after it
struct Trace {
    std::vector<int32_t> seq;
    std::vector<std::vector<float>> logits;
    std::vector<double>
        ref_nll;  // -log p of seq[i + 1] under the oracle after step i; NaN where the oracle did not run
};

// TONA_REF_TRACE_OUT=file: the oracle's residual after each mixer and each layer of every step, in the layout of the
// GPU tap (gpu/mega_tap.h, TONA_TAP_OUT), for tools/tap_compare.py
struct TraceFile {
    std::FILE* fp = nullptr;
    TraceFile() {
        if (const char* f = std::getenv("TONA_REF_TRACE_OUT"); f && *f) fp = std::fopen(f, "wb");
    }
    TraceFile(const TraceFile&) = delete;
    TraceFile& operator=(const TraceFile&) = delete;
    ~TraceFile() {
        if (fp) std::fclose(fp);
    }
};

std::vector<float> ref_step(tona::ref::Oracle& ref, int32_t token, TraceFile& out) {
    if (!out.fp) return ref.step(token);
    const int64_t pos = ref.pos();
    tona::ref::Trace trace;
    std::vector<float> logits = ref.step(token, &trace);
    if (trace.residual_mid.size() != trace.residual.size())
        throw std::runtime_error(
            "TONA_REF_TRACE_OUT: this oracle does not trace the residual after the mixers (Ornith only)");
    const int32_t hd[4] = {(int32_t)(2 * trace.residual.size()), (int32_t)trace.residual[0].size(), 1, (int32_t)pos};
    std::fwrite(hd, sizeof hd, 1, out.fp);
    for (size_t l = 0; l < trace.residual.size(); l++) {
        std::fwrite(trace.residual_mid[l].data(), sizeof(float), trace.residual_mid[l].size(), out.fp);
        std::fwrite(trace.residual[l].data(), sizeof(float), trace.residual[l].size(), out.fp);
    }
    std::fflush(out.fp);
    return logits;
}

struct Lane {
    size_t p = 0;  // next token of the trace
};

struct WidthTally {
    int passes = 0, rows = 0, bad = 0;
};

// Slot 0 saves at a third of the trace, goes back to it from two thirds, then stashes, resets and unstashes
void save_point_trip(tona::backend::Backend& be, Lane& l, const Trace& tr, size_t& saved_at, bool& done) {
    be.use_slot(0);
    const size_t n = tr.seq.size();
    if (saved_at == 0 && l.p >= n / 3) {
        be.save_state();
        saved_at = l.p;
    } else if (saved_at > 0 && !done && l.p >= 2 * n / 3) {
        be.load_state();
        tona::backend::HostSnap snap;
        be.stash(snap);
        be.reset();
        be.unstash(snap);
        if ((size_t)be.pos() != saved_at) throw std::runtime_error("widths: unstash lands at the wrong position");
        l.p = saved_at;
        done = true;
    }
}

int widths(tona::backend::Backend& be, const Trace& tr, unsigned seed, size_t vocab) {
    std::mt19937 rng(seed);
    const int k = std::min(be.seqs(), 2);
    std::vector<Lane> lanes(k);
    for (int s = 0; s < k; s++) be.use_slot(s), be.reset();
    WidthTally w;
    size_t saved_at = 0;
    bool done = false;
    for (;;) {
        save_point_trip(be, lanes[0], tr, saved_at, done);
        std::vector<int> slots, n;
        std::vector<int32_t> tok;
        for (int s = 0; s < k; s++) {
            const size_t left = tr.seq.size() - lanes[s].p;
            if (left == 0) continue;
            const int nb = (int)std::min<size_t>(left, 1 + rng() % tona::backend::BMAX);
            slots.push_back(s), n.push_back(nb);
            tok.insert(tok.end(), tr.seq.begin() + lanes[s].p, tr.seq.begin() + lanes[s].p + nb);
        }
        if (slots.empty()) break;
        const float* lg = be.forward_multi(slots.data(), n.data(), (int)slots.size(), tok.data());
        for (size_t i = 0; i < slots.size(); i++) {
            Lane& l = lanes[slots[i]];
            for (int t = 0; t < n[i]; t++, lg += vocab, w.rows++)
                w.bad += std::memcmp(lg, tr.logits[l.p + t].data(), vocab * sizeof(float)) != 0;
            const int m = 1 + (int)(rng() % n[i]);
            be.use_slot(slots[i]);
            be.commit(m);
            l.p += m;
        }
        w.passes++;
    }
    std::fprintf(stderr, "Widths (seed %u, %d slots): %d passes, %d rows, %d not bit-identical, save point at %zu\n",
                 seed, k, w.passes, w.rows, w.bad, saved_at);
    return w.bad == 0 && done ? 0 : 1;
}

// The state after prefilling the first n tokens of the trace, cut at the given positions
tona::backend::HostSnap prefill_state(tona::backend::Backend& be, const Trace& tr, int n, std::vector<int> cuts) {
    be.use_slot(0);
    be.reset();
    cuts.push_back(n);
    int at = 0;
    for (int c : cuts) {
        if (c > at) be.prefill(tr.seq.data() + at, c - at, nullptr);
        at = std::max(at, c);
    }
    be.save_state();
    tona::backend::HostSnap s;
    be.stash(s);
    return s;
}

bool same_state(const tona::backend::HostSnap& a, const tona::backend::HostSnap& b) {
    return a.state == b.state && a.kv == b.kv && a.pool == b.pool && a.tokens == b.tokens;
}

// -log p of the next token from the width-1 run's logits
double trace_nll(const Trace& tr, int i, size_t vocab) {
    return tona::backend::neg_log_p(tr.logits[i].data(), vocab, tr.seq[i + 1]);
}

// Prefill, then the rest of the trace token by token against the width-1 run; the -log p of the prefilled tokens
int prefill_quality(tona::backend::Backend& be, const Trace& tr, int n, size_t vocab) {
    be.use_slot(0);
    be.reset();
    std::vector<float> nll;
    be.prefill(tr.seq.data(), n, &nll);
    // The maximum alone cannot tell a tie (one spike) from a drift: also the means of both paths, how many tokens
    // differ by more than 0.01 nat (the rule of thumb in docs/backends/README.md) and the first of them
    double nll_diff = 0, sum_pre = 0, sum_dec = 0, sum_abs = 0;
    int over = 0, first = -1;
    for (int i = 0; i + 1 < n; i++) {
        const double dec = trace_nll(tr, i, vocab), d = std::fabs(nll[i] - dec);
        nll_diff = std::max(nll_diff, d);
        sum_pre += nll[i], sum_dec += dec, sum_abs += d;
        if (d > 0.01 && over++ == 0) first = i + 1;
    }
    const double m = std::max(1, n - 1);
    // Against the oracle (where it ran): which of the two paths is closer to the f32 reference
    double pre_ref = 0, dec_ref = 0;
    int n_ref = 0;
    for (int i = 0; i + 1 < n && i < (int)tr.ref_nll.size(); i++) {
        if (std::isnan(tr.ref_nll[i])) continue;
        pre_ref += std::fabs(nll[i] - tr.ref_nll[i]), dec_ref += std::fabs(trace_nll(tr, i, vocab) - tr.ref_nll[i]);
        n_ref++;
    }
    if (n_ref > 0)
        std::fprintf(stderr, "Against the oracle over %d tokens: mean |d nll| prefill %.3e, decode %.3e\n", n_ref,
                     pre_ref / n_ref, dec_ref / n_ref);
    std::fprintf(stderr,
                 "Prefill -log p per token: mean %.5f (prefill) against %.5f (decode), mean |d| %.2e; %d tokens over "
                 "0.01, the first at position %d\n",
                 sum_pre / m, sum_dec / m, sum_abs / m, over, first);
    Tally t;
    t.steps = n;
    for (size_t i = n; i < tr.seq.size(); i++) {
        const float* lg = be.forward(&tr.seq[i], 1);
        compare(lg, tr.logits[i], tr.seq[i], t);
        be.commit(1);
    }
    std::fprintf(stderr,
                 "Prefill against decode: %d tokens prefilled, max |d nll| %.2e; then %d steps, top-1 equal in %d, "
                 "worst relative deviation %.2e\n",
                 n, nll_diff, t.steps - n, t.same_top1, t.worst);
    return 0;
}

int prefill_check(tona::backend::Backend& be, const Trace& tr, unsigned seed, size_t vocab) {
    if (be.prefill_chunk() <= 0) throw std::runtime_error("--prefill: the backend has no prefill kernels");
    const int n = (int)tr.seq.size() * 2 / 3;
    const tona::backend::HostSnap whole = prefill_state(be, tr, n, {});
    std::mt19937 rng(seed);
    int differ = 0;
    for (int round = 0; round < 3; round++) {
        std::vector<int> cuts;
        for (int c = 1 + (int)(rng() % 40); c < n; c += 1 + (int)(rng() % be.prefill_chunk())) cuts.push_back(c);
        const bool same = same_state(whole, prefill_state(be, tr, n, cuts));
        std::fprintf(stderr, "Cut invariance (seed %u, round %d): %zu cuts, state %s\n", seed, round, cuts.size(),
                     same ? "byte-equal" : "DIFFERS");
        differ += !same;
    }
    prefill_quality(be, tr, n, vocab);
    return differ == 0 ? 0 : 1;
}

int run(int argc, char** argv) {
    const Options o = parse(argc, argv);
    tona::backend::register_builtin();
    const auto model_ptr = tona::load_model(o.model);
    const tona::ModelBase& model = *model_ptr;
    const tona::Tokenizer tok(model.gguf());
    tona::backend::CreateOptions co;
    co.max_ctx = o.ctx;
    if (!o.mtp.empty()) {
        if (o.dump.empty() && o.golden.empty()) throw std::invalid_argument("--mtp needs --dump or --golden");
        const tona::Qwen4ExpModel& coder = tona::model_as<tona::Qwen4ExpModel>(model);  // the MTP file is the Coder's
        const tona::MtpHead head(o.mtp, coder.hp());
        co.mtp = &head;
        auto be = tona::backend::registry::create(o.backend, model, co);
        const tona::check::MtpOptions mo{o.prompt, o.n_gen, o.mtp_steps, 8, o.dump, o.golden, 0.1, o.mtp_prompt};
        return tona::check::mtp_check(*be, tok, mo, coder.hp().vocab);
    }
    auto be = tona::backend::registry::create(o.backend, model, co);
    // Its own mapping: the reference backend unmaps the model file once the weights are on the GPU
    const auto ref_model = tona::load_model(o.model);
    const auto ref_oracle = tona::ref::make_oracle(*ref_model);
    tona::ref::Oracle& ref = *ref_oracle;
    std::vector<int32_t> ids = tok.encode(o.prompt);
    std::fprintf(stderr, "Backend %s, %zu prompt tokens, %d generated\n", be->info().name.c_str(), ids.size(), o.n_gen);
    Tally t;
    Trace tr;
    TraceFile trace_out;
    const size_t vocab = model.moe_shape().vocab;
    int32_t next = -1;
    for (size_t i = 0; i < ids.size() + (size_t)o.n_gen; i++) {
        const int32_t token = i < ids.size() ? ids[i] : next;
        const float* lg = be->forward(&token, 1);
        tr.seq.push_back(token);
        tr.logits.emplace_back(lg, lg + vocab);
        std::vector<float> rl;
        if (o.ref_steps < 0 || (int64_t)i < o.ref_steps) {
            rl = ref_step(ref, token, trace_out);
            next = compare(lg, rl, token, t);
        } else {
            next = argmax(lg, vocab);
        }
        const int32_t target = i + 1 < ids.size() ? ids[i + 1] : next;
        tr.ref_nll.push_back(rl.empty() ? std::nan("") : tona::backend::neg_log_p(rl.data(), vocab, target));
        be->commit(1);
    }
    std::fprintf(stderr, "Check: %d steps, top-1 equal in %d, worst relative deviation %.2e\n", t.steps, t.same_top1,
                 t.worst);
    const int wide = o.widths_seed < 0 ? 0 : widths(*be, tr, (unsigned)o.widths_seed, vocab);
    const int pre = o.prefill_seed < 0 ? 0 : prefill_check(*be, tr, (unsigned)o.prefill_seed, vocab);
    return t.same_top1 == t.steps && wide == 0 && pre == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 2;
    }
}
