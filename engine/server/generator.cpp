#include "server/generator.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <tuple>

#include "backend/budget.h"
#include "backend/env.h"
#include "backend/host_trace.h"
#include "model/progress.h"
#include "server/generator_env.h"

namespace tona::server {

namespace {

// TONA_TOKEN_LOG: per request "P" and the tokens of the context, then per pass the accepted ones
void log_prompt_tokens(const std::vector<int32_t>& tokens) {
    const char* f = tona::env::text("TONA_TOKEN_LOG");
    if (!f) return;
    static FILE* tp = std::fopen(f, "a");
    if (!tp) return;
    std::fprintf(tp, "P");
    for (int32_t t : tokens) std::fprintf(tp, " %d", t);
    std::fprintf(tp, "\n"), std::fflush(tp);
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

Generator::Generator(const ModelBase& m, backend::Backend& mg, ref::DFlash* dfl, const Tokenizer& tok,
                     const ChatFormat& chat)
    : m_(m), mg_(mg), dfl_(dfl), max_ctx_(mg.max_ctx()), last_logits_(m.dims().vocab) {
    init_stop_tokens(tok, chat);
    read_env();
    reset_calib();
    ram_gb_ = backend::prompt_ram_gb();
    im_start_ = tok.find(std::string(chat.turn_start()));
    if (!disk_dir_.empty()) progress::begin("Checkpoints from the SSD"), disk_scan();
}

void Generator::init_stop_tokens(const Tokenizer& tok, const ChatFormat& chat) {
    for (const std::string_view t : chat.stop_texts())
        if (const int32_t id = tok.find(std::string(t)); id >= 0) stops_.push_back(id);
    if (tok.eos() >= 0 && std::find(stops_.begin(), stops_.end(), tok.eos()) == stops_.end())
        stops_.push_back(tok.eos());
    // Closing text when the thinking budget is reached; TONA_THINK_FORCE_MSG=0: only the end of the block, without the
    // sentence that asks the model to answer now (the format's own text, e.g. server/chat_qwen3_render.cpp)
    think_end_ = tok.find(std::string(chat.think_end()));
    think_force_ = tok.encode(chat.think_close(tona::env::integer("TONA_THINK_FORCE_MSG", 1) != 0));
}

void Generator::read_env() {
    sfx_min_ = (int)tona::env::integer("TONA_SUFFIX_MIN", sfx_min_);
    if (const backend::PassCosts* c = mg_.pass_costs()) {
        std::copy(c->pass, c->pass + backend::BMAX + 1, t_mpass_);
        std::copy(c->multi, c->multi + backend::BMAX + 1, t_xpass_);
        t_step_ = c->step, t_fill_ = c->fill;
    }
    t_step_ = tona::env::number("TONA_MTP_STEP_MS", t_step_);
    t_fill_ = tona::env::number("TONA_MTP_FILL_MS", t_fill_);
    pf_min_ = (int)tona::env::integer("TONA_PREFILL_MIN", pf_min_);
    pf_slice_ = (int)tona::env::integer("TONA_PREFILL_SLICE", pf_slice_);
    slots_ = (int)std::max<int64_t>(1, tona::env::integer("TONA_PROMPT_SLOTS", slots_));
    if (const char* e = tona::env::text("TONA_PROMPT_CACHE")) disk_dir_ = e;
    disk_gb_ = tona::env::number("TONA_PROMPT_CACHE_GB", disk_gb_);
}

// Start values of the calibration (w, b_0 ... b_6), fitted offline on 14 prompts that are not in the benchmark (11
// code, 3 text, 300 tokens each): greedy 3,148 and sampling 1,769 observations; mean log-likelihood greedy -0.291 ->
// -0.269, sampling -0.468 -> -0.369 against the raw q (research/measurements/2026-10-25-width-8.md)
void Generator::reset_calib() {
    mcal_[0] = mcal_[1] = spec::DraftCalib<7>();
    calib_[0] = spec::DraftCalib<7>({0.758, 0.065, -0.253, -0.522, -0.023, -0.533, -0.808, -1.099});
    calib_[1] = spec::DraftCalib<7>({0.582, -0.694, -0.436, -0.552, 0.203, -0.178, -0.054, -0.141});
}

Stop Generator::generate(int max_tokens, const std::function<bool(int32_t)>& on_token, const Sampling& smp) {
    auto s = begin(max_tokens, on_token, smp);
    while (!s->done) advance({s.get()});
    return s->why;
}

std::unique_ptr<Stream> Generator::begin(int max_tokens, std::function<bool(int32_t)> on_token, const Sampling& smp,
                                         double weight) {
    const size_t V = m_.dims().vocab;
    auto sp = std::make_unique<Stream>();
    Stream& s = *sp;
    s.slot = mg_.slot(), s.max_tokens = max_tokens, s.on_token = std::move(on_token), s.smp = smp, s.weight = weight;
    s.greedy = smp.temperature <= 0;
    // Sampling with MTP: the draft is drawn from q (logits of the MTP head, same temperature, top_k, top_p) and
    // verified with min(1, p/q), otherwise the residual. Greedily drawn drafts (q = 1) get through only with p(draft),
    // sum min(p, q) is larger. TONA_MTP_SAMPLE=0: greedy draft as before
    static const bool mtp_sample = tona::env::integer("TONA_MTP_SAMPLE", 1) != 0;
    s.qs = !s.greedy && mtp_sample && !dfl_;
    if (!dfl_ && mg_.has_mtp()) mg_.mtp_keep_logits(s.qs);
    s.rng.seed(smp.seed ? smp.seed : std::random_device{}());
    s.last = s.greedy ? argmax(last_logits_.data(), V) : sample(make_dist(last_logits_.data(), V, smp), s.rng);
    // Thinking budget: count tokens in the <think> block; when it runs out, think_force_ follows in passes of at most
    // n_draft + 1 tokens. The check is per accepted token, so a pass overshoots by at most 8 tokens (in SGLang drafts
    // slipped past the logit processor: budget 200, ~1400 tokens, SGLang issue 25536)
    s.think_left = think_end_ >= 0 ? smp.think_budget : -1;
    if (FILE* dl = draft_log()) std::fprintf(dl, "R\n"), std::fflush(dl);
    log_prompt_tokens(mg_.tokens());
    if (!calib_keep() && !backend::deterministic()) {
        reset_calib();
        probe_ = 0;
        full_ = true;
    }
    for (int i = 0; i < 2; i++) s.calib[i] = calib_[i], s.mcal[i] = mcal_[i];
    s.probe = probe_, s.full = full_;
    return sp;
}

// Emits last and checks the stop conditions; false: the stream is done
bool Generator::pre(Stream& s) {
    if (is_stop(s.last)) {
        s.why = Stop::eos;
        return false;
    }
    count_think(s, s.last);
    if (!s.on_token(s.last)) {
        s.why = Stop::callback;
        return false;
    }
    if (++s.n >= s.max_tokens) return false;
    // Context: the draft block (up to n_draft + 1) must still fit
    if (mg_.pos() + n_draft_max() + 1 >= max_ctx_) return false;
    return true;
}

// Catches the drafter up while the features (for MTP: h) of the last pass are still there
void Generator::finish(Stream& s) {
    s.done = true;
    catch_up_drafter();
    if (calib_keep() || backend::deterministic()) {
        for (int i = 0; i < 2; i++) calib_[i] = s.calib[i], mcal_[i] = s.mcal[i];
        probe_ = s.probe, full_ = s.full;
    }
}

// What the other conversations bring into the pass: those drafted already with their block and expected tokens, the
// rest as in their last pass
void Generator::others_expectation(const std::vector<Stream*>& go, size_t i, int& n_o, double& e_o) const {
    n_o = 0, e_o = 0;
    for (size_t j = 0; j < go.size(); j++) {
        if (j == i) continue;
        if (j > i) {
            n_o += go[j]->prev_n, e_o += go[j]->weight * go[j]->prev_e;
            continue;
        }
        double e = 1, run = 1;
        for (size_t k = 0; k + 1 < go[j]->blk.size() && k < go[j]->pc.size(); k++) run *= go[j]->pc[k], e += run;
        n_o += (int)go[j]->blk.size(), e_o += go[j]->weight * (go[j]->pc.empty() ? (double)go[j]->blk.size() : e);
    }
}

// Cuts a fixed block to its share; cut-off tokens of the thinking close stay in the queue
void Generator::trim_fixed_block(Stream& s, int cap) {
    if (s.forcing) s.forced.insert(s.forced.begin(), s.blk.begin() + cap, s.blk.end());
    s.blk.resize(cap);
    if (s.qd.size() > s.blk.size() - 1) s.qd.resize(s.blk.size() - 1);
}

// Steps of all MTP chains still running, in one MTP pass (bit-exact to one pass each)
void Generator::run_chains(const std::vector<Stream*>& chains, size_t n_go, int cap,
                           std::chrono::steady_clock::time_point t_start) {
    std::vector<Stream*> on;
    for (Stream* s : chains)
        if (chain_push(*s)) on.push_back(s);
    while (!on.empty()) {
        std::vector<int> sl;
        std::vector<int32_t> tk;
        std::vector<backend::MtpPick> pk(on.size());
        for (Stream* s : on) sl.push_back(s->slot), tk.push_back(s->blk.back());
        mg_.mtp_step_multi(sl.data(), tk.data(), (int)on.size(), pk.data());
        std::vector<Stream*> next;
        for (size_t i = 0; i < on.size(); i++) {
            mg_.use_slot(on[i]->slot);
            on[i]->ch_pk = pk[i];
            chain_draw(*on[i]);
            if (chain_push(*on[i])) next.push_back(on[i]);
        }
        on = std::move(next);
    }
    // Drafting time in equal parts (log)
    const double ms = ms_since(t_start);
    for (Stream* s : chains) {
        chain_end(*s);
        s->ms_drf = ms / (double)n_go;
        if (s->fixed && (int)s->blk.size() > cap) trim_fixed_block(*s, cap);  // probe: cut to the share
    }
}

// One forward pass over the blocks of all streams, then verify each
void Generator::forward_and_verify(const std::vector<Stream*>& go) {
    const size_t V = m_.dims().vocab;
    const auto t_fw = std::chrono::steady_clock::now();
    if (go.size() == 1) {
        mg_.use_slot(go[0]->slot);
        const float* lg = mg_.forward(go[0]->blk.data(), (int)go[0]->blk.size());
        post(*go[0], lg, ms_since(t_fw));
        return;
    }
    std::vector<int> slots, n;
    std::vector<int32_t> tok;
    for (Stream* s : go) {
        slots.push_back(s->slot), n.push_back((int)s->blk.size());
        tok.insert(tok.end(), s->blk.begin(), s->blk.end());
    }
    const float* lg = mg_.forward_multi(slots.data(), n.data(), (int)go.size(), tok.data());
    const double ms_fw = ms_since(t_fw);
    size_t row = 0;
    for (Stream* s : go) {
        mg_.use_slot(s->slot);
        post(*s, lg + row * V, ms_fw);
        row += s->blk.size();
    }
}

// Starts the pass: pre() each stream, those that go on are returned
std::vector<Stream*> Generator::begin_pass(const std::vector<Stream*>& live) {
    std::vector<Stream*> go;
    for (Stream* s : live) {
        if (s->done) continue;
        mg_.use_slot(s->slot);
        if (pre(*s))
            go.push_back(s);
        else
            finish(*s);
    }
    return go;
}

// Drafts the block of every stream; streams that only began an MTP chain are returned
std::vector<Stream*> Generator::draft_all(const std::vector<Stream*>& go, int cap) {
    const bool multi = go.size() > 1;
    std::vector<Stream*> chains;
    for (size_t i = 0; i < go.size(); i++) {
        Stream* s = go[i];
        int n_o = 0;
        double e_o = 0;
        others_expectation(go, i, n_o, e_o);
        mg_.use_slot(s->slot);
        // Several conversations with MTP: begin the chains first, take the steps together later
        if (!draft(*s, n_draft_max() + 1, multi, std::min(n_o, backend::BMAX - 1), e_o, multi && mtp_multi_ && !dfl_)) {
            chains.push_back(s);
            continue;
        }
        if (multi && s->fixed && (int)s->blk.size() > cap) trim_fixed_block(*s, cap);
    }
    return chains;
}

void Generator::advance(const std::vector<Stream*>& live) {
    backend::HostTrace::mark("cycle_begin");
    const std::vector<Stream*> go = begin_pass(live);
    if (go.empty()) return;
    const bool multi = go.size() > 1;
    // Fixed blocks get at most an equal share per conversation; allocate shares out the MTP chains
    const int cap = multi ? std::max(1, backend::BMAX / (int)go.size()) : n_draft_max() + 1;
    const auto t_chain = std::chrono::steady_clock::now();
    const std::vector<Stream*> chains = draft_all(go, cap);
    if (!chains.empty()) run_chains(chains, go.size(), cap, t_chain);
    backend::HostTrace::mark("drafted");
    if (multi) allocate(go);
    forward_and_verify(go);
    backend::HostTrace::mark("verified");
    for (Stream* s : go)
        if (s->done) mg_.use_slot(s->slot), finish(*s);
    backend::HostTrace::mark("cycle_end");
}

void Generator::select_slot(const std::vector<int32_t>& prompt, const std::vector<int>& busy) {
    int best = -1;
    std::tuple<size_t, int, uint64_t> key{};
    for (int s = 0; s < mg_.seqs(); s++) {
        if (std::find(busy.begin(), busy.end(), s) != busy.end()) continue;
        mg_.use_slot(s);
        auto prefix = [&](const std::vector<int32_t>& p) {
            return !p.empty() && p.size() <= prompt.size() && std::equal(p.begin(), p.end(), prompt.begin()) ? p.size()
                                                                                                             : 0;
        };
        const size_t len = std::max(prefix(mg_.tokens()), mg_.has_state() ? prefix(mg_.snap_tokens()) : 0);
        const bool empty = mg_.pos() == 0 && !mg_.has_state();
        // longest start, then empty, then the longest unused
        const std::tuple<size_t, int, uint64_t> k{len, empty, ~slot_used_[s]};
        if (best < 0 || k > key) best = s, key = k;
    }
    if (best < 0) throw std::runtime_error("no free slot");
    mg_.use_slot(best);
    slot_used_[best] = ++clock_;
}

}  // namespace tona::server
