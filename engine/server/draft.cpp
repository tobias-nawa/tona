// Drafting: the block for the next pass (forced tokens, history, MTP head chain or DFlash2) and its sharing across
// conversations
#include <algorithm>
#include <chrono>
#include <cstdlib>

#include "backend/env.h"
#include "server/generator.h"
#include "server/generator_env.h"

namespace tona::server {

// Feed the drafter the features (for MTP: h) of the last pass while they are still there
void Generator::catch_up_drafter() {
    if (!has_drafter()) return;
    if (!dfl_)
        mg_.mtp_fill(-1);
    else if (pending_)
        mg_.draft(pending_, nullptr, 0), pending_ = 0;
}

// MTP head chain, split up so that several conversations can take their steps together. Per draft p = calibrated q
// (softmax of the MTP head); the width k with the largest E[tokens] / (verify pass(k + 1) + catch-up + k steps),
// E = 1 + sum_j prod_{i<=j} p_i. The chain ends when even the next draft with the same p no longer raises the rate.
// Several conversations: rate over the whole pass (the others with n_o tokens and w * E = e_o), cost from the table
// for several conversations; the chain ends before the pass exceeds BMAX. TONA_MTP_FIXED=1: always n_draft
void Generator::chain_draw(Stream& s) {
    if (!s.qs) return;
    s.qd.push_back(make_dist_rows(mg_.mtp_logits(), mg_.mtp_rows(), [&](size_t r) { return mg_.mtp_token(r); }, s.smp));
    s.ch_pk.token = sample(s.qd.back(), s.rng), s.ch_pk.q = s.qd.back().prob(s.ch_pk.token);
}

void Generator::chain_begin(Stream& s, bool multi, int n_o, double e_o) {
    s.ch_probe = ++s.probe % 32 == 0;
    s.raw_q.clear();
    s.ch_best = 0, s.ch_i = 0, s.ch_e = 1, s.ch_run = 1;
    s.ch_multi = multi, s.ch_no = n_o, s.ch_eo = e_o, s.ch_w = multi ? s.weight : 1;
    const double c0 = multi ? t_xpass_[std::min(n_o + 1, backend::BMAX)] : t_mpass_[1] + t_fill_;
    s.ch_best_rate = (e_o + s.ch_w) / c0;
}

bool Generator::chain_push(Stream& s) {
    static const bool fixed = tona::env::flag("TONA_MTP_FIXED");
    const auto cost = [&](int k) {
        return s.ch_multi ? t_xpass_[std::clamp(s.ch_no + k + 1, 0, backend::BMAX)] + k * t_step_
                          : t_mpass_[k + 1] + t_fill_ + k * t_step_;
    };
    const int i = s.ch_i;
    if (i >= s.ch_nmax) return false;
    const double e_o = s.ch_eo, w = s.ch_w;
    s.raw_q.push_back(s.ch_pk.q);
    const double p = s.mcal[s.greedy ? 0 : 1].predict(s.raw_q.back(), i);
    s.pc.push_back(p);
    s.blk.push_back(s.ch_pk.token);
    s.ch_run *= p, s.ch_e += s.ch_run;
    s.ch_i = i + 1;
    if ((e_o + w * s.ch_e) / cost(i + 1) > s.ch_best_rate)
        s.ch_best_rate = (e_o + w * s.ch_e) / cost(i + 1), s.ch_best = i + 1;
    if (i + 1 == s.ch_nmax) return false;
    if (!fixed && !s.ch_probe && (e_o + w * (s.ch_e + s.ch_run * p)) / cost(i + 2) <= s.ch_best_rate) return false;
    return true;
}

void Generator::chain_end(Stream& s) {
    static const bool fixed = tona::env::flag("TONA_MTP_FIXED");
    s.fixed = fixed || s.ch_probe;
    if (!s.fixed && !s.ch_multi) s.blk.resize(1 + s.ch_best);
    if (s.qd.size() > s.blk.size() - 1) s.qd.resize(s.blk.size() - 1);
    s.ms_drf = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s.ch_t0).count();
}

// Closing text of the thinking budget: fixed tokens like a draft that is certain to be accepted
void Generator::draft_forced(Stream& s, int n_draft) {
    catch_up_drafter();
    while (!s.forced.empty() && (int)s.blk.size() < n_draft + 1)
        s.blk.push_back(s.forced.front()), s.forced.pop_front();
}

// Draft from the history: the drafter only catches up with the features of the last pass
void Generator::draft_from_history(Stream& s, int n_draft) {
    catch_up_drafter();
    s.blk.insert(s.blk.end(), s.sfx.begin(), s.sfx.begin() + std::min<size_t>(s.sfx.size(), n_draft));
}

// MTP head: catch up with the entries of the last pass (last at pos()), then step by step greedily. false: the chain
// was only begun (defer), advance finishes it
bool Generator::draft_mtp(Stream& s, int n_draft, bool multi, int n_o, double e_o, bool defer) {
    s.ch_t0 = std::chrono::steady_clock::now();
    mg_.mtp_keep_logits(s.qs);
    s.ch_pk = mg_.mtp_fill(s.last);
    chain_draw(s);
    chain_begin(s, multi, n_o, e_o);
    s.ch_nmax = multi ? std::min(n_draft, backend::BMAX - n_o - 1) : n_draft;
    if (defer) return false;
    while (chain_push(s)) {
        s.ch_pk = mg_.mtp_step(s.blk.back());
        chain_draw(s);
    }
    chain_end(s);
    return true;
}

void Generator::draft_dflash(Stream& s) {
    const int64_t d = m_.dims().d_model;
    std::vector<float> emb((size_t)(s.nd + 1) * d);
    dfl_->embed(s.last, emb.data());
    for (int i = 1; i <= s.nd; i++) dfl_->embed(dfl_->mask_token(), emb.data() + i * d);
    mg_.draft(pending_, emb.data(), s.nd + 1);
    pending_ = 0;
    for (int32_t t : dfl_->select(s.last, mg_.draft_logits(), mg_.draft_gate(), s.nd + 1, &s.probs)) s.blk.push_back(t);
    // Calibrated: p_i = sigmoid(w * logit(q_i) + b_i), fitted to the acceptances so far
    static const bool calib = tona::env::integer("TONA_DRAFT_CALIB", 1) != 0;
    s.raw_q.assign(s.probs.begin(), s.probs.begin() + s.nd);
    if (calib)
        for (int i = 0; i < s.nd; i++) s.probs[i] = (float)s.calib[s.greedy ? 0 : 1].predict(s.raw_q[i], i);
    // Width: k drafts bring 1 + sum_j prod_{i<=j} p_i tokens in t_pass[k + 1] + drafter
    int best = 0;
    double e = 1, run = 1, best_rate = 1 / (t_pass_[1] + t_drf_);
    for (int k = 1; k <= s.nd; k++) {
        run *= s.probs[k - 1], e += run;
        if (e / (t_pass_[k + 1] + t_drf_) > best_rate) best_rate = e / (t_pass_[k + 1] + t_drf_), best = k;
    }
    // TONA_DRAFT_FULL: always full width (to measure the calibration)
    static const bool full = tona::env::flag("TONA_DRAFT_FULL");
    // Probe: every 32nd pass at full width, otherwise the calibration would get no observations after "no drafts"
    // and stay there
    if (calib && ++s.probe % 32 == 0) best = s.nd;
    s.blk.resize(1 + (full ? s.nd : best));
}

// Block for the next pass: last and drafts (thinking close, history, MTP head or DFlash2), at most cap tokens.
// multi: do not shorten the MTP chain by its own rate, allocate does that across all conversations
bool Generator::draft(Stream& s, int cap, bool multi, int n_o, double e_o, bool defer) {
    const int n_draft = std::min(n_draft_max(), cap - 1);
    // A large drafter block (8) costs 8.8 instead of 5.3 ms: only after a pass that accepted all drafts, otherwise 4
    s.nd = s.full ? n_draft : std::min(N_SMALL, n_draft);
    s.blk.assign(1, s.last);
    s.qd.clear(), s.pc.clear();
    s.from_sfx = false, s.forcing = !s.forced.empty(), s.fixed = true;
    s.ms_drf = 0;
    // Baseline for measuring what speculation is worth: one token per pass, no drafter work at all
    static const bool spec_off = tona::env::flag("TONA_SPEC_OFF");
    if (spec_off) return true;
    // A model without a drafter (Bolt) still drafts from the history: the verify pass is exact and needs no drafter
    // state; only the thinking close stays off for it, as before
    if (!has_drafter() && s.forcing) return true;
    if (sfx_min_ > 0 && !s.forcing) {
        sfx_[s.slot].sync(mg_.tokens());
        s.from_sfx = sfx_[s.slot].draft(s.last, s.nd, s.sfx) >= sfx_min_ && !s.sfx.empty();
    }
    if (s.forcing)
        draft_forced(s, n_draft);
    else if (s.from_sfx)
        draft_from_history(s, n_draft);
    else if (!has_drafter())
        return true;
    else if (!dfl_) {
        if (!draft_mtp(s, n_draft, multi, n_o, e_o, defer)) return false;
    } else
        draft_dflash(s);
    // Do not draft past max_tokens
    s.blk.resize(std::min<size_t>(s.blk.size(), (size_t)(s.max_tokens - s.n + 1)));
    return true;
}

// Shares the drafts of the MTP chains across the conversations (DSpark pattern, vLLM PR #47808): first only last per
// conversation, then in turn the draft that raises the expected rate sum w * E / t(N) most (E = 1 + sum_j
// prod_{i<=j} p_i per conversation), as long as it rises and N <= BMAX. Fixed blocks (history, thinking close,
// probe) count in full.
void Generator::allocate(const std::vector<Stream*>& go) {
    int N = 0;
    double E = 0;
    std::vector<int> k(go.size(), 0), len(go.size(), 0);
    std::vector<double> run(go.size(), 1);
    for (size_t i = 0; i < go.size(); i++) {
        Stream& s = *go[i];
        if (s.fixed || s.pc.empty()) {
            N += (int)s.blk.size(), E += s.weight * (double)s.blk.size();
            len[i] = -1;
        } else {
            N += 1, E += s.weight;
            len[i] = (int)std::min(s.pc.size(), s.blk.size() - 1);
        }
    }
    double rate = E / t_xpass_[std::min(N, backend::BMAX)];
    while (N < backend::BMAX) {
        int best = -1;
        double best_rate = rate;
        for (size_t i = 0; i < go.size(); i++) {
            if (len[i] < 0 || k[i] >= len[i]) continue;
            const double r = (E + go[i]->weight * run[i] * go[i]->pc[k[i]]) / t_xpass_[N + 1];
            if (r > best_rate) best_rate = r, best = (int)i;
        }
        if (best < 0) break;
        run[best] *= go[best]->pc[k[best]];
        E += go[best]->weight * run[best], N++, k[best]++, rate = best_rate;
    }
    for (size_t i = 0; i < go.size(); i++)
        if (len[i] >= 0) {
            go[i]->blk.resize(1 + k[i]);
            if (go[i]->qd.size() > (size_t)k[i]) go[i]->qd.resize(k[i]);
        }
}

}  // namespace tona::server
