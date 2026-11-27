// What the engine reports about requests: the /status state, the per-request log line and the recent-request record
#include <algorithm>
#include <ctime>
#include <format>

#include "backend/env.h"
#include "model/progress.h"
#include "server/engine.h"
#include "server/http_util.h"

namespace tona::server {

namespace {

json status_entry(const Job& j, double now, int64_t prefill_done, int64_t prefill_total) {
    const int st = j.live_state.load();
    const double dt = now - j.live_t0.load();
    const int out = j.live_out.load();
    return {{"id", j.id},
            {"state", st == 0   ? "idle"
                      : st == 1 ? "prefill"
                                : "decode"},
            {"elapsed", st ? dt : 0.0},
            {"prompt", j.prompt.size()},
            {"prefill_done", st == 1 ? prefill_done : 0},
            {"prefill_total", st == 1 ? prefill_total : 0},
            {"out_tokens", st == 2 ? out : 0},
            {"tok_s", st == 2 && dt > 0 ? out / dt : 0.0},
            {"origin", Engine::origin(j)}};
}

// How the conversation was found: ", conversation from RAM in 12 ms, checkpoint at 4096, slot 1"
std::string describe_slot(const Job& j, int seqs, int slot) {
    std::string s = j.info.slot > 0   ? std::format(", conversation from RAM in {:.0f} ms", j.info.swap_ms)
                    : j.info.slot < 0 ? std::format(", new conversation (stashing {:.0f} ms)", j.info.swap_ms)
                                      : std::string();
    if (j.info.ckpt)
        s += std::format(", checkpoint at {}{}", j.info.ckpt, j.info.tail ? " (before the fixed end)" : "");
    if (j.info.disk_ms > 0) s += std::format(", from the SSD in {:.0f} ms", j.info.disk_ms);
    if (seqs > 1) s += std::format(", slot {}", slot);
    return s;
}

std::string describe_thinking(const Job& j) {
    if (!j.thinking) return "";
    return std::format(", effort {}, {} tokens thought{}", j.effort, j.think_tokens,
                       j.smp.think_budget >= 0 ? std::format(" (budget {})", j.smp.think_budget) : std::string());
}

}  // namespace

json Engine::origin(const Job& j) {
    return {{"project", j.project}, {"session", j.session}, {"agent", j.agent}, {"role", j.role}};
}

json Engine::status() const {
    int64_t pd = 0, pt = 0;
    tona::progress::prefill_state(pd, pt);
    const double now = tona::progress::now();
    json e = {{"state", "idle"},    {"elapsed", 0.0},  {"prompt", 0}, {"prefill_done", 0},
              {"prefill_total", 0}, {"out_tokens", 0}, {"tok_s", 0.0}};
    {
        std::lock_guard lk(mu_);
        json act = json::array();
        for (const auto& a : cur_) act.push_back(status_entry(*a, now, pd, pt));
        // engine: the request in prefill, otherwise the first (fields as with a single request)
        for (const json& a : act)
            if (a["state"] == "prefill" || e["state"] == "idle") e = a;
        e["active"] = act;
        e["slots"] = gen_.seqs();
        e["queue"] = q_.size();
        json w = json::array();
        for (const auto& q : q_) w.push_back(origin(*q));
        e["waiting"] = w;
    }
    if (hold_) {  // TONA_CLOCK_HOLD: what the rule did (server/hold_controller.h)
        const auto h = hold_->totals();
        e["clock_hold"] = {
            {"holds", h.holds}, {"covered", h.covered}, {"held_s", h.held_s}, {"last_hold_s", h.last_hold_s}};
    }
    std::lock_guard lk(st_mu_);
    return {{"engine", e}, {"totals", totals_}, {"recent", recent_}};
}

// TONA_DECODE_DIAG=1: what the CPU and the expert cache did during this decode (counter differences), to tell a slow
// decode from a normal one
std::string Engine::decode_diag(const Act& a) const {
    static const bool on = tona::env::flag("TONA_DECODE_DIAG");
    if (!on) return "";
    const auto& n = gen_.mega().stats();
    const auto& o = a.s0;
    const double passes = std::max<double>(a.st->passes, 1);
    return std::format(
        "\n  DIAG hits {} misses {} cpu_experts {} ({:.1f}/pass) gpu_direct {} fills {} yields {} launches {} "
        "wait {:.0f} ms read {:.0f} ms cpu {:.0f} ms select {:.0f} ms fill {:.0f} ms prefetch_wait {:.0f} ms "
        "prefetched {} used {} mtp_stale {} mtp_dyn {} early {}/{}",
        n.expert_hits - o.expert_hits, n.expert_misses - o.expert_misses, n.cpu_experts - o.cpu_experts,
        (n.cpu_experts - o.cpu_experts) / passes, n.gpu_direct - o.gpu_direct, n.fills - o.fills, n.yields - o.yields,
        n.launches - o.launches, (n.wait_s - o.wait_s) * 1e3, (n.read_s - o.read_s) * 1e3, (n.cpu_s - o.cpu_s) * 1e3,
        (n.select_s - o.select_s) * 1e3, (n.fill_s - o.fill_s) * 1e3, (n.prefetch_wait_s - o.prefetch_wait_s) * 1e3,
        n.prefetched - o.prefetched, n.prefetch_used - o.prefetch_used, n.mtp_stale - o.mtp_stale,
        n.mtp_dyn - o.mtp_dyn, n.early_hits - o.early_hits, n.early_misses - o.early_misses);
}

void Engine::finish(Act& a) {
    Job& j = *a.job;
    const Stream& st = *a.st;
    j.why = st.why;
    j.gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - a.t0).count();
    if (!a.hit && !a.pend.empty()) j.push(a.pend);
    const int64_t np = st.passes, sp = st.sfx_passes;
    const double tpp = np ? (double)st.gen_tokens / np : 0.0;
    // Origin: [project, role] (role only if not main)
    std::string who = j.project;
    if (j.role != "main") who += (who.empty() ? "" : ", ") + j.role;
    if (!who.empty()) who = " [" + who + "]";
    fprintf(stderr,
            "%s Request%s: %zu prompt tokens (%zu reused, %zu computed in %.1f s%s), %d tokens in "
            "%.1f s = %.1f tok/s, %.2f tokens per pass, %lld of %lld passes from the history (%.2f per pass)%s%s%s\n",
            hms().c_str(), who.c_str(), j.prompt.size(), j.info.reused, j.info.computed, j.info.seconds,
            describe_slot(j, gen_.seqs(), st.slot).c_str(), j.out_tokens, j.gen_s,
            j.out_tokens / std::max(j.gen_s, 1e-9), tpp, (long long)sp, (long long)np,
            sp ? (double)st.sfx_tokens / sp : 0.0, j.cancel ? ", cancelled" : "", describe_thinking(j).c_str(),
            decode_diag(a).c_str());
    record(j, tpp);
    close(j);
    a.job.reset();
}

// A warm-up ends after the prefill: no answer, the job only reports how far the stored state reaches
void Engine::finish_warm(Job& j) {
    j.why = Stop::length, j.warm_done = j.info.done;
    fprintf(stderr, "%s Warm-up%s: %zu prompt tokens (%zu reused, %zu computed in %.1f s%s), state covers %zu%s\n",
            hms().c_str(), (j.project.empty() ? std::string() : " [" + j.project + "]").c_str(), j.prompt.size(),
            j.info.reused, j.info.computed, j.info.seconds, describe_slot(j, gen_.seqs(), 0).c_str(), j.info.done,
            j.info.aborted ? ", interrupted by a request" : "");
    record(j, 0);
    close(j);
}

void Engine::record(const Job& j, double tpp) {
    const auto& ms = gen_.mega().stats();
    const json r = {{"time", (int64_t)std::time(nullptr)},
                    {"prompt", j.prompt.size()},
                    {"reused", j.info.reused},
                    {"computed", j.info.computed},
                    {"prefill_s", j.info.seconds},
                    {"out", j.out_tokens},
                    {"gen_s", j.gen_s},
                    {"tok_s", j.out_tokens / std::max(j.gen_s, 1e-9)},
                    {"prefill_tok_s", j.info.computed / std::max(j.info.seconds, 1e-9)},
                    {"tokens_per_pass", tpp},
                    {"slot", j.info.slot},
                    {"cancel", (bool)j.cancel},
                    {"effort", j.thinking ? j.effort : "off"},
                    {"think_tokens", j.think_tokens},
                    {"think_budget", j.smp.think_budget},
                    {"project", j.project},
                    {"session", j.session},
                    {"agent", j.agent},
                    {"role", j.role}};
    std::lock_guard lk(st_mu_);
    recent_.push_back(r);
    if (recent_.size() > 64) recent_.erase(recent_.begin());
    const int64_t n = totals_.value("requests", (int64_t)0) + 1;
    totals_ = {{"requests", n},
               {"passes", gen_.passes()},
               {"gen_tokens", gen_.gen_tokens()},
               {"tokens_per_pass", gen_.tokens_per_pass()},
               {"stashed", gen_.stashed()},
               {"on_disk", gen_.on_disk()},
               {"expert_hits", ms.expert_hits},
               {"expert_misses", ms.expert_misses},
               {"early_hits", ms.early_hits},
               {"early_misses", ms.early_misses},
               {"mtp_passes", ms.mtp_passes},
               {"multi_passes", ms.multi_passes},
               {"yields", ms.yields}};
}

}  // namespace tona::server
