#include "server/engine.h"

#include <algorithm>
#include <cstdlib>

#include "backend/env.h"
#include "model/progress.h"
#include "server/http_util.h"

namespace tona::server {

namespace {

// Length of the start of s that consists of complete UTF-8 characters
size_t utf8_complete(const std::string& s) {
    size_t i = s.size();
    for (int k = 0; k < 4 && i > 0; k++) {
        const unsigned char c = s[i - 1];
        if ((c & 0xc0) == 0x80) {  // continuation byte
            i--;
            continue;
        }
        const int need = c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : c >= 0xc0 ? 2 : 1;
        return (int)(s.size() - (i - 1)) >= need ? s.size() : i - 1;
    }
    return s.size();
}

}  // namespace

Engine::~Engine() {
    {
        std::lock_guard lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    th_.join();
}

void Engine::submit(std::shared_ptr<Job> j) {
    if (hold_) hold_->arrival(j->warm);  // before the job is queued: the hold has stopped when the engine takes it
    {
        std::lock_guard lk(mu_);
        q_.push_back(std::move(j));
    }
    cv_.notify_all();
}

std::vector<Stream*> Engine::live_streams() const {
    std::vector<Stream*> live;
    live.reserve(act_.size());
    for (const auto& a : act_) live.push_back(a->st.get());
    return live;
}

void Engine::loop() {
    while (true) {
        // Admit while slots are free (the prefill runs alone, the other conversations wait for it)
        while (true) {
            std::shared_ptr<Job> j;
            {
                std::unique_lock lk(mu_);
                const auto ready = [&] { return quit_ || !q_.empty(); };
                // The backend's housekeeping (VRAM governor) runs here too, because a program that takes VRAM while no
                // request runs must not wait for the next request
                while (act_.empty() && poll_idle_ && !cv_.wait_for(lk, std::chrono::milliseconds(250), ready)) {
                    lk.unlock();
                    poll_idle_ = gen_.idle_tick();
                    lk.lock();
                }
                if (act_.empty()) cv_.wait(lk, ready);
                if (quit_) return;
                if (q_.empty() || (int)act_.size() >= gen_.seqs()) break;
                // Highest priority first, in order within the same level; a warm-up only when nothing else runs
                auto it = next_job(q_, !act_.empty());
                if (it == q_.end()) break;
                j = *it;
                q_.erase(it);
                cur_.push_back(j);
            }
            admit(j);
        }
        if (act_.empty()) continue;
        step(live_streams());
    }
}

// After an exception in a pass: reset all involved slots; requests without output start over, the others get the error
void Engine::fail_all(const std::exception& e) {
    fprintf(stderr, "%s Pass failed: %s\n", hms().c_str(), e.what());
    for (auto& a : act_) {
        gen_.use_slot(a->st->slot);
        gen_.recover();
        Job& j = *a->job;
        if (j.out_tokens == 0 && !j.cancel && j.attempt++ == 0) {
            j.live_state = 0;
            std::lock_guard lk(mu_);
            std::erase(cur_, a->job);
            q_.push_front(a->job);
        } else {
            j.error = e.what();
            close(j);
        }
        a->st->done = true, a->job.reset();
    }
    act_.clear();
}

// One pass over the running conversations, finish the ones that are done
void Engine::step(const std::vector<Stream*>& live) {
    try {
        gen_.advance(live);
    } catch (const std::exception& e) {
        fail_all(e);
        return;
    }
    for (auto& a : act_)
        if (a->st->done) finish(*a);
    std::erase_if(act_, [](const auto& a) { return !a->job; });
}

void Engine::close(Job& j) {
    j.live_state = 0;
    {
        std::lock_guard lk(mu_);
        std::erase_if(cur_, [&](const auto& p) { return p.get() == &j; });
        if (hold_ && cur_.empty() && q_.empty()) hold_->idle();
    }
    {
        std::lock_guard lk(j.mu);
        j.started = j.done = true;
    }
    j.cv.notify_all();
}

// A long prompt next to running conversations: in slices, in between up to TONA_PREFILL_YIELD_MS (default 1000) of
// decode for the others. A request that outranks all running ones (the auto-mode reviewer) computes in one go.
void Engine::install_yield(const Job& j, int best_prio) {
    static const double yield_ms = tona::env::number("TONA_PREFILL_YIELD_MS", 1000);
    if (act_.empty() || j.prio() < best_prio || yield_ms <= 0) return;
    gen_.set_yield([this] {
        const auto t0 = std::chrono::steady_clock::now();
        while (!act_.empty() &&
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < yield_ms)
            step(live_streams());
    });
}

// A warm-up gives way when its client is gone or a real request waits (checked at every prefill slice)
bool Engine::interrupted(const Job& j) {
    if (j.cancel) return true;
    std::lock_guard lk(mu_);
    return std::any_of(q_.begin(), q_.end(), [](const auto& q) { return !q->warm; });
}

// Pick a slot and compute the prompt; on an error once more from scratch, then report it. false: the job is closed.
bool Engine::prefill_job(Job& j, const std::vector<int>& busy) {
    for (;; j.attempt++) {
        try {
            gen_.select_slot(j.prompt, busy);
            j.live_t0 = tona::progress::now();
            tona::progress::prefill(0, 0);
            j.live_state = 1;
            j.info =
                j.warm ? gen_.prefill_warm(j.prompt, [this, &j] { return interrupted(j); }) : gen_.prefill(j.prompt);
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "%s Request failed: %s\n", hms().c_str(), e.what());
            gen_.recover();
            if (j.attempt == 0 && !j.cancel) continue;
            j.error = e.what();
            close(j);
            return false;
        }
    }
}

void Engine::admit(std::shared_ptr<Job> jp) {
    Job& j = *jp;
    // Cancelled in the queue: nothing was touched, so no recover (it would discard the running conversation)
    if (j.cancel) {
        j.error = "cancelled";
        close(j);
        return;
    }
    std::vector<int> busy;
    int best = 3;
    for (const auto& a : act_) busy.push_back(a->st->slot), best = std::min(best, a->job->prio());
    install_yield(j, best);
    struct Unset {
        Generator& g;
        ~Unset() { g.set_yield(nullptr); }
    } unset{gen_};
    if (!prefill_job(j, busy)) return;
    if (j.warm) return finish_warm(j);
    start_generation(std::move(jp));
}

// Emits a decoded token to the job: stop sequences end the stream, incomplete UTF-8 and a possible start of a stop
// sequence are held back. false ends the generation.
bool Engine::on_token(Act& a, int32_t t) {
    static const int32_t think_end = tok_.find("</think>");
    Job& j = *a.job;
    if (j.cancel) return false;
    j.live_out = ++j.out_tokens;
    if (a.in_think) t == think_end ? (void)(a.in_think = false) : (void)j.think_tokens++;
    std::string& pend = a.pend;
    pend += tok_.decode(t);
    for (const auto& s : j.stops)
        if (const size_t p = pend.find(s); !s.empty() && p != std::string::npos) {
            j.push(pend.substr(0, p));
            pend.clear();
            j.stop_seq = s;
            a.hit = true;
            return false;
        }
    const size_t hold = a.hold;
    size_t n = utf8_complete(pend);
    n = n > (hold ? hold - 1 : 0) ? n - (hold ? hold - 1 : 0) : 0;
    n = utf8_complete(pend.substr(0, n));
    if (n) j.push(pend.substr(0, n)), pend.erase(0, n);
    return true;
}

void Engine::start_generation(std::shared_ptr<Job> jp) {
    auto a = std::make_unique<Act>();
    Job& j = *jp;
    a->job = std::move(jp);
    j.live_out = 0;
    j.live_t0 = tona::progress::now();
    j.live_state = 2;
    {
        std::lock_guard lk(j.mu);
        j.started = true;
    }
    j.cv.notify_all();
    const int room = (int)(gen_.max_ctx() - (int64_t)j.prompt.size() - 4);
    const int max_tokens = std::max(1, std::min(j.max_tokens, room));
    // Room for the answer: thinking ends at max_tokens - min(2048, max_tokens / 4) at the latest, otherwise high
    // effort on hard tasks thinks past Claude Code's limit (8,192) and no code arrives
    // (research/measurements/2026-10-26-effort.md: 8 of 23 tasks above 8,000 thinking tokens)
    if (j.thinking) {
        const int cap = max_tokens - std::min(2048, max_tokens / 4);
        j.smp.think_budget = j.smp.think_budget < 0 ? cap : std::min(j.smp.think_budget, cap);
    }
    for (const auto& s : j.stops) a->hold = std::max(a->hold, s.size());
    a->in_think = j.thinking;
    a->t0 = std::chrono::steady_clock::now();
    a->s0 = gen_.mega().stats();
    Act* ap = a.get();
    a->st = gen_.begin(max_tokens, [this, ap](int32_t t) { return on_token(*ap, t); }, j.smp, weight(j));
    act_.push_back(std::move(a));
}

}  // namespace tona::server
