// The engine thread: takes jobs from the queue, runs up to seqs() conversations at once (one per slot) and hands the
// output to the HTTP threads piece by piece
#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "model/tokenizer.h"
#include "server/generator.h"
#include "server/hold_controller.h"
#include "server/job.h"

namespace tona::server {

class Engine {
public:
    Engine(Generator& gen, const tona::Tokenizer& tok)
        : gen_(gen), tok_(tok), hold_(HoldController::from_env(gen.backend())), th_([this] { loop(); }) {}
    ~Engine();
    void submit(std::shared_ptr<Job> j);
    int64_t max_ctx() const { return gen_.max_ctx(); }

    // State for /status: running requests (engine: the first, active: all), counters, the latest requests
    json status() const;
    static json origin(const Job& j);

private:
    // A running request: up to seqs() at once, each in its own slot; output piece by piece
    struct Act {
        std::shared_ptr<Job> job;
        std::unique_ptr<Stream> st;
        std::string pend;  // not emitted yet: incomplete UTF-8 and a possible start of a stop sequence
        size_t hold = 0;
        bool hit = false, in_think = false;
        std::chrono::steady_clock::time_point t0;
        backend::Stats s0;  // counters at the start of the decode, for TONA_DECODE_DIAG
    };

    // Priority: auxiliary before main before subagent before warm; weight when drafts are shared across conversations
    static double weight(const Job& j) { return j.prio() == 0 ? 1.2 : j.prio() == 1 ? 1.0 : 0.8; }

    void loop();
    std::vector<Stream*> live_streams() const;
    void step(const std::vector<Stream*>& live);
    void close(Job& j);
    void admit(std::shared_ptr<Job> jp);
    void install_yield(const Job& j, int best_prio);
    bool prefill_job(Job& j, const std::vector<int>& busy);
    bool interrupted(const Job& j);
    void finish_warm(Job& j);
    void start_generation(std::shared_ptr<Job> jp);
    bool on_token(Act& a, int32_t t);
    void finish(Act& a);
    void record(const Job& j, double tpp);
    std::string decode_diag(const Act& a) const;
    void fail_all(const std::exception& e);

    Generator& gen_;
    const tona::Tokenizer& tok_;
    std::vector<std::unique_ptr<Act>> act_;  // engine thread only
    mutable std::mutex st_mu_;
    json totals_ = json::object(), recent_ = json::array();
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> q_;
    std::vector<std::shared_ptr<Job>> cur_;  // running right now (under mu_)
    bool quit_ = false;
    bool poll_idle_ = true;                 // the backend wants a tick every 250 ms while idle
    std::unique_ptr<HoldController> hold_;  // TONA_CLOCK_HOLD, else null; before th_, the thread uses it
    std::thread th_;
};

}  // namespace tona::server
