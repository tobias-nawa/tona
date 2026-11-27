// One request on its way through the server: filled by the route, worked on by the engine thread, read by the
// route's HTTP thread
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "server/chat.h"
#include "server/generator.h"
#include "third_party/httplib.h"

namespace tona::server {

struct Job {
    std::vector<int32_t> prompt;
    int max_tokens = 1024;
    Sampling smp;
    std::vector<std::string> stops;
    std::atomic<bool> cancel{false};
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> pieces;
    bool started = false, done = false;
    Stop why = Stop::length;
    std::string stop_seq, error;
    PrefillInfo info;
    int out_tokens = 0, think_tokens = 0;  // think_tokens: those inside the <think> block
    double gen_s = 0;
    bool thinking = false;  // the answer starts inside the <think> block
    std::string effort;
    // Origin: project (X-Tona-Project or the working directory in the system prompt), Claude Code session and agent,
    // role main / subagent / auxiliary (the auto-mode reviewer) / warm (max_tokens 0); priority: auxiliary before main
    // before subagent before warm
    std::string project, session, agent, role = "main";
    bool warm = false;  // warm-up: computes the prompt and stores checkpoints, generates nothing
    int prio() const { return warm ? 3 : role == "auxiliary" ? 0 : role == "subagent" ? 2 : 1; }
    // Warm-up result: prompt tokens the stored state covers now (all of them unless a real request interrupted it)
    size_t warm_done = 0;
    // For /status while the job runs: 0 waiting, 1 prefill, 2 decode; output count and start of the phase
    inline static std::atomic<int64_t> next_id{0};
    int64_t id = ++next_id;
    std::atomic<int> live_state{0}, live_out{0};
    std::atomic<double> live_t0{0};
    int attempt = 0;

    void push(std::string s) {
        {
            std::lock_guard lk(mu);
            pieces.push_back(std::move(s));
        }
        cv.notify_all();
    }
};

// The job to admit next: best priority first, in order within a level. A warm-up is taken only when `busy` is false
// (nothing else runs), so it never costs a running request a slice pause. q.end() if there is none.
std::deque<std::shared_ptr<Job>>::iterator next_job(std::deque<std::shared_ptr<Job>>& q, bool busy);

// Origin from headers (Claude Code: x-claude-code-session-id, -agent-id, with CLAUDE_CODE_GATEWAY_HINT_HEADERS=1
// -request-class) and project from X-Tona-Project, otherwise from "Primary working directory:" in the system prompt
void tag_origin(Job& j, const httplib::Request& req, const json& body);

}  // namespace tona::server
