#include "server/request_options.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <map>

#include "backend/env.h"

namespace tona::server {

namespace {

// Parses "key=value,key=value" from an environment variable; value_of turns the text after "=" into T and returns
// false if it cannot (the entry is then ignored with a warning)
template <class T, class F>
std::map<std::string, T> parse_pairs(const char* env_name, F value_of) {
    std::map<std::string, T> t;
    if (const char* e = tona::env::text(env_name))
        for (std::string s = e; !s.empty();) {
            const size_t c = s.find(','), q = s.find('=');
            const std::string kv = s.substr(0, c);
            T v{};
            if (q < kv.size() && value_of(kv.substr(q + 1), v))
                t[kv.substr(0, q)] = v;
            else
                fprintf(stderr, "%s: ignoring entry \"%s\"\n", env_name, kv.c_str());
            s = c == std::string::npos ? "" : s.substr(c + 1);
        }
    return t;
}

// Thinking budget per effort level: TONA_THINK_BUDGET="low=2048,medium=8192,..." (-1 or missing: unlimited)
int think_budget_of(const std::string& effort) {
    static const std::map<std::string, int> table =
        parse_pairs<int>("TONA_THINK_BUDGET", [](const std::string& text, int& out) {
            const auto r = std::from_chars(text.data(), text.data() + text.size(), out);
            return r.ec == std::errc() && r.ptr == text.data() + text.size();
        });
    const auto it = table.find(effort);
    return it == table.end() ? -1 : it->second;
}

// Sampling of the request. Without a temperature (Claude Code sends none; the Anthropic API then means T 1) Qwen's
// recommendation: with thinking T 1.0 / top_p 0.95 / top_k 20, without T 0.7 / 0.8 / 20; the request's top_p and
// top_k still apply. Greedy with thinking runs into loops (6 of 8 hard tasks,
// research/measurements/2026-10-26-effort.md). TONA_TEMPERATURE fixes the default (0: greedy as until October 2026)
Sampling sampling_of(const Request& q) {
    static const double t_env = tona::env::number("TONA_TEMPERATURE", NAN);  // NaN: unset
    Sampling s;
    s.top_p = q.top_p > 0 && q.top_p <= 1 ? q.top_p : 1;
    s.top_k = std::max(0, q.top_k);
    s.seed = q.seed;
    if (q.temperature >= 0) {
        s.temperature = q.temperature;
    } else if (!std::isnan(t_env)) {
        s.temperature = (float)t_env;
    } else {
        s.temperature = q.thinking ? 1.0f : 0.7f;
        if (s.top_p >= 1) s.top_p = q.thinking ? 0.95f : 0.8f;
        if (s.top_k == 0) s.top_k = 20;
    }
    return s;
}

}  // namespace

std::string role_effort(const std::string& role, const std::string& effort) {
    static const std::map<std::string, std::string> table =
        parse_pairs<std::string>("TONA_ROLE_EFFORT", [](const std::string& text, std::string& out) {
            out = effort_level(text);
            return true;
        });
    const auto it = table.find(role);
    return it == table.end() ? effort : it->second;
}

void configure_job(Job& job, const Request& q) {
    job.max_tokens = q.max_tokens;
    if (q.warm) job.warm = true, job.role = "warm";
    job.smp = sampling_of(q);
    job.thinking = q.thinking, job.effort = q.effort;
    job.smp.think_budget = !q.thinking ? -1 : q.think_budget >= 0 ? q.think_budget : think_budget_of(q.effort);
    job.stops = q.stops;
}

}  // namespace tona::server
