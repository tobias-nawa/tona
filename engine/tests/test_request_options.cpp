// Per-request options: sampling defaults, thinking budgets per effort level and role efforts from the environment.
// The environment is read once per process, so the program takes the scenario as its argument: `defaults` (only
// TONA_THINK_BUDGET and TONA_ROLE_EFFORT set) or `temperature` (TONA_TEMPERATURE=0 as well).
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "server/request_options.h"
#include "tests/test_support.h"

using tona::server::Job;
using tona::server::Request;

namespace {

std::unique_ptr<Job> configured(const Request& q) {
    auto j = std::make_unique<Job>();  // a Job holds atomics and a mutex: it cannot be copied or moved
    tona::server::configure_job(*j, q);
    return j;
}

Request request(bool thinking) {
    Request q;
    q.thinking = thinking;
    return q;
}

void test_qwen_defaults_without_a_temperature() {
    const auto t = configured(request(true));
    EXPECT(t->smp.temperature == 1.0f && t->smp.top_p == 0.95f && t->smp.top_k == 20);
    const auto n = configured(request(false));
    EXPECT(n->smp.temperature == 0.7f && n->smp.top_p == 0.8f && n->smp.top_k == 20);
}

void test_request_values_win() {
    Request q = request(true);
    q.top_p = 0.5f, q.top_k = 7, q.seed = 99;
    auto j = configured(q);
    EXPECT(j->smp.temperature == 1.0f && j->smp.top_p == 0.5f && j->smp.top_k == 7 && j->smp.seed == 99);
    q.temperature = 0.3f;
    j = configured(q);
    // An explicit temperature switches the recommended defaults off
    EXPECT(j->smp.temperature == 0.3f && j->smp.top_p == 0.5f && j->smp.top_k == 7);
    q = request(false);
    q.temperature = 0;
    j = configured(q);
    EXPECT(j->smp.temperature == 0 && j->smp.top_p == 1 && j->smp.top_k == 0);  // greedy
}

void test_out_of_range_values() {
    Request q = request(false);
    q.temperature = 0.2f;
    for (float p : {0.0f, -1.0f, 1.5f}) {
        q.top_p = p;
        EXPECT(configured(q)->smp.top_p == 1);
    }
    q.top_p = 1, q.top_k = -4;
    EXPECT(configured(q)->smp.top_k == 0);
}

void test_copied_fields() {
    Request q = request(true);
    q.max_tokens = 321, q.effort = "medium", q.stops = {"END"};
    const auto j = configured(q);
    EXPECT(j->max_tokens == 321 && j->thinking && j->effort == "medium" && j->stops == q.stops);
}

void test_warm_up_job() {
    Request q = request(true);
    EXPECT(!configured(q)->warm && configured(q)->role == "main");
    q.max_tokens = 0, q.warm = true;
    const auto j = configured(q);
    EXPECT(j->warm && j->role == "warm" && j->prio() == 3);
}

// TONA_THINK_BUDGET="low=2048,medium=8192,high=oops,xhigh=-1,broken"
void test_thinking_budget() {
    Request q = request(true);
    q.effort = "low";
    EXPECT(configured(q)->smp.think_budget == 2048);
    q.effort = "medium";
    EXPECT(configured(q)->smp.think_budget == 8192);
    q.effort = "high";
    EXPECT(configured(q)->smp.think_budget == -1);  // unreadable entry: ignored
    q.effort = "xhigh";
    EXPECT(configured(q)->smp.think_budget == -1);
    q.effort = "max";
    EXPECT(configured(q)->smp.think_budget == -1);  // no entry
    q.effort = "low", q.think_budget = 5;
    EXPECT(configured(q)->smp.think_budget == 5);  // an explicit budget beats the table
    q.think_budget = 0;
    EXPECT(configured(q)->smp.think_budget == 0);
    q.thinking = false;
    EXPECT(configured(q)->smp.think_budget == -1);  // no thinking, no budget
}

// TONA_ROLE_EFFORT="subagent=low,auxiliary=bogus,main"
void test_role_effort() {
    EXPECT(tona::server::role_effort("subagent", "xhigh") == "low");
    EXPECT(tona::server::role_effort("auxiliary", "xhigh") == "high");  // an unknown level is high
    EXPECT(tona::server::role_effort("main", "max") == "max");          // entry without "=" is ignored
    EXPECT(tona::server::role_effort("other", "medium") == "medium");
}

void test_fixed_default_temperature() {
    auto j = configured(request(true));
    EXPECT(j->smp.temperature == 0 && j->smp.top_p == 1 && j->smp.top_k == 0);
    Request q = request(true);
    q.temperature = 0.4f;
    j = configured(q);
    EXPECT(j->smp.temperature == 0.4f);
}

}  // namespace

int main(int argc, char** argv) {
    const bool fixed = argc > 1 && std::strcmp(argv[1], "temperature") == 0;
    return tona::test::run("test_request_options", [fixed] {
        tona::test::set_env("TONA_THINK_BUDGET", "low=2048,medium=8192,high=oops,xhigh=-1,broken");
        tona::test::set_env("TONA_ROLE_EFFORT", "subagent=low,auxiliary=bogus,main");
        tona::test::unset_env("TONA_TEMPERATURE");
        if (fixed) tona::test::set_env("TONA_TEMPERATURE", "0");
        if (fixed) {
            test_fixed_default_temperature();
        } else {
            test_qwen_defaults_without_a_temperature();
            test_request_values_win();
            test_out_of_range_values();
        }
        test_copied_fields();
        test_warm_up_job();
        test_thinking_budget();
        test_role_effort();
    });
}
