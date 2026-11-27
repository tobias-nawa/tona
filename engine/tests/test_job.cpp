// Job origin: headers and the working directory in the system prompt decide project, session and role.
#include <deque>
#include <memory>
#include <string>

#include "server/job.h"
#include "tests/test_support.h"

using tona::server::Job;
using tona::server::json;

namespace {

httplib::Request request(std::initializer_list<std::pair<const char*, const char*>> headers) {
    httplib::Request r;
    for (const auto& [k, v] : headers) r.set_header(k, v);
    return r;
}

void test_priorities_and_ids() {
    Job a, b;
    EXPECT(b.id == a.id + 1);
    a.role = "auxiliary", b.role = "subagent";
    Job main_job;
    EXPECT(a.prio() < main_job.prio() && main_job.prio() < b.prio());
}

// The warm-up is the last priority, and the queue hands it out only when nothing else runs
void test_warm_up_queue() {
    using JobPtr = std::shared_ptr<Job>;
    auto make = [](const char* role, bool warm) {
        auto j = std::make_shared<Job>();
        j->role = role, j->warm = warm;
        return j;
    };
    const JobPtr sub = make("subagent", false), warm = make("warm", true), mainj = make("main", false);
    EXPECT(warm->prio() > sub->prio() && sub->prio() > mainj->prio());
    std::deque<JobPtr> q;
    EXPECT(tona::server::next_job(q, false) == q.end());
    q = {warm};
    EXPECT(tona::server::next_job(q, true) == q.end());  // something runs: the warm-up waits
    EXPECT(tona::server::next_job(q, false) == q.begin());
    q = {warm, sub, mainj};
    EXPECT(*tona::server::next_job(q, true) == mainj);  // a real request goes first, also while something runs
    q = {warm, sub};
    EXPECT(*tona::server::next_job(q, true) == sub && *tona::server::next_job(q, false) == sub);
    const JobPtr main2 = make("main", false);
    q = {mainj, main2};
    EXPECT(*tona::server::next_job(q, false) == mainj);  // in order within a level
}

void test_push_keeps_order() {
    Job j;
    j.push("a");
    j.push("b");
    EXPECT(j.pieces.size() == 2 && j.pieces[0] == "a" && j.pieces[1] == "b");
}

void test_role_from_headers() {
    Job j;
    tona::server::tag_origin(j, request({{"x-claude-code-session-id", "s1"}}), json::object());
    EXPECT(j.session == "s1" && j.agent.empty() && j.role == "main");
    Job sub;
    tona::server::tag_origin(sub, request({{"x-claude-code-agent-id", "ag"}}), json::object());
    EXPECT(sub.agent == "ag" && sub.role == "subagent");
    Job aux;
    tona::server::tag_origin(
        aux, request({{"x-claude-code-request-class", "auxiliary"}, {"x-claude-code-agent-id", "ag"}}), json::object());
    EXPECT(aux.role == "auxiliary");
    Job odd;
    tona::server::tag_origin(odd, request({{"x-claude-code-request-class", "other"}}), json::object());
    EXPECT(odd.role == "main");
}

void test_project_from_header_or_system_prompt() {
    Job h;
    tona::server::tag_origin(h, request({{"x-tona-project", "from-header"}}),
                             json::parse(R"({"system": "Primary working directory: /home/u/other"})"));
    EXPECT(h.project == "from-header");

    Job s;
    tona::server::tag_origin(
        s, request({}),
        json::parse(R"({"system": [{"type": "text", "text": "x\nPrimary working directory: /home/u/proj/ \nmore"}]})"));
    EXPECT(s.project == "proj");

    Job m;
    tona::server::tag_origin(m, request({}), json::parse(R"({"messages": [{"role": "user", "content": "hi"},
        {"role": "system", "content": "Primary working directory: /srv/app\r\n"}]})"));
    EXPECT(m.project == "app");

    Job none;
    tona::server::tag_origin(none, request({}), json::parse(R"({"system": 5, "messages": [5, {"role": "system"}]})"));
    EXPECT(none.project.empty());
}

}  // namespace

int main() {
    return tona::test::run("test_job", [] {
        test_priorities_and_ids();
        test_warm_up_queue();
        test_push_keeps_order();
        test_role_from_headers();
        test_project_from_header_or_system_prompt();
    });
}
