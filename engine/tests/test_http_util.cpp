// Helpers of the HTTP routes: ids, server-sent events, error bodies, and the waits that watch a job.
#include <chrono>
#include <set>
#include <string>
#include <thread>

#include "server/http_util.h"
#include "tests/test_support.h"

using tona::server::Job;
using tona::server::json;

namespace {

using namespace std::chrono_literals;

bool all_hex(const std::string& s) { return s.find_first_not_of("0123456789abcdef") == std::string::npos; }

void test_ids() {
    std::set<std::string> seen;
    for (int i = 0; i < 2000; i++) {
        const std::string id = tona::server::rand_id("msg_");
        EXPECT(id.rfind("msg_", 0) == 0 && id.size() == 4 + 16 + 8 && all_hex(id.substr(4)));
        seen.insert(id);
    }
    EXPECT(seen.size() == 2000);
}

void test_clock_text() {
    const std::string t = tona::server::hms();
    EXPECT(t.size() == 8 && t[2] == ':' && t[5] == ':');
    EXPECT(t.find_first_not_of("0123456789:") == std::string::npos);
}

void test_model_identity() {
    using tona::server::identity_of;
    // The Coder keeps the id that clients are configured with
    EXPECT(identity_of("Qwen3.8-Flash-Next GSQ-RCO Coder").id == tona::server::kModelName);
    EXPECT(identity_of("Qwen3.8-Flash-Next GSQ-RCO Coder").display_name == "Qwen3.8 Flash Next Coder (local)");
    EXPECT(identity_of("").id == tona::server::kModelName);
    // Another model reports its own name
    const auto full = identity_of("Qwen3.8-Flash-Next GSQ-RCO");
    EXPECT(full.id == "qwen3.8-flash-next-gsq-rco");
    EXPECT(full.display_name == "Qwen3.8-Flash-Next GSQ-RCO (local)");
    EXPECT(identity_of("  ??  ").id == tona::server::kModelName);
    EXPECT(identity_of("Ornith 1.5 / 35B_A3B").id == "ornith-1.5-35b-a3b");
    // The process-wide identity defaults to the Coder and follows set_model_identity
    EXPECT(tona::server::model_id() == tona::server::kModelName);
    tona::server::set_model_identity(full);
    EXPECT(tona::server::model_id() == "qwen3.8-flash-next-gsq-rco");
    EXPECT(tona::server::model_display_name() == "Qwen3.8-Flash-Next GSQ-RCO (local)");
    tona::server::set_model_identity(identity_of(""));
}

void test_sse() {
    EXPECT(tona::server::sse("ping", json{{"a", 1}}) == "event: ping\ndata: {\"a\":1}\n\n");
    EXPECT(tona::server::sse("", json{{"a", 1}}) == "data: {\"a\":1}\n\n");
    // Invalid UTF-8 from the model must not make the stream throw
    EXPECT(tona::server::sse("", json{{"t", "a\xFF"}}) == "data: {\"t\":\"a\xEF\xBF\xBD\"}\n\n");
    EXPECT(tona::server::sse("", json("line\nbreak")) == "data: \"line\\nbreak\"\n\n");
}

void test_error_body() {
    const json e = tona::server::anthropic_error("invalid_request_error", "bad");
    EXPECT(e["type"] == "error" && e["error"]["type"] == "invalid_request_error" && e["error"]["message"] == "bad");
}

void finish_later(Job& j, std::chrono::milliseconds after) {
    std::this_thread::sleep_for(after);
    {
        std::lock_guard lk(j.mu);
        j.done = true;
    }
    j.cv.notify_all();
}

void test_drain_collects_all_pieces() {
    Job j;
    std::thread producer([&] {
        for (const char* p : {"a", "b", "c"}) {
            std::this_thread::sleep_for(20ms);
            j.push(p);
        }
        finish_later(j, 20ms);
    });
    std::string got;
    tona::server::drain(j, [&](const std::string& s) { return got += s, true; }, [] { return true; });
    producer.join();
    EXPECT(got == "abc" && !j.cancel);
}

void test_drain_cancels_when_the_consumer_stops() {
    Job j;
    j.push("x");
    j.push("y");
    int calls = 0;
    tona::server::drain(j, [&](const std::string&) { return ++calls < 1; }, [] { return true; });
    EXPECT(calls == 1 && j.cancel);
}

void test_drain_cancels_when_the_client_is_gone() {
    Job j;  // nothing arrives and the job never ends: only the liveness check can stop the wait
    const auto t0 = std::chrono::steady_clock::now();
    tona::server::drain(j, [](const std::string&) { return true; }, [] { return false; });
    EXPECT(j.cancel);
    EXPECT(std::chrono::steady_clock::now() - t0 < 5s);
}

void test_wait_started() {
    Job ready;
    ready.started = true;
    EXPECT(tona::server::wait_started(ready, [] { return false; }) && !ready.cancel);

    Job late;
    std::thread starter([&] {
        std::this_thread::sleep_for(100ms);
        {
            std::lock_guard lk(late.mu);
            late.started = true;
        }
        late.cv.notify_all();
    });
    EXPECT(tona::server::wait_started(late, [] { return true; }));
    starter.join();
}

// The ping runs every 5 s while the prefill is busy; a false answer means the client left
void test_wait_started_gives_up_when_the_client_leaves() {
    Job j;
    int pings = 0;
    EXPECT(!tona::server::wait_started(j, [&] { return ++pings < 1; }));
    EXPECT(j.cancel && pings == 1);
}

}  // namespace

int main() {
    return tona::test::run("test_http_util", [] {
        test_ids();
        test_clock_text();
        test_model_identity();
        test_sse();
        test_error_body();
        test_drain_collects_all_pieces();
        test_drain_cancels_when_the_consumer_stops();
        test_drain_cancels_when_the_client_is_gone();
        test_wait_started();
        test_wait_started_gives_up_when_the_client_leaves();
    });
}
