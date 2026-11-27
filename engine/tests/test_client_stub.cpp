// The feed against a stub HTTP server with a recorded-shape /status: the poller thread of tona, the typed view and
// the Start page, and what the screen shows when the server goes away. Port 8673 is the test port of this work; the
// server on 8662 is never touched.
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

#include "client/commands.h"
#include "client/feed.h"
#include "client/screen.h"
#include "client/term.h"
#include "tests/client_testkit.h"
#include "third_party/httplib.h"

using namespace tona::client;
using namespace tona::test;
namespace top = tona::top;

namespace {

constexpr int kPort = 8673;
std::string g_data;

// Serves the document; each request advances the decode token count of the first running request by 100
class StubServer {
public:
    explicit StubServer(const std::string& file) : doc_(nlohmann::json::parse(read_text(file))) {
        svr_.Get("/status", [this](const httplib::Request&, httplib::Response& res) {
            nlohmann::json d = doc_;
            d["engine"]["active"][0]["out_tokens"] = 480 + 100 * polls_++;
            res.set_content(d.dump(), "application/json");
        });
        thread_ = std::thread([this] { svr_.listen("127.0.0.1", kPort); });
        svr_.wait_until_ready();
    }
    ~StubServer() { stop(); }
    void stop() {
        svr_.stop();
        if (thread_.joinable()) thread_.join();
    }
    int polls() const { return polls_; }

private:
    nlohmann::json doc_;
    httplib::Server svr_;
    std::thread thread_;
    std::atomic<int> polls_{0};
};

void test_fetch_and_commands_against_the_stub() {
    StubServer stub(g_data + "/status-decode.json");
    const ServerView v = fetch_server_view("http://127.0.0.1:" + std::to_string(kPort));
    EXPECT(v.up && v.ready && v.conversation_count() == 2);
    stub.stop();
    const ServerView gone = fetch_server_view("http://127.0.0.1:" + std::to_string(kPort));
    EXPECT(!gone.up && !gone.error.empty());
}

void test_poller_feeds_the_start_page() {
    StubServer stub(g_data + "/status-decode.json");
    top::Shared sh;
    std::atomic<bool> quit{false}, log{false};
    // A clock that moves exactly 250 ms per poll: the rate is then the stub's 100 tokens per poll and nothing else.
    // With the real clock a slow runner's longer sleeps lowered it below the bound (macOS CI, 2026-11-26)
    std::atomic<int> ticks{0};
    const top::PollerConfig cfg{"http://127.0.0.1:" + std::to_string(kPort), false, true, &quit, &log, [] {},
                                [&ticks] { return 0.25 * ticks++; }};
    std::thread poller([&] { top::poll_status(sh, cfg); });
    for (int i = 0; i < 40 && stub.polls() < 6; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    UiState ui = test_state("status-decode.json", g_data);
    fill_from_feed(ui, sh);
    EXPECT(ui.server.up && ui.server.conversation_count() == 2);
    // The rate comes from the token counts of consecutive polls: 100 tokens per 250 ms of the test's clock
    double best = 0;
    for (double r : ui.rates) best = std::max(best, r);
    EXPECT(std::abs(best - 400) < 1e-3);
    const std::string page = joined(render(ui));
    EXPECT(has(strip_color(page), "tokens/s") && has(strip_color(page), "last "));
    stub.stop();
    for (int i = 0; i < 30 && sh.up; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    fill_from_feed(ui, sh);
    EXPECT(!ui.server.up && !ui.server.error.empty());  // the screen sees the server go, and why
    {
        // The restart is done and the new server answers: the connection error of the gap is gone, so the page shows
        // the server and never "Load failed: Could not establish connection"
        StubServer again(g_data + "/status-decode.json");
        for (int i = 0; i < 30 && !sh.up; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        UiState back = ui;
        fill_from_feed(back, sh);
        EXPECT(back.server.up && back.server.error.empty());
        EXPECT(!has(strip_color(joined(render(back))), "Load failed"));
    }
    for (int i = 0; i < 30 && sh.up; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    quit = true;
    poller.join();
    EXPECT(has(strip_color(joined(render(ui))), "Server not running"));
}

}  // namespace

int main(int argc, char** argv) {
    return run("client_stub", [argc, argv] {
        g_data = argc > 1 ? argv[1] : ".";  // inside run: the copy may throw, and run reports it
        test_fetch_and_commands_against_the_stub();
        test_poller_feeds_the_start_page();
    });
}
