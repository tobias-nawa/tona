// The control centre without a terminal: the start on a free port ends in a restart of the program on the new port.
// The server is a fake; nothing listens, so any port above 8670 will do (8662 is never used).
#include <chrono>
#include <filesystem>
#include <thread>

#include "client/app.h"
#include "client/env.h"
#include "core/sys.h"
#include "tests/client_testkit.h"

using namespace tona::client;
using namespace tona::test;
namespace fs = std::filesystem;

namespace tona::client {

// Friend of Center: reaches the private action and the state it leaves behind
struct CenterProbe {
    static void start_on(Center& c, int port) {
        c.ui_.free_port = port;
        c.do_start(true);
    }
    static void wait_idle_and_drain(Center& c) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (c.busy_ && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        c.drain_posted();
    }
    static bool quit(const Center& c) { return c.quit_; }
    static const std::vector<std::string>& argv(const Center& c) { return c.exec_.argv; }
    static const Env& env(const Center& c) { return c.env_; }
};

}  // namespace tona::client

namespace {

constexpr int kOldPort = 8680, kNewPort = 8681;

struct FakeControl : Control {
    std::string start_error;
    std::string start() override { return start_error; }
    std::string stop() override { return ""; }
    std::string restart() override { return ""; }
    std::string download(const std::string&, const std::string&, const LineFn&) override { return ""; }
    std::string remove(const std::string&) override { return ""; }
};

Env checkout_on(const TempDir& dir) {
    fs::create_directories(dir.path() / "work");
    Env env = Env::for_checkout(dir.path().string());
    env.port = kOldPort;
    return env;
}

// A url that is not local: the centre starts no updater and no network thread
CenterOptions remote() {
    CenterOptions o;
    o.url = "http://example.invalid:8680";
    return o;
}

void test_free_port_restarts_the_program_on_the_new_port() {
    TempDir dir("center-restart");
    Center center(checkout_on(dir), remote(), std::make_unique<FakeControl>());
    CenterProbe::start_on(center, kNewPort);
    CenterProbe::wait_idle_and_drain(center);
    EXPECT(CenterProbe::quit(center));
    const std::vector<std::string>& argv = CenterProbe::argv(center);
    EXPECT(argv.size() == 3);
    if (argv.size() != 3) return;
    EXPECT(argv[0] == tona::install::self_exe());
    EXPECT(argv[1] == "--url");
    EXPECT(argv[2] == "http://127.0.0.1:" + std::to_string(kNewPort));
    EXPECT(CenterProbe::env(center).port == kNewPort);
}

void test_failed_start_does_not_restart() {
    TempDir dir("center-no-restart");
    auto control = std::make_unique<FakeControl>();
    control->start_error = "port busy";
    Center center(checkout_on(dir), remote(), std::move(control));
    CenterProbe::start_on(center, kNewPort);
    CenterProbe::wait_idle_and_drain(center);
    EXPECT(!CenterProbe::quit(center));
    EXPECT(CenterProbe::argv(center).empty());
}

}  // namespace

int main() {
    return run("client_center", [] {
        test_free_port_restarts_the_program_on_the_new_port();
        test_failed_start_does_not_restart();
    });
}
