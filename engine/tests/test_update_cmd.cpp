// `tona update`: check, show, ask, install, restart, and the way back, against a fake release server and a fake status
// endpoint (the same kit as the updater tests); plus the pieces of `tona start --foreground` that need no process.
#include <sstream>

#include "client/foreground.h"
#include "client/update_cmd.h"
#include "tests/client_testkit.h"
#include "tests/update_testkit.h"
#include "tests/update_teststatus.h"
#include "update/activate.h"
#include "update/fsutil.h"
#include "update/state.h"

using namespace tona::test;
namespace {

struct Rig {
    TempDir dir;
    TestServer server;
    FakeStatus status;
    up::UpdateEnv env;
    std::atomic<int> restarts{0};
    std::atomic<bool> next_start_fails{false};
    std::ostringstream out, err;

    explicit Rig(const std::string& name) : dir(name), env(make_env(server, dir.path())) {}

    up::SwitchConfig switch_config() {
        up::SwitchConfig sw;
        sw.status_url = status.url();
        sw.poll_ms = 20;
        sw.ready_seconds = 1;
        sw.idle_seconds = 2;
        sw.restart = [this](std::string&) {
            restarts++;
            // A server that never gets ready, once: the new version that does not start
            status.set(next_start_fails.exchange(false) ? FakeStatus::Mode::Loading : FakeStatus::Mode::Idle);
            return true;
        };
        return sw;
    }
    std::unique_ptr<up::Updater> updater(bool release = true) {
        return std::make_unique<up::Updater>(env, switch_config(), release);
    }
    // The command as `tona update` runs it; `answer` is the user's reply, nullopt: no terminal
    int run(up::Updater& u, const std::vector<std::string>& args, std::optional<bool> answer) {
        out.str(""), err.str("");
        tona::client::Io io;
        io.out = &out, io.err = &err;
        if (answer) io.ask = [&, a = *answer](const std::string& q) { return asked = q, a; };
        return tona::client::run_update(u, args, io, std::chrono::milliseconds(10));
    }
    std::string asked;
    std::string current() const { return up::linked_version(env.layout, "current"); }
};

void test_a_checkout_has_no_update() {
    Rig rig("cmd-update-source");
    rig.server.serve_latest(make_release("0.3.0", 3));
    const auto u = rig.updater(false);
    EXPECT(rig.run(*u, {}, true) == tona::client::kExitFailed && has(rig.err.str(), "git pull"));
    EXPECT(rig.run(*u, {"--bogus"}, true) == tona::client::kExitUsage);
    EXPECT(rig.server.hits("/releases/latest/download/manifest.json") == 0);
}

void test_up_to_date_changes_nothing() {
    Rig rig("cmd-update-current");
    rig.server.serve_latest(make_release("0.3.0", 3));
    tona::update::ensure_private_root(rig.env.layout);
    EXPECT(tona::update::save_state(rig.env.layout, {3, false}).empty());
    const auto u = rig.updater();
    EXPECT(rig.run(*u, {}, true) == 0 && has(rig.out.str(), "Up to date"));
    EXPECT(rig.asked.empty() && rig.restarts == 0);
}

void test_shows_asks_installs_and_restarts() {
    Rig rig("cmd-update-install");
    rig.server.serve_latest(make_release("0.3.0", 3));
    const auto u = rig.updater();
    // Without a terminal and without --yes: it shows what it found and stops with the exit code for a question
    EXPECT(rig.run(*u, {}, std::nullopt) == tona::client::kExitNeedsConfirm);
    EXPECT(has(rig.out.str(), "Version 0.3.0 is available") && has(rig.out.str(), "MB") &&
           has(rig.out.str(), "Test release 0.3.0"));
    EXPECT(has(rig.err.str(), "--yes") && rig.current().empty() && rig.restarts == 0);
    // The user says no
    EXPECT(rig.run(*u, {}, false) == 0 && has(rig.asked, "Install it and restart the server?"));
    EXPECT(rig.current().empty() && rig.restarts == 0);
    // The user says yes
    EXPECT(rig.run(*u, {}, true) == 0);
    EXPECT(rig.current() == "0.3.0" && rig.restarts == 1);
    EXPECT(has(rig.out.str(), "running the new version"));
    EXPECT(tona::update::load_state(rig.env.layout).state.seq == 3);
}

void test_yes_needs_no_terminal() {
    Rig rig("cmd-update-yes");
    rig.server.serve_latest(make_release("0.3.0", 3));
    const auto u = rig.updater();
    EXPECT(rig.run(*u, {"--yes"}, std::nullopt) == 0 && rig.current() == "0.3.0" && rig.restarts == 1);
}

// The new version does not start: back to the old one, reported as a failure, and the release is not offered again
void test_failing_release_rolls_back_and_is_not_offered_again() {
    Rig rig("cmd-update-rollback");
    rig.server.serve_latest(make_release("0.3.0", 3));
    {
        const auto first = rig.updater();
        EXPECT(rig.run(*first, {"--yes"}, std::nullopt) == 0 && rig.current() == "0.3.0");
    }
    rig.server.serve_latest(make_release("0.4.0", 4));
    rig.next_start_fails = true;
    const auto u = rig.updater();
    EXPECT(rig.run(*u, {"--yes"}, std::nullopt) == tona::client::kExitFailed);
    EXPECT(has(rig.err.str(), "the previous one runs again") && rig.current() == "0.3.0");
    // The sequence of the failed release is kept: 0.4.0 is not offered again
    EXPECT(tona::update::load_state(rig.env.layout).state.seq == 4);
    EXPECT(rig.run(*u, {}, true) == 0 && has(rig.out.str(), "Up to date") && rig.asked.empty());
}

void test_rollback_on_purpose() {
    Rig rig("cmd-update-back");
    for (const char* v : {"0.3.0", "0.4.0"}) {
        rig.server.serve_latest(make_release(v, v[2] == '3' ? 3 : 4));
        const auto u = rig.updater();
        EXPECT(rig.run(*u, {"--yes"}, std::nullopt) == 0);
    }
    EXPECT(rig.current() == "0.4.0");
    const auto u = rig.updater();
    EXPECT(rig.run(*u, {"--rollback"}, false) == 0 && has(rig.asked, "previous version") && rig.current() == "0.4.0");
    EXPECT(rig.run(*u, {"--rollback"}, true) == 0 && rig.current() == "0.3.0");
    EXPECT(has(rig.out.str(), "running the previous version"));
}

void test_unit_name_from_the_cgroup() {
    using tona::client::own_unit_name;
    EXPECT(own_unit_name("0::/user.slice/user-1000.slice/user@1000.service/app.slice/tona-server.service\n") ==
           "tona-server.service");
    EXPECT(own_unit_name("0::/user.slice/user-1000.slice/session-3.scope\n").empty());  // a terminal, not the unit
    EXPECT(own_unit_name("").empty());
}

// Without a model file nothing starts; a checkout is not a release installation
void test_foreground_plan_refusals() {
    TempDir dir("cmd-foreground");
    ::setenv("HOME", dir.path().c_str(), 1);
    const tona::client::Env release = tona::client::Env::for_release((dir.path() / "root").string());
    const tona::client::ForegroundPlan p = tona::client::plan_foreground(release);
    EXPECT(has(p.error, "model file missing") && has(p.error, "tona setup"));
    const tona::client::ForegroundPlan checkout =
        tona::client::plan_foreground(tona::client::Env::for_checkout(dir.path().string()));
    EXPECT(has(checkout.error, "just start"));
}

}  // namespace

int main() {
    return run("update_cmd", [] {
        test_a_checkout_has_no_update();
        test_up_to_date_changes_nothing();
        test_shows_asks_installs_and_restarts();
        test_yes_needs_no_terminal();
        test_failing_release_rolls_back_and_is_not_offered_again();
        test_rollback_on_purpose();
        test_unit_name_from_the_cgroup();
        test_foreground_plan_refusals();
    });
}
