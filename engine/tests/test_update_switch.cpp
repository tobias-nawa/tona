// Switching to a new version and back: idle wait, restart, health check, automatic and manual rollback, and the
// command runner. The server and systemctl are replaced; no process is started except `true`, `false` and `sleep`.
#include <filesystem>
#include <fstream>
#include <string>

#include "tests/test_support.h"
#include "tests/update_testkit.h"
#include "tests/update_teststatus.h"
#include "update/activate.h"
#include "update/fsutil.h"
#include "update/switch.h"

using namespace tona::test;
namespace fs = std::filesystem;
namespace {

// Install root with the given versions; each has a marker file naming how the "server" of that version behaves
struct Root {
    TempDir dir;
    up::Layout layout;

    explicit Root(const std::string& name) : dir(name) {
        layout.root = dir.path() / "tona";
        fs::create_directories(layout.versions());
        ::chmod(layout.root.c_str(), 0700);
    }
    void add(const std::string& version, const std::string& behavior) const {
        fs::create_directories(layout.version(version));
        std::ofstream(layout.version(version) / "marker") << behavior;
    }
    std::string current() const { return up::linked_version(layout, "current"); }
    std::string previous() const { return up::linked_version(layout, "previous"); }
    std::string current_behavior() const {
        std::ifstream in(layout.version(current()) / "marker");
        std::string behavior;
        in >> behavior;
        return behavior;
    }
};

struct Harness {
    FakeStatus status;
    Root root;
    int restarts = 0;
    std::vector<std::string> started;
    bool restart_fails = false;
    up::SwitchConfig cfg;

    explicit Harness(const std::string& name) : root(name) {
        cfg.status_url = status.url();
        cfg.ready_seconds = 2;
        cfg.idle_seconds = 1;
        cfg.poll_ms = 20;
        cfg.restart = [this](std::string& why) {
            restarts++;
            if (restart_fails) {
                why = "systemctl failed";
                return false;
            }
            const std::string v = root.current();
            started.push_back(v);
            std::ifstream in(root.layout.version(v) / "marker");
            std::string behavior;
            in >> behavior;
            status.set(behavior == "good" ? FakeStatus::Mode::Idle : FakeStatus::Mode::Loading);
            return true;
        };
        // "crash": the process is gone as soon as it was started, as a unit that failed
        cfg.alive = [this] { return root.current_behavior() != "crash"; };
    }
    // 0.3.0 runs and is `previous`; the update to 0.4.0 has been installed (`current`)
    void installed(const std::string& new_behavior) const {
        root.add("0.3.0", "good");
        root.add("0.4.0", new_behavior);
        up::activate_version(root.layout, "0.3.0");
        up::activate_version(root.layout, "0.4.0");
    }
};

void test_good_update() {
    Harness h("switch-good");
    h.installed("good");
    std::vector<std::string> said;
    h.cfg.say = [&](const std::string& t) { said.push_back(t); };
    const auto out = up::switch_to_current(h.root.layout, h.cfg);
    EXPECT(out.kind == up::SwitchOutcome::Kind::Switched);
    EXPECT(h.restarts == 1);
    EXPECT(h.root.current() == "0.4.0");
    EXPECT(!said.empty());
}

void test_failing_update_rolls_back() {
    Harness h("switch-bad");
    h.installed("bad");
    const auto out = up::switch_to_current(h.root.layout, h.cfg);
    EXPECT(out.kind == up::SwitchOutcome::Kind::RolledBack);
    EXPECT(h.restarts == 2);
    EXPECT((h.started == std::vector<std::string>{"0.4.0", "0.3.0"}));
    EXPECT(h.root.current() == "0.3.0");
    EXPECT(h.root.previous() == "0.4.0");
    EXPECT(out.message.contains("did not become ready"));
}

// The new server is gone: the switch hears it at once, not after the whole load time, and goes back
void test_exited_server_is_noticed_at_once() {
    Harness h("switch-crash");
    h.installed("crash");
    h.cfg.ready_seconds = 30;
    const auto start = std::chrono::steady_clock::now();
    const auto out = up::switch_to_current(h.root.layout, h.cfg);
    EXPECT(std::chrono::steady_clock::now() - start < std::chrono::seconds(5));
    EXPECT(out.kind == up::SwitchOutcome::Kind::RolledBack && out.message.contains("the new server exited"));
    EXPECT(h.root.current() == "0.3.0" && h.restarts == 2);
}

void test_both_fail() {
    Harness h("switch-both");
    h.root.add("0.3.0", "bad");
    h.root.add("0.4.0", "bad");
    up::activate_version(h.root.layout, "0.3.0");
    up::activate_version(h.root.layout, "0.4.0");
    EXPECT(up::switch_to_current(h.root.layout, h.cfg).kind == up::SwitchOutcome::Kind::Failed);
    EXPECT(h.restarts == 2);
}

void test_restart_command_fails() {
    Harness h("switch-restart");
    h.installed("good");
    h.restart_fails = true;
    const auto out = up::switch_to_current(h.root.layout, h.cfg);
    EXPECT(out.kind == up::SwitchOutcome::Kind::Failed);
    EXPECT(out.message.contains("systemctl failed"));
    EXPECT(h.root.current() == "0.3.0");  // went back before trying again
}

void test_no_previous() {
    Harness h("switch-noprev");
    h.root.add("0.4.0", "bad");
    up::activate_version(h.root.layout, "0.4.0");
    const auto out = up::switch_to_current(h.root.layout, h.cfg);
    EXPECT(out.kind == up::SwitchOutcome::Kind::Failed);
    EXPECT(out.message.contains("no previous"));
    EXPECT(!up::roll_back_to_previous(h.root.layout, h.cfg).message.empty());
    EXPECT(up::roll_back_to_previous(h.root.layout, h.cfg).kind == up::SwitchOutcome::Kind::Failed);
}

void test_waits_for_idle() {
    Harness h("switch-busy");
    h.installed("good");
    h.status.set(FakeStatus::Mode::Busy);
    h.cfg.idle_seconds = 1;
    const auto out = up::switch_to_current(h.root.layout, h.cfg);
    EXPECT(out.kind == up::SwitchOutcome::Kind::Failed);
    EXPECT(h.restarts == 0);  // a running request is never interrupted
    EXPECT(up::roll_back_to_previous(h.root.layout, h.cfg).kind == up::SwitchOutcome::Kind::Failed);
    EXPECT(h.restarts == 0);
    EXPECT(h.root.current() == "0.4.0");
    // The request ends while we wait
    h.cfg.idle_seconds = 5;
    std::thread finish([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        h.status.set(FakeStatus::Mode::Idle);
    });
    EXPECT(up::switch_to_current(h.root.layout, h.cfg).kind == up::SwitchOutcome::Kind::Switched);
    finish.join();
    EXPECT(h.restarts == 1);
}

void test_manual_rollback() {
    Harness h("switch-manual");
    h.installed("good");
    const auto out = up::roll_back_to_previous(h.root.layout, h.cfg);
    EXPECT(out.kind == up::SwitchOutcome::Kind::Switched);
    EXPECT(h.root.current() == "0.3.0");
    EXPECT(h.root.previous() == "0.4.0");
    EXPECT(h.started.back() == "0.3.0");
}

void test_cancel() {
    Harness h("switch-cancel");
    h.installed("good");
    h.status.set(FakeStatus::Mode::Busy);
    std::atomic<bool> stop{true};
    h.cfg.cancel = &stop;
    h.cfg.idle_seconds = 30;
    const auto begin = std::chrono::steady_clock::now();
    EXPECT(up::switch_to_current(h.root.layout, h.cfg).kind == up::SwitchOutcome::Kind::Failed);
    EXPECT(std::chrono::steady_clock::now() - begin < std::chrono::seconds(5));
}

void test_status_probes() {
    FakeStatus s;
    EXPECT(up::server_is_ready(s.url()) && up::server_is_idle(s.url()));
    s.set(FakeStatus::Mode::Busy);
    EXPECT(up::server_is_ready(s.url()) && !up::server_is_idle(s.url()));
    s.set(FakeStatus::Mode::Loading);
    EXPECT(!up::server_is_ready(s.url()) && up::server_is_idle(s.url()));
    s.set(FakeStatus::Mode::Error);
    EXPECT(!up::server_is_ready(s.url()));
    EXPECT(up::server_is_idle("http://127.0.0.1:1"));  // nobody answers: nothing to interrupt
    EXPECT(!up::server_is_ready("http://127.0.0.1:1"));
}

void test_run_command() {
    std::string why;
    EXPECT(up::run_command({"true"}, 5, why));
    EXPECT(!up::run_command({"false"}, 5, why));
    EXPECT(why.contains("failed"));
    EXPECT(!up::run_command({"tona-no-such-program-xyz"}, 5, why));
    const auto begin = std::chrono::steady_clock::now();
    EXPECT(!up::run_command({"sleep", "10"}, 1, why));
    EXPECT(why.contains("timed out"));
    EXPECT(std::chrono::steady_clock::now() - begin < std::chrono::seconds(4));
}

// A fake systemctl on PATH writes down its arguments: reset-failed must come before restart, and its failure is ignored
void test_systemctl_resets_the_start_limit_first() {
    TempDir dir("switch-systemctl");
    const fs::path bin = dir.path() / "bin", log = dir.path() / "calls.log";
    fs::create_directories(bin);
    // reset-failed exits 1 (nothing was failed); restart must still run
    std::ofstream(bin / "systemctl") << "#!/bin/sh\necho \"$*\" >> " << log
                                     << "\n[ \"$2\" = reset-failed ] && exit 1\nexit 0\n";
    fs::permissions(bin / "systemctl", fs::perms::owner_all);
    const std::string old_path = std::getenv("PATH");
    ::setenv("PATH", (bin.string() + ":" + old_path).c_str(), 1);
    std::string why;
    const bool ok = up::restart_with_systemctl(why);
    ::setenv("PATH", old_path.c_str(), 1);
    EXPECT(ok);
    std::ifstream in(log);
    const std::string calls((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT(calls == "--user reset-failed tona-server\n--user restart tona-server\n");
}

void test_activate_links() {
    Root r("switch-links");
    r.add("0.1.0", "good");
    r.add("0.2.0", "good");
    EXPECT(!up::swap_current_previous(r.layout));
    up::activate_version(r.layout, "0.1.0");
    EXPECT(r.current() == "0.1.0" && r.previous().empty());
    up::activate_version(r.layout, "0.1.0");  // again: previous must not become current itself
    EXPECT(r.previous().empty());
    up::activate_version(r.layout, "0.2.0");
    EXPECT(r.previous() == "0.1.0");
    EXPECT(up::swap_current_previous(r.layout));
    EXPECT(r.current() == "0.1.0" && r.previous() == "0.2.0");
    fs::remove_all(r.layout.version("0.2.0"));
    EXPECT(!up::swap_current_previous(r.layout));  // previous vanished: no dangling current
    EXPECT(r.current() == "0.1.0");
    EXPECT(up::linked_version(r.layout, "nothing").empty());
}

}  // namespace

int main() {
    return run("update_switch", [] {
        test_good_update();
        test_failing_update_rolls_back();
        test_exited_server_is_noticed_at_once();
        test_both_fail();
        test_restart_command_fails();
        test_no_previous();
        test_waits_for_idle();
        test_manual_rollback();
        test_cancel();
        test_status_probes();
        test_run_command();
        test_systemctl_resets_the_start_limit_first();
        test_activate_links();
    });
}
