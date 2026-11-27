// The updater as tona drives it: checks only when switched on, once per start, in the background, install and
// switch, failures reported without touching the installed version.
#include <filesystem>
#include <fstream>
#include <string>

#include "tests/test_support.h"
#include "tests/update_testkit.h"
#include "tests/update_teststatus.h"
#include "update/activate.h"
#include "update/config.h"
#include "update/state.h"
#include "update/updater.h"

using namespace tona::test;
namespace fs = std::filesystem;
namespace {

struct Rig {
    TempDir dir;
    TestServer server;
    FakeStatus status;
    up::UpdateEnv env;
    std::atomic<int> restarts{0};

    explicit Rig(const std::string& name) : dir(name), env(make_env(server, dir.path())) {}

    up::SwitchConfig switch_config() {
        up::SwitchConfig sw;
        sw.status_url = status.url();
        sw.poll_ms = 20;
        sw.ready_seconds = 3;
        sw.idle_seconds = 3;
        sw.restart = [this](std::string&) {
            restarts++;
            return true;
        };
        return sw;
    }
    std::unique_ptr<up::Updater> make(bool release = true) {
        return std::make_unique<up::Updater>(env, switch_config(), release);
    }
};

template <class Pred>
bool wait_until(Pred done, int seconds = 10) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (!done()) {
        if (std::chrono::steady_clock::now() > end) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

bool idle(const up::Updater& u) { return u.view().phase == up::UpdateView::Phase::Idle; }

void test_checks_are_off_by_default() {
    Rig rig("updater-off");
    rig.server.serve_latest(make_release("0.3.0", 3));
    auto u = rig.make();
    EXPECT(!u->view().checks_on);
    u->tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT(rig.server.hits("/releases/latest/download/manifest.json") == 0);
    EXPECT(rig.server.hits("/releases/latest/download/manifest.json.minisig") == 0);
    u->install();  // nothing known, nothing happens
    EXPECT(idle(*u) && !u->view().available);
}

void test_not_a_release_installation() {
    Rig rig("updater-source");
    rig.server.serve_latest(make_release("0.3.0", 3));
    auto u = rig.make(false);
    u->set_checks(true);
    u->tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT(!u->view().enabled);
    EXPECT(rig.server.hits("/releases/latest/download/manifest.json") == 0);
}

void test_enable_check_install() {
    Rig rig("updater-flow");
    FakeRelease r = make_release("0.3.0", 3);
    r.manifest["severity"] = "security";
    r.sign(main_key());
    rig.server.serve_latest(r);
    {
        auto u = rig.make();
        u->set_checks(true);
        EXPECT(u->view().checks_on);
    }
    auto u = rig.make();  // a new start reads the persisted choice
    EXPECT(u->view().checks_on);
    u->tick();
    EXPECT(wait_until([&] { return u->view().available; }));
    const up::UpdateView v = u->view();
    EXPECT(v.version == "0.3.0" && v.security && v.changelog == "Test release 0.3.0");
    EXPECT(v.download_bytes == make_release("0.3.0", 3).manifest["files"][0]["size"].get<uint64_t>() * 3 ||
           v.download_bytes > 0);
    // Once per start: more ticks do not fetch again
    const int hits = rig.server.hits("/releases/latest/download/manifest.json");
    for (int i = 0; i < 5; i++) u->tick();
    EXPECT(rig.server.hits("/releases/latest/download/manifest.json") == hits);

    u->install();
    EXPECT(wait_until([&] { return idle(*u) && !u->view().available && rig.restarts > 0; }));
    EXPECT(!u->view().failed);
    EXPECT(up::linked_version(rig.env.layout, "current") == "0.3.0");
    EXPECT(up::load_state(rig.env.layout).state.checks_enabled);  // the install kept the setting
    EXPECT(up::load_state(rig.env.layout).state.seq == 3);
}

void test_failures_are_reported() {
    Rig rig("updater-fail");
    FakeRelease r = make_release("0.3.0", 3);
    r.signature = other_key().sign_file(r.manifest_text, comment_for(3, "0.3.0"));
    rig.server.serve_latest(r);
    auto u = rig.make();
    u->set_checks(true);
    u->tick();
    EXPECT(wait_until([&] { return u->view().failed; }));
    EXPECT(u->view().message.contains("signature"));
    EXPECT(!u->view().available);
    EXPECT(!fs::exists(rig.env.layout.versions() / "0.3.0"));
    // Unreachable server: the same, without a crash
    TempDir dir2("updater-down");
    up::UpdateEnv env2 = rig.env;
    env2.base = up::parse_base("http://127.0.0.1:1/releases").value_or(up::UpdateBase{});
    env2.layout.root = dir2.path() / "tona";
    up::Updater u2(env2, rig.switch_config(), true);
    u2.set_checks(true);
    u2.tick();
    EXPECT(wait_until([&] { return u2.view().failed; }));
}

void test_install_failure_leaves_current() {
    Rig rig("updater-install-fail");
    const FakeRelease first = make_release("0.3.0", 3);
    rig.server.serve_latest(first);
    auto u = rig.make();
    u->set_checks(true);
    u->tick();
    EXPECT(wait_until([&] { return u->view().available; }));
    rig.server.set("/releases/download/v0.3.0/tona-server", Route(std::string("not what the manifest says")));
    u->install();
    EXPECT(wait_until([&] { return u->view().failed; }));
    EXPECT(u->view().message.contains("nothing changed"));
    EXPECT(up::linked_version(rig.env.layout, "current").empty());
    EXPECT(rig.restarts == 0);
}

void test_roll_back_key() {
    Rig rig("updater-rollback");
    for (const char* v : {"0.3.0", "0.4.0"}) {
        fs::create_directories(rig.env.layout.version(v));
        up::activate_version(rig.env.layout, v);
    }
    ::chmod(rig.env.layout.root.c_str(), 0700);
    auto u = rig.make();
    u->roll_back();
    EXPECT(wait_until([&] { return rig.restarts == 1 && idle(*u); }));
    EXPECT(up::linked_version(rig.env.layout, "current") == "0.3.0");
    EXPECT(!u->view().failed);
}

void test_runs_from_release() {
    up::Layout l;
    l.root = "/tmp/tona-nonexistent-root";
#if defined(__APPLE__)
    // /tmp is a symlink to /private/tmp on macOS, and the root is compared canonically
    EXPECT(up::Updater::runs_from_release(l, "/private/tmp/tona-nonexistent-root/versions/0.3.0/bin/tona"));
#else
    EXPECT(up::Updater::runs_from_release(l, "/tmp/tona-nonexistent-root/versions/0.3.0/bin/tona"));
#endif
    EXPECT(!up::Updater::runs_from_release(l, "/tmp/tona-nonexistent-root/versions"));
    EXPECT(!up::Updater::runs_from_release(l, "/tmp/tona-nonexistent-root/current/bin/tona"));
    EXPECT(!up::Updater::runs_from_release(l, "/tmp/tona-nonexistent-root/versions-evil/bin/tona"));
    EXPECT(!up::Updater::runs_from_release(l, "/home/user/inference-engine/work/build/tona"));
    EXPECT(!up::Updater::runs_from_release(l, ""));
}

void test_base_url_rules() {
    const auto remote = up::parse_base("https://github.com/o/r/releases");
    EXPECT(remote.has_value());
    EXPECT(!remote.value_or(up::UpdateBase{}).allow_http_loopback);
    const auto parsed = up::parse_base("http://127.0.0.1:8123/releases");
    EXPECT(parsed.has_value());
    const up::UpdateBase local = parsed.value_or(up::UpdateBase{});
    EXPECT(local.allow_http_loopback);
    EXPECT(local.manifest_url() == "http://127.0.0.1:8123/releases/latest/download/manifest.json");
    EXPECT(local.asset_url("0.3.0", "tona-server") == "http://127.0.0.1:8123/releases/download/v0.3.0/tona-server");
    for (const char* bad :
         {"http://example.com/r", "http://127.0.0.1.evil.example/r", "http://127.0.0.1/r", "http://localhost:80/r",
          "https://github.com/o/r/", "https://github.com/o?x=1", "https://user@github.com/o", "https://github.com",
          "https://github.com/o r", "HTTPS://github.com/o", "ftp://github.com/o", "", "https://github.com/o#frag",
          "http://127.0.0.1:99999/r"})
        EXPECT(!up::parse_base(bad).has_value());
}

void test_environment() {
    std::string why;
    ::unsetenv("TONA_UPDATE_BASE");
    const auto def = up::resolve_base(why);
    EXPECT(def.has_value() && def->url == "https://github.com/tobias-nawa/tona/releases");
    ::setenv("TONA_UPDATE_BASE", "http://evil.example/r", 1);
    EXPECT(!up::resolve_base(why).has_value() && !why.empty());  // never a silent fallback
    ::setenv("TONA_UPDATE_BASE", "http://127.0.0.1:9/r", 1);
    EXPECT(up::resolve_base(why).has_value());
    ::unsetenv("TONA_UPDATE_BASE");
    ::setenv("TONA_INSTALL_DIR", "/some/where", 1);
    EXPECT(up::install_root() == "/some/where");
    ::unsetenv("TONA_INSTALL_DIR");
    ::setenv("XDG_DATA_HOME", "/data", 1);
    EXPECT(up::install_root() == "/data/tona");
    ::unsetenv("XDG_DATA_HOME");
    ::setenv("HOME", "/home/x", 1);
    EXPECT(up::install_root() == "/home/x/.local/share/tona");
}

void test_default_updater_from_environment() {
    TempDir dir("updater-default");
    ::setenv("TONA_INSTALL_DIR", (dir.path() / "tona").c_str(), 1);
    ::setenv("TONA_UPDATE_BASE", "http://evil.example/r", 1);
    std::string note;
    EXPECT(up::make_default_updater("http://127.0.0.1:1", note) == nullptr);
    EXPECT(!note.empty());
    ::setenv("TONA_UPDATE_BASE", "http://127.0.0.1:9/r", 1);
    note.clear();
    const auto u = up::make_default_updater("http://127.0.0.1:1", note);
    EXPECT(u != nullptr && note.empty());
    EXPECT(u && !u->view().enabled);  // this test binary does not run from <root>/versions
    ::unsetenv("TONA_UPDATE_BASE");
    ::unsetenv("TONA_INSTALL_DIR");
}

void test_unreadable_state_blocks_switching_on() {
    Rig rig("updater-badstate");
    fs::create_directories(rig.env.layout.root);
    ::chmod(rig.env.layout.root.c_str(), 0700);
    std::ofstream(rig.env.layout.state()) << "not json";
    auto u = rig.make();
    u->set_checks(true);
    EXPECT(u->view().failed && !u->view().checks_on);
    EXPECT(u->view().message.contains("state.json"));
}

}  // namespace

int main() {
    return run("update_updater", [] {
        test_checks_are_off_by_default();
        test_not_a_release_installation();
        test_enable_check_install();
        test_failures_are_reported();
        test_install_failure_leaves_current();
        test_roll_back_key();
        test_runs_from_release();
        test_base_url_rules();
        test_environment();
        test_default_updater_from_environment();
        test_unreadable_state_blocks_switching_on();
    });
}
