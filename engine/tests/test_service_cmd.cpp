// tona start | stop | restart | autostart with a fake systemctl, launchctl and just on PATH
// that write down their arguments: what a release installation and a checkout run, what is shown, and what fails.
#include <cstdlib>

#include "client/service_cmd.h"
#include "core/autostart.h"
#include "core/unit.h"
#include "tests/client_testkit.h"

using namespace tona::client;
using namespace tona::test;
namespace fs = std::filesystem;

namespace {

struct Rig {
    TempDir dir;
    fs::path bin, log;
    std::string old_path;
    std::ostringstream out, err;

    explicit Rig(const std::string& name) : dir(name) {
        bin = dir.path() / "fakebin", log = dir.path() / "calls.log";
        fs::create_directories(bin);
        fake("systemctl", 0), fake("just", 0), fake("launchctl", 0);
        old_path = std::getenv("PATH");
        ::setenv("PATH", (bin.string() + ":" + old_path).c_str(), 1);
        ::unsetenv("TONA_START_CMD");
        ::setenv("HOME", dir.path().c_str(), 1);
    }
    ~Rig() { ::setenv("PATH", old_path.c_str(), 1); }
    Rig(const Rig&) = delete;
    Rig& operator=(const Rig&) = delete;

    void fake(const std::string& name, int exit_code) const {
        const fs::path script = bin / name;
        std::ofstream(script) << "#!/bin/sh\necho \"" << name << " $*\" >> " << log << "\necho \"output of " << name
                              << "\"\nexit " << exit_code << "\n";
        fs::permissions(script, fs::perms::owner_all);
    }
    std::string calls() const { return read_text(log.string()); }
    int run(const Env& env, const std::string& action) { return run_service_command(env, action, out); }
};

// The Linux release (systemd unit), also on a Mac host, where for_release says macOS
Env linux_release(const Rig& rig) {
    Env env = Env::for_release((rig.dir.path() / "root").string());
    env.macos = false;
    return env;
}

void test_release_starts_and_stops_the_unit() {
    Rig rig("service-release");
    const Env env = linux_release(rig);
    EXPECT(rig.run(env, "start") == 0);
    EXPECT(has(rig.calls(), "systemctl --user start tona-server"));
    EXPECT(has(rig.out.str(), "output of systemctl"));  // what the command prints is relayed
    EXPECT(has(rig.out.str(), "started on port 8662 (progress: tona, log: journalctl --user -u tona-server -f)"));
    EXPECT(rig.run(env, "stop") == 0);
    EXPECT(has(rig.calls(), "systemctl --user stop tona-server") && has(rig.out.str(), "stopped\n"));
}

void test_release_failures_are_not_reported_as_success() {
    Rig rig("service-release-fail");
    rig.fake("systemctl", 1);
    const Env env = linux_release(rig);
    EXPECT(rig.run(env, "start") == 1 && !has(rig.out.str(), "started on port"));
    EXPECT(rig.run(env, "stop") == 1 && !has(rig.out.str(), "stopped\n"));
}

// A unit of an earlier tona holds the settings itself: the next start writes the one-line form, so that the settings of
// tona.conf (a model switch) take effect
void test_old_unit_is_replaced() {
    Rig rig("service-old-unit");
    const Env env = linux_release(rig);
    const fs::path unit = rig.dir.path() / ".config/systemd/user/tona-server.service";
    fs::create_directories(unit.parent_path());
    std::ofstream(unit) << "[Service]\nExecStart=/x/current/bin/tona-server /x/models/m.gguf --port 8662\n";
    EXPECT(rig.run(env, "start") == 0);
    EXPECT(read_text(unit.string()) == tona::install::render_unit(env.root));
    EXPECT(has(rig.calls(), "systemctl --user daemon-reload") &&
           has(rig.calls(), "systemctl --user start tona-server"));
    rig.fake("systemctl", 0);
    fs::remove(rig.log);  // the calls of the first start are not the question now
    EXPECT(rig.run(env, "restart") == 0 && !has(rig.calls(), "daemon-reload"));  // already in the new form
}

void test_release_restarts_through_systemd() {
    Rig rig("service-release-restart");
    const Env env = linux_release(rig);
    EXPECT(rig.run(env, "restart") == 0 && has(rig.calls(), "systemctl --user restart tona-server"));
    EXPECT(has(rig.calls(), "reset-failed tona-server\nsystemctl --user restart tona-server"));  // the limit first
    rig.fake("systemctl", 1);
    EXPECT(rig.run(env, "restart") == 1);
}

// macOS: the server is `tona start --foreground` run in the background; the pid file ties stop to that process. The
// fake tona is a script that writes its pid and becomes the fake tona-server (a link to sleep, so that `ps` shows the
// name the stop command insists on), as the real one does.
void test_macos_runs_the_foreground_start() {
    Rig rig("service-macos");
    const fs::path root = rig.dir.path() / "root";
    fs::create_directories(root);
    // A link, not a copy: macOS kills a copy of a system program outside /bin at once (SIGKILL), so a copied sleep
    // was never running and the second start saw no server. ps names the process after the link
    fs::create_symlink("/bin/sleep", rig.bin / "tona-server");
    Env env = Env::for_release(root.string());
    env.macos = true;
    env.state_dir = (rig.dir.path() / "state").string();
    fs::create_directories(root / "current/bin");
    std::ofstream(root / "current/bin/tona")
        << "#!/bin/sh\n[ \"$1 $2\" = \"start --foreground\" ] || exit 2\necho \"started $TONA_INSTALL_DIR\" >> "
        << rig.log << "\necho $$ > " << env.server_pid() << "\nexec tona-server 30\n";
    fs::permissions(root / "current/bin/tona", fs::perms::owner_all);
    EXPECT(rig.run(env, "start") == 0);
    EXPECT(has(rig.out.str(), "started on port 8662 (progress: tona, log: " + env.server_log() + ")"));
    EXPECT(!has(rig.calls(), "systemctl"));
    for (int i = 0; i < 100 && !fs::exists(env.server_pid()); ++i) ::usleep(50000);  // it runs in the background
    EXPECT(fs::exists(env.server_pid()));  // written by the started program itself, as the real one does
    EXPECT(has(rig.calls(), "started " + root.string()));
    rig.out.str("");
    EXPECT(rig.run(env, "start") == 0 && has(rig.out.str(), "already running"));  // no second server
    EXPECT(!has(rig.out.str(), "started on port"));                               // and no claim of one
    EXPECT(rig.run(env, "stop") == 0 && has(rig.out.str(), "stopped\n"));
    EXPECT(!fs::exists(env.server_pid()));
    rig.out.str("");
    EXPECT(rig.run(env, "stop") == 0 && has(rig.out.str(), "not running") && !has(rig.out.str(), "stopped"));
    fs::create_directories(env.state_dir);
    std::ofstream(env.server_pid()) << ::getpid() << "\n";  // a stale file naming some other program
    EXPECT(rig.run(env, "stop") == 0 && !fs::exists(env.server_pid()));
    EXPECT(rig.run(env, "restart") == 0);  // nothing runs: the restart is a start
    for (int i = 0; i < 100 && !fs::exists(env.server_pid()); ++i) ::usleep(50000);
    EXPECT(rig.run(env, "restart") == 0);  // a running one is replaced
    for (int i = 0; i < 100 && !fs::exists(env.server_pid()); ++i) ::usleep(50000);
    EXPECT(fs::exists(env.server_pid()));
    EXPECT(rig.run(env, "stop") == 0);
}

// A program that never writes the pid file: `tona start` says so after 3 s instead of claiming a start
void test_macos_start_without_a_pid_file_says_so() {
    Rig rig("service-macos-nopid");
    const fs::path root = rig.dir.path() / "root";
    fs::create_directories(root / "current/bin");
    std::ofstream(root / "current/bin/tona") << "#!/bin/sh\nexit 0\n";
    fs::permissions(root / "current/bin/tona", fs::perms::owner_all);
    Env env = Env::for_release(root.string());
    env.macos = true;
    env.state_dir = (rig.dir.path() / "state").string();
    EXPECT(rig.run(env, "start") == 1 && has(rig.out.str(), "did not start") && !has(rig.out.str(), "started on"));
}

int autostart(Rig& rig, const Env& env, const std::string& arg) {
    rig.out.str(""), rig.err.str("");
    return run_autostart_command(env, arg, rig.out, rig.err);
}

void test_autostart_on_linux_enables_the_unit() {
    Rig rig("autostart-linux");
    const Env env = linux_release(rig);
    EXPECT(autostart(rig, env, "on") == 0 && has(rig.calls(), "systemctl --user enable tona-server\n"));
    EXPECT(has(rig.out.str(), "next login"));  // enable without --now: nothing starts before the next login
    EXPECT(autostart(rig, env, "status") == 0 && has(rig.out.str(), "on: "));
    EXPECT(autostart(rig, env, "off") == 0 && has(rig.calls(), "systemctl --user disable tona-server"));
    rig.fake("systemctl", 1);  // is-enabled says no; a failing enable is an error
    EXPECT(autostart(rig, env, "status") == 0 && has(rig.out.str(), "off: "));
    EXPECT(autostart(rig, env, "on") == 1 && has(rig.err.str(), "enable tona-server failed"));
    EXPECT(autostart(rig, env, "maybe") == 2 && has(rig.err.str(), "on|off|status"));
    EXPECT(autostart(rig, Env::for_checkout(rig.dir.path().string()), "on") == 2);
    EXPECT(!has(rig.calls(), "launchctl"));
}

// macOS: a LaunchAgent that runs `tona start` at login; launchctl is the fake on PATH, HOME is the rig's directory
void test_autostart_on_macos_writes_the_launch_agent() {
    Rig rig("autostart-macos");
    Env env = Env::for_release((rig.dir.path() / "a&b").string());
    env.macos = true;
    const fs::path plist = rig.dir.path() / "Library/LaunchAgents/com.tobiasnawa.tona.plist";
    EXPECT(autostart(rig, env, "off") == 0 && rig.calls().empty());  // nothing written, nothing unloaded
    EXPECT(autostart(rig, env, "on") == 0 && fs::exists(plist));
    EXPECT(!has(rig.calls(), "bootstrap"));  // as on Linux: the next login starts it
    const std::string text = read_text(plist.string());
    EXPECT(has(text, "<string>" + rig.dir.path().string() +
                         "/a&amp;b/current/bin/tona</string><string>start</string><string>--foreground</string>"));
    EXPECT(has(text, "<key>RunAtLoad</key><true/>") && !has(text, "AbandonProcessGroup"));
    EXPECT(has(text, "a&amp;b/autostart.log"));
#if defined(__APPLE__)
    EXPECT(std::system(("plutil -lint -s '" + plist.string() + "'").c_str()) == 0);
#endif
    EXPECT(autostart(rig, env, "status") == 0 && has(rig.out.str(), "on: "));
    EXPECT(autostart(rig, env, "off") == 0 && !fs::exists(plist));
    EXPECT(has(rig.calls(), "launchctl bootout gui/") && has(rig.calls(), "/com.tobiasnawa.tona"));
    EXPECT(autostart(rig, env, "status") == 0 && has(rig.out.str(), "off: "));
    // tona setup with "At login": written and loaded at once
    EXPECT(tona::install::set_autostart(true, env.root, true, true, [](const std::string&) {}).empty());
    EXPECT(has(rig.calls(), "launchctl bootstrap gui/") && has(rig.calls(), plist.string()));
    rig.fake("launchctl", 5);
    EXPECT(!tona::install::set_autostart(true, env.root, true, true, [](const std::string&) {}).empty());
    EXPECT(!has(rig.calls(), "systemctl"));
}

void test_checkout_goes_through_just() {
    Rig rig("service-checkout");
    fs::create_directories(rig.dir.path() / "repo/work");
    Env env = Env::for_checkout((rig.dir.path() / "repo").string());
    env.port = 8123;
    const std::string justfile = (rig.dir.path() / "repo/justfile").string();
    EXPECT(rig.run(env, "start") == 0 && has(rig.calls(), "just -f " + justfile + " port=8123 start"));
    EXPECT(rig.run(env, "stop") == 0 && has(rig.calls(), "port=8123 stop"));
    EXPECT(rig.run(env, "restart") == 0 && has(rig.calls(), "port=8123 stop") && has(rig.calls(), "port=8123 start"));
    EXPECT(has(rig.out.str(), "output of just"));
    rig.fake("just", 3);
    EXPECT(rig.run(env, "stop") == 1);

    ::setenv("TONA_START_CMD", "echo custom start", 1);  // the user's own start command wins
    EXPECT(rig.run(env, "start") == 0 && has(rig.out.str(), "custom start"));
    ::unsetenv("TONA_START_CMD");
}

void test_layout_and_names() {
    TempDir dir("service-layout");
    const Env lone = Env::for_release((dir.path() / "nowhere").string());
    EXPECT(has(layout_error(lone), "neither a checkout nor a release installation"));
    fs::create_directories(dir.path() / "installed/versions/0.0.1");
    EXPECT(has(layout_error(Env::for_release((dir.path() / "installed").string())), "no version is installed"));
    fs::create_directory_symlink("versions/0.0.1", dir.path() / "installed/current");
    EXPECT(layout_error(Env::for_release((dir.path() / "installed").string())).empty());
    EXPECT(layout_error(Env::for_checkout(dir.path().string())).empty());  // a checkout needs no versions/
    for (const char* w : {"start", "stop", "restart"}) EXPECT(is_service_command(w));
    EXPECT(!is_service_command("claude") && !is_service_command("") && !is_service_command("install-service"));
}

}  // namespace

int main() {
    return run("service_cmd", [] {
        test_release_starts_and_stops_the_unit();
        test_release_failures_are_not_reported_as_success();
        test_old_unit_is_replaced();
        test_release_restarts_through_systemd();
        test_macos_runs_the_foreground_start();
        test_macos_start_without_a_pid_file_says_so();
        test_autostart_on_linux_enables_the_unit();
        test_autostart_on_macos_writes_the_launch_agent();
        test_checkout_goes_through_just();
        test_layout_and_names();
    });
}
