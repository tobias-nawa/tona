// The updater's restart without systemd (macOS): stop the server named by the pid file, start
// `<root>/current/bin/tona start --foreground` in the background, and the whole update against a local release server:
// 0.0.1, then 0.0.2, then a 0.0.3 whose server does not start and is rolled back, with the user's way back in between.
// The server is fake_tona_server (argv[1]), installed as bin/tona-server of each release; it listens only on 127.0.0.1.
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "tests/test_support.h"
#include "tests/update_testkit.h"
#include "tests/update_teststatus.h"
#include "update/activate.h"
#include "update/script_service.h"
#include "update/state.h"
#include "update/updater.h"

using namespace tona::test;
namespace fs = std::filesystem;
namespace {

std::string g_fake_server;

// A port nothing listens on: bind to port 0, read the port, close. (httplib's bind_to_any_port listens, and its
// stop() without a run leaves that socket open: the kernel then accepts connections nobody answers.)
int free_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof a;
    const bool ok = ::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0 &&
                    ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) == 0;
    ::close(fd);
    return ok ? ntohs(a.sin_port) : 0;
}

int read_pid(const fs::path& file) {
    std::ifstream in(file);
    int pid = 0;
    in >> pid;
    return pid;
}

bool alive(int pid) { return up::process_running(pid); }

template <class Pred>
bool wait_until(Pred done, int seconds = 20) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (!done()) {
        if (std::chrono::steady_clock::now() > end) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return true;
}

// What `tona start --foreground` does once the settings are read: become tona-server of the same version
std::string tona_stand_in(const std::string& port) {
    return "#!/bin/sh\n[ \"$1 $2\" = \"start --foreground\" ] || exit 2\nexec \"$(dirname \"$0\")/tona-server\" "
           "model.gguf --port " +
           port + "\n";
}

// An install root whose `tona start --foreground` is a stand-in that execs current/bin/tona-server (as the real one
// does after reading the settings), and a state directory; whatever server is still running at the end is ended
struct Service {
    TempDir dir;
    up::Layout layout;
    up::ScriptService svc;
    std::string port;

    explicit Service(const std::string& name) : dir(name) {
        layout.root = dir.path() / "tona";
        fs::create_directories(layout.versions());
        ::chmod(layout.root.c_str(), 0700);
        const std::string url = "http://127.0.0.1:" + std::to_string(free_port());
        svc = {(layout.current() / "bin/tona").string(),
               layout.root.string(),
               (dir.path() / "state/server.pid").string(),
               (dir.path() / "state/server.log").string(),
               url,
               5};
        port = url.substr(url.rfind(':') + 1);
    }
    ~Service() {
        if (const int pid = read_pid(svc.pid_file); alive(pid) && up::process_name(pid) == "tona-server")
            ::kill(pid, SIGKILL);
    }
    // A version whose server is the fake (installed by hand, as the updater would)
    void add_fake_version(const std::string& v) const {
        fs::create_directories(layout.version(v) / "bin");
        fs::copy_file(g_fake_server, layout.version(v) / "bin/tona-server");
        const fs::path program = layout.version(v) / "bin/tona";
        std::ofstream(program) << tona_stand_in(port);
        fs::permissions(program, fs::perms::owner_all);
    }
    bool ready() const { return up::server_is_ready(svc.status_url); }
};

void test_restart_starts_and_replaces_the_server() {
    Service s("script-restart");
    s.add_fake_version("0.0.1");
    up::activate_version(s.layout, "0.0.1");
    std::string why;
    EXPECT(up::restart_script_service(s.svc, why));  // nothing ran: a plain start
    EXPECT(wait_until([&] { return s.ready(); }));
    const int first = read_pid(s.svc.pid_file);
    EXPECT(alive(first) && up::process_name(first) == "tona-server");
    EXPECT(up::restart_script_service(s.svc, why));
    EXPECT(wait_until([&] { return s.ready(); }));
    const int second = read_pid(s.svc.pid_file);
    EXPECT(second != first && alive(second) && !alive(first));
}

void test_stale_pid_file_is_not_killed() {
    Service s("script-stale");
    s.add_fake_version("0.0.1");
    up::activate_version(s.layout, "0.0.1");
    fs::create_directories(fs::path(s.svc.pid_file).parent_path());
    // The pid of this test: alive, but not a tona-server (as a pid file from before a reboot)
    std::ofstream(s.svc.pid_file) << ::getpid();
    std::string why;
    EXPECT(up::restart_script_service(s.svc, why));
    EXPECT(wait_until([&] { return s.ready(); }));
    EXPECT(read_pid(s.svc.pid_file) != ::getpid());
}

void test_foreign_server_blocks_the_restart() {
    FakeStatus foreign;  // a server that tona start did not start, on the address of the service
    Service s("script-foreign");
    s.svc.status_url = foreign.url();
    std::string why;
    EXPECT(!up::restart_script_service(s.svc, why));
    EXPECT(why.contains("did not start"));
    EXPECT(!fs::exists(s.svc.pid_file));  // nothing was started
}

// The release a server of these bytes is installed from; its `tona` is the stand-in that starts that server
FakeRelease release_with_server(const std::string& version, uint64_t seq, const std::string& server,
                                const std::string& port) {
    FakeRelease r = make_release_without_runtime(version, seq);
    const std::string tona = tona_stand_in(port);
    for (json& f : r.manifest["files"]) {
        const std::string path = f["path"];
        if ((path != "bin/tona-server" && path != "bin/tona") || f["platform"] != std::string(up::kPlatform)) continue;
        const std::string& bytes = path == "bin/tona" ? tona : server;
        up::Sha256 h;
        h.update(bytes.data(), bytes.size());
        const auto d = h.finish();
        f["size"] = bytes.size();
        f["sha256"] = up::to_hex(d.data(), d.size());
        r.assets[f["asset"].get<std::string>()] = bytes;
    }
    r.sign(main_key());
    return r;
}

// One check and install of a fresh updater (a new start of tona); returns its last view
up::UpdateView update_once(const up::UpdateEnv& env, const up::SwitchConfig& sw) {
    up::Updater u(env, sw, true);
    u.set_checks(true);
    u.tick();
    if (!wait_until([&] { return u.view().available; })) return u.view();
    u.install();
    wait_until([&] { return u.view().phase == up::UpdateView::Phase::Idle && !u.view().available; }, 60);
    return u.view();
}

void test_update_and_roll_back_end_to_end() {
    Service s("script-e2e");
    TestServer releases;
    up::UpdateEnv env = make_env(releases, s.dir.path());
    env.layout = s.layout;
    up::SwitchConfig sw;
    sw.status_url = s.svc.status_url;
    sw.ready_seconds = 5;  // the fake server answers within 0.1 s; the broken release waits this out
    sw.idle_seconds = 3;
    sw.poll_ms = 50;
    sw.restart = [&s](std::string& why) { return up::restart_script_service(s.svc, why); };
    sw.alive = [&s] { return up::script_service_alive(s.svc); };
    const std::string server = read_file(g_fake_server);

    releases.serve_latest(release_with_server("0.0.1", 1, server, s.port));
    up::UpdateView v = update_once(env, sw);
    EXPECT(!v.failed && up::linked_version(s.layout, "current") == "0.0.1" && s.ready());
    const int first = read_pid(s.svc.pid_file);

    releases.serve_latest(release_with_server("0.0.2", 2, server, s.port));
    v = update_once(env, sw);
    EXPECT(!v.failed && v.message == "running the new version");
    EXPECT(up::linked_version(s.layout, "current") == "0.0.2" && up::linked_version(s.layout, "previous") == "0.0.1");
    const int second = read_pid(s.svc.pid_file);
    EXPECT(s.ready() && second != first && !alive(first));

    // The user's way back (R twice on the Start page): 0.0.1 runs again
    {
        up::Updater u(env, sw, true);
        u.roll_back();
        wait_until([&] { return u.view().phase == up::UpdateView::Phase::Idle && !u.view().message.empty(); }, 30);
        EXPECT(!u.view().failed && up::linked_version(s.layout, "current") == "0.0.1" && s.ready());
    }

    // A server that exits at once: the switch goes back to the version that ran before and starts it again
    releases.serve_latest(release_with_server("0.0.3", 3, "#!/bin/sh\nexit 1\n", s.port));
    v = update_once(env, sw);
    EXPECT(v.failed && v.message.contains("the new server exited") &&
           v.message.contains("the previous one runs again"));
    EXPECT(up::linked_version(s.layout, "current") == "0.0.1" && s.ready());
    // The rolled-back release is not offered again: the sequence (3) stays in state.json while 0.0.1 runs
    EXPECT(up::check_for_update(env).kind == up::CheckOutcome::Kind::UpToDate);
    EXPECT(up::load_state(s.layout).state.seq == 3);
}

// posix_spawn, not system(): no shell parses the paths; stdout goes to `out`
int run_to_file(const std::string& exe, const std::string& env_entry, const std::string& out) {
    std::string a0 = exe, a1 = "--report-enabled", e0 = env_entry;
    char* argv[] = {a0.data(), a1.data(), nullptr};
    char* envp[] = {e0.data(), nullptr};
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 1, out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    pid_t pid = 0;
    int status = -1;
    if (posix_spawn(&pid, exe.c_str(), &actions, nullptr, argv, envp) == 0) waitpid(pid, &status, 0);
    posix_spawn_file_actions_destroy(&actions);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// The detection of tona: this test, copied into <root>/versions/0.0.1/bin and started through `current`, must count
// as a release installation (the path of the running program with its links resolved, on macOS without /proc)
void test_a_copy_in_versions_counts_as_release(const std::string& self) {
    Service s("script-self");
    fs::create_directories(s.layout.version("0.0.1") / "bin");
    const fs::path copy = s.layout.version("0.0.1") / "bin/tona";
    fs::copy_file(self, copy);
    up::activate_version(s.layout, "0.0.1");
    const std::string out = (s.dir.path() / "enabled.txt").string();
    const std::string root = "TONA_INSTALL_DIR=" + s.layout.root.string();
    EXPECT(run_to_file((s.layout.current() / "bin/tona").string(), root, out) == 0);
    EXPECT(read_file(out) == "enabled\n");
    EXPECT(run_to_file(self, root, out) == 0);
    EXPECT(read_file(out) == "not a release\n");  // the build directory is not
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--report-enabled") {
        std::string note;
        const auto u = up::make_default_updater("http://127.0.0.1:1", note);
        std::printf("%s\n", u && u->view().enabled ? "enabled" : "not a release");
        return 0;
    }
    return run("update_script_service", [argc, argv] {
        g_fake_server = argc > 1 ? argv[1] : "";
        EXPECT(!g_fake_server.empty() && fs::exists(g_fake_server));
        test_restart_starts_and_replaces_the_server();
        test_stale_pid_file_is_not_killed();
        test_foreign_server_blocks_the_restart();
        test_update_and_roll_back_end_to_end();
        test_a_copy_in_versions_counts_as_release(argv[0]);
    });
}
