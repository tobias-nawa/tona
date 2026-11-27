#include "update/script_service.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <thread>

#include "update/config.h"
#include "update/switch.h"

#if defined(__APPLE__)
#include <libproc.h>
#include <sys/proc.h>
#endif

namespace fs = std::filesystem;

namespace tona::update {

namespace {

using Clock = std::chrono::steady_clock;

int read_pid(const std::string& file) {
    std::ifstream in(file);
    long pid = 0;
    in >> pid;
    return in && pid > 1 && pid < (1L << 30) ? static_cast<int>(pid) : 0;
}

bool wait_gone(int pid, int seconds) {
    const auto end = Clock::now() + std::chrono::seconds(seconds);
    while (process_running(pid)) {
        if (Clock::now() >= end) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return true;
}

// The pid file's server, if it is one; a missing or stale pid file is not an error
bool stop_server(const ScriptService& s, std::string& why) {
    const int pid = read_pid(s.pid_file);
    if (const std::string name = process_name(pid); pid > 0 && (name == "tona-server" || name == "tona")) {
        ::kill(pid, SIGTERM);
        if (!wait_gone(pid, s.stop_seconds)) {
            ::kill(pid, SIGKILL);
            if (!wait_gone(pid, 5)) {
                why = "the old server (pid " + std::to_string(pid) + ") does not exit";
                return false;
            }
        }
    }
    std::error_code ec;
    fs::remove(s.pid_file, ec);
    return true;
}

// A server that answers after ours is gone was not started by `tona start`: starting another one would fail on the
// port, and the readiness check would see the old server and call the update a success
bool address_free(const ScriptService& s) {
    const auto end = Clock::now() + std::chrono::seconds(5);
    while (server_answers(s.status_url)) {
        if (Clock::now() >= end) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return true;
}

}  // namespace

// A zombie still answers kill(pid, 0). The server is orphaned on purpose (nohup), and when PID 1 does not reap (a
// container whose PID 1 is `tail -f /dev/null`, as on GitHub's container jobs) an ended server stays a zombie
bool process_running(int pid) {
    if (pid <= 0 || (::kill(pid, 0) != 0 && errno != EPERM)) return false;
#if defined(__APPLE__)
    proc_bsdinfo info{};
    if (::proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, sizeof info) == static_cast<int>(sizeof info))
        return info.pbi_status != SZOMB;
    return true;
#else
    std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
    std::string stat;
    std::getline(in, stat);
    const auto close = stat.rfind(')');
    return close == std::string::npos || close + 2 >= stat.size() || stat[close + 2] != 'Z';
#endif
}

std::string process_name(int pid) {
#if defined(__APPLE__)
    char name[2 * MAXCOMLEN + 1] = {};
    return ::proc_name(pid, name, sizeof name) > 0 ? std::string(name) : std::string();
#else
    std::ifstream in("/proc/" + std::to_string(pid) + "/comm");
    std::string name;
    std::getline(in, name);
    return name;
#endif
}

ScriptService release_script_service(const Layout& layout, const std::string& status_url) {
    const std::string state = state_dir();
    return {(layout.current() / "bin/tona").string(), layout.root.string(), state + "/server.pid",
            state + "/server.log", status_url};
}

bool script_service_alive(const ScriptService& s) { return process_running(read_pid(s.pid_file)); }

bool restart_script_service(const ScriptService& s, std::string& why) {
    if (!stop_server(s, why)) return false;
    if (!address_free(s)) {
        why = "a server that tona start did not start answers at " + s.status_url + "; stop it first";
        return false;
    }
    std::error_code ec;
    fs::create_directories(fs::path(s.pid_file).parent_path(), ec);
    // As `tona start`: nohup in the background, so that the server outlives tona and is not its child (no zombie when
    // it exits); the shell records the pid, which the server keeps through its exec
    const std::string cmd =
        R"(TONA_INSTALL_DIR="$3" nohup "$0" start --foreground >>"$1" 2>&1 </dev/null & echo $! >"$2")";
    return run_command({"/bin/sh", "-c", cmd, s.program, s.log_file, s.pid_file, s.install_dir}, 10, why);
}

}  // namespace tona::update
