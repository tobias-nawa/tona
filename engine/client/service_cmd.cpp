#include "client/service_cmd.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <ostream>
#include <thread>

#include "core/autostart.h"
#include "core/sys.h"

namespace fs = std::filesystem;

namespace tona::client {

namespace {

// Runs a shell command and relays its output; the exit status of the command is the result
int relay(const std::string& cmd, std::ostream& out) {
    return install::run(cmd, [&out](const std::string& line) { out << line << "\n"; }).status == 0 ? 0 : 1;
}

std::string just(const Env& env, const std::string& recipe) {
    return std::format("just -f {} port={} {}", install::quote(env.repo + "/justfile"), env.port, recipe);
}

bool wait_for_pid_file(const Env& env) {
    std::error_code ec;
    for (int i = 0; i < 30; ++i) {
        if (fs::exists(env.server_pid(), ec)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

// On macOS the start command itself says "already running" when the pid file names a live server; "started" after it
// would claim a start that did not happen
int start_release(const Env& env, std::ostream& out) {
    if (const std::string err = env.refresh_unit(); !err.empty()) return out << err << "\n", 1;
    bool running = false;
    const auto line = [&](const std::string& l) {
        out << l << "\n";
        running = running || l == "already running";
    };
    if (install::run(env.start_command(), line).status != 0) return 1;
    if (running) return 0;
    // The started program writes the pid file itself; `tona stop` right after this command needs it
    if (env.macos && !std::getenv("TONA_START_CMD") && !wait_for_pid_file(env)) {
        out << "the server did not start (no " << env.server_pid() << " after 3 s); see " << env.server_log() << "\n";
        return 1;
    }
    out << "started on port " << env.port << " (progress: tona, log: "
        << (env.macos ? env.server_log() : std::string("journalctl --user -u tona-server -f")) << ")\n";
    return 0;
}

// Runs the stop command; `idle` is set when it said that nothing was running
int stop_release(const Env& env, std::ostream& out, bool& idle) {
    const auto line = [&](const std::string& l) {
        out << l << "\n";
        idle = idle || l == "not running";
    };
    return install::run(env.stop_command(), line).status == 0 ? 0 : 1;
}

int run_release(const Env& env, const std::string& action, std::ostream& out) {
    if (action == "start") return start_release(env, out);
    if (action == "restart" && !env.macos) {
        if (const std::string err = env.refresh_unit(); !err.empty()) return out << err << "\n", 1;
        return relay(env.restart_command(), out);
    }
    if (action == "stop") {
        bool idle = false;
        const int rc = stop_release(env, out, idle);
        if (rc == 0 && !idle) out << "stopped\n";
        return rc;
    }
    bool idle = false;  // nothing of ours running is fine before a restart
    if (const int rc = stop_release(env, out, idle); rc != 0) return rc;
    return start_release(env, out);
}

}  // namespace

bool is_service_command(const std::string& word) { return word == "start" || word == "stop" || word == "restart"; }

int run_autostart_command(const Env& env, const std::string& arg, std::ostream& out, std::ostream& err) {
    if (!env.release) {
        err << "tona: autostart is for a release installation; a checkout has just install-service\n";
        return 2;
    }
    if (arg == "status") {
        out << (install::autostart_on(env.macos) ? "on: the server starts at login\n"
                                                 : "off: the server starts with tona start\n");
        return 0;
    }
    if (arg != "on" && arg != "off") {
        err << "usage: tona autostart on|off|status\n";
        return 2;
    }
    const std::string error = install::set_autostart(env.macos, env.root, arg == "on", false,
                                                     [&out](const std::string& l) { out << l << "\n"; });
    if (!error.empty()) {
        err << "tona: " << error << "\n";
        return 1;
    }
    out << (arg == "on" ? "on: the server starts at the next login (now: tona start)\n"
                        : "off: the server no longer starts at login\n");
    return 0;
}

std::string layout_error(const Env& env) {
    std::error_code ec;
    if (!env.release) return {};
    if (!fs::is_directory(fs::path(env.root) / "versions", ec))
        return "neither a checkout nor a release installation (no justfile and engine/ above the program, no versions/ "
               "in " +
               env.root + "); run tona setup";
    if (!fs::exists(fs::path(env.root) / "current", ec))
        return "no version is installed in " + env.root + "; run tona setup";
    return {};
}

int run_service_command(const Env& env, const std::string& action, std::ostream& out) {
    if (env.release) return run_release(env, action, out);
    // `just start` builds first and writes the unit of the checkout; its own messages are the output
    if (action == "restart") return relay(just(env, "stop"), out) != 0 ? 1 : relay(env.start_command(), out);
    return relay(action == "start" ? env.start_command() : just(env, action), out);
}

}  // namespace tona::client
