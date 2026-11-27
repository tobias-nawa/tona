#include "update/switch.h"

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <thread>

#include "third_party/httplib.h"
#include "third_party/nlohmann/json.hpp"
#include "update/activate.h"

extern char** environ;

namespace tona::update {

namespace {

using Clock = std::chrono::steady_clock;
using nlohmann::json;
using Kind = SwitchOutcome::Kind;

std::optional<json> read_status(const std::string& url) {
    httplib::Client cli(url);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(3, 0);
    const auto res = cli.Get("/status");
    if (!res || res->status != 200) return std::nullopt;
    json j = json::parse(res->body, nullptr, false);
    if (!j.is_object()) return std::nullopt;
    return j;
}

bool cancelled(const SwitchConfig& cfg) { return cfg.cancel && cfg.cancel->load(); }

void say(const SwitchConfig& cfg, const std::string& text) {
    if (cfg.say) cfg.say(text);
}

bool wait_for(const SwitchConfig& cfg, int seconds, const std::function<bool()>& done) {
    const auto end = Clock::now() + std::chrono::seconds(seconds);
    while (!done()) {
        if (cancelled(cfg) || Clock::now() >= end) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(cfg.poll_ms));
    }
    return true;
}

bool restart(const SwitchConfig& cfg, std::string& why) {
    return cfg.restart ? cfg.restart(why) : restart_with_systemctl(why);
}

// Restart and wait until the server is ready, or until its process is gone: a server that exited will not become
// ready, and the user should not wait out the whole load time to hear it
bool restart_and_wait(const SwitchConfig& cfg, std::string& why) {
    if (!restart(cfg, why)) return false;
    bool exited = false;
    const bool done = wait_for(cfg, cfg.ready_seconds, [&] {
        if (server_is_ready(cfg.status_url)) return true;
        exited = cfg.alive && !cfg.alive();
        return exited;
    });
    if (done && !exited) return true;
    why = exited ? "the new server exited" : "the server did not become ready in time";
    return false;
}

}  // namespace

bool server_answers(const std::string& status_url) { return read_status(status_url).has_value(); }

bool server_is_ready(const std::string& status_url) {
    const auto j = read_status(status_url);
    return j && j->value("ready", false) && !j->contains("error");
}

bool server_is_idle(const std::string& status_url) {
    const auto j = read_status(status_url);
    if (!j || !j->value("ready", false)) return true;  // nothing to interrupt
    const json engine = j->value("engine", json::object());
    const bool no_active = !engine.contains("active") || !engine["active"].is_array() || engine["active"].empty();
    return engine.value("state", "idle") == "idle" && engine.value("queue", 0) == 0 && no_active;
}

bool run_command(const std::vector<std::string>& argv, int seconds, std::string& why) {
    std::vector<char*> args;
    args.reserve(argv.size() + 1);
    for (const std::string& a : argv) args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);
    pid_t pid = 0;
    if (posix_spawnp(&pid, args[0], nullptr, nullptr, args.data(), environ) != 0) {
        why = "cannot start " + argv[0];
        return false;
    }
    const auto end = Clock::now() + std::chrono::seconds(seconds);
    int status = 0;
    while (waitpid(pid, &status, WNOHANG) == 0) {
        if (Clock::now() >= end) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            why = argv[0] + " timed out";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return true;
    why = argv[0] + " failed";
    return false;
}

bool restart_with_systemctl(std::string& why) {
    // Several restarts in a row (update, rollback, fallback) hit the unit's start limit; the result does not matter
    std::string ignored;
    run_command({"systemctl", "--user", "reset-failed", "tona-server"}, 10, ignored);
    return run_command({"systemctl", "--user", "restart", "tona-server"}, 120, why);
}

bool systemd_unit_active() {
    std::string why;
    return run_command({"systemctl", "--user", "is-active", "--quiet", "tona-server"}, 10, why);
}

SwitchOutcome switch_to_current(const Layout& layout, const SwitchConfig& cfg) {
    say(cfg, "Waiting for running requests to finish");
    if (!wait_for(cfg, cfg.idle_seconds, [&] { return server_is_idle(cfg.status_url); }))
        return {Kind::Failed, "requests are still running; the new version is installed but not started"};
    say(cfg, "Restarting the server");
    std::string why;
    if (restart_and_wait(cfg, why)) return {Kind::Switched, "running the new version"};
    const std::string failed = why;
    say(cfg, "New version failed (" + failed + "), going back");
    if (!swap_current_previous(layout))
        return {Kind::Failed, "new version failed (" + failed + ") and there is no previous version"};
    if (restart_and_wait(cfg, why))
        return {Kind::RolledBack, "new version failed (" + failed + "); the previous one runs again"};
    return {Kind::Failed, "new version failed (" + failed + ") and the previous one did not start (" + why + ")"};
}

SwitchOutcome roll_back_to_previous(const Layout& layout, const SwitchConfig& cfg) {
    say(cfg, "Waiting for running requests to finish");
    if (!wait_for(cfg, cfg.idle_seconds, [&] { return server_is_idle(cfg.status_url); }))
        return {Kind::Failed, "requests are still running; nothing was changed"};
    if (!swap_current_previous(layout)) return {Kind::Failed, "there is no previous version"};
    say(cfg, "Restarting the server");
    std::string why;
    if (restart_and_wait(cfg, why)) return {Kind::Switched, "running the previous version"};
    return {Kind::Failed, "the previous version did not start (" + why + ")"};
}

}  // namespace tona::update
