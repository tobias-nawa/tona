// Making the new version run: wait until the server is idle, restart it, check that it comes up, and go back to the
// previous version if it does not. No HIP call and nothing blocking on the GPU: the server is only asked over HTTP.
#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "update/layout.h"

namespace tona::update {

struct SwitchConfig {
    std::string status_url = "http://127.0.0.1:8662";
    // Restarts the server; returns false with a reason. Replaceable so tests never call systemctl.
    std::function<bool(std::string& why)> restart;
    // Whether the server process that the restart started still exists (the unit is active, the pid file's process
    // runs). Empty: unknown. A server that is gone ends the wait at once instead of after ready_seconds.
    std::function<bool()> alive;
    // 17.2 s model load (research/measurements/2026-10-23-pinned-buffers.md) with room for a slow disk
    int ready_seconds = 180;
    int idle_seconds = 3600;  // longest wait for running requests to finish
    int poll_ms = 500;
    const std::atomic<bool>* cancel = nullptr;
    std::function<void(const std::string&)> say;  // progress text for the UI
};

struct SwitchOutcome {
    enum class Kind { Switched, RolledBack, Failed };
    Kind kind = Kind::Failed;
    std::string message;
};

// `systemctl --user restart tona-server`, without a shell
bool restart_with_systemctl(std::string& why);
// `systemctl --user is-active tona-server`: false while the unit failed or waits to restart after a crash
bool systemd_unit_active();

// Runs argv (no shell) and waits at most `seconds`
bool run_command(const std::vector<std::string>& argv, int seconds, std::string& why);

// After install_release: restart onto `current`; if the new version is not ready in time, point `current` back at
// `previous`, restart again and report RolledBack.
SwitchOutcome switch_to_current(const Layout& layout, const SwitchConfig& cfg);

// The user's way back: swap current and previous, restart. Kind::Switched if the older version came up.
SwitchOutcome roll_back_to_previous(const Layout& layout, const SwitchConfig& cfg);

// /status answers at all (ready or not)
bool server_answers(const std::string& status_url);
// /status says no request is running (or no server answers)
bool server_is_idle(const std::string& status_url);
// /status says the engine is loaded and has no load error
bool server_is_ready(const std::string& status_url);

}  // namespace tona::update
