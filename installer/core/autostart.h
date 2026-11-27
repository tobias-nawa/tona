// Start of the server at login: optional, off unless the user turns it on (tona setup's "At login", or
// `tona autostart on`). Linux enables the systemd user unit tona-server that tona setup wrote. macOS has no systemd:
// a launchd LaunchAgent runs `<root>/current/bin/tona start --foreground` at login, the start the unit runs on Linux;
// it writes the pid file, so `tona stop` and the updater's restart work on that server too.
#pragma once

#include <string>

#include "core/steps.h"

namespace tona::install {

constexpr const char* kLaunchAgentLabel = "com.tobiasnawa.tona";

// ~/Library/LaunchAgents/com.tobiasnawa.tona.plist
std::string launch_agent_path();

// The LaunchAgent for an installation in `root`; the output of the server goes to <root>/autostart.log
std::string render_launch_agent(const std::string& root);

// Turns the start at login on or off. `now` also starts the server at once (tona setup), as `enable --now` does;
// `tona autostart on` leaves that to the next login. Output of the commands goes to `out`; returns an error text,
// empty on success.
std::string set_autostart(bool macos, const std::string& root, bool on, bool now, const LineFn& out);

// Whether the server starts at login
bool autostart_on(bool macos);

}  // namespace tona::install
