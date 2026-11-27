// `tona start --foreground`: the one way the server starts. The systemd unit and the macOS LaunchAgent run exactly this
// command, so the settings live in one place (tona.conf) and a change takes effect at the next start. It reads them,
// then becomes tona-server (exec): the pid stays the same, so the unit, the pid file and the updater track the server.
#pragma once

#include <iosfwd>
#include <string>

#include "client/env.h"
#include "core/unit.h"

namespace tona::client {

struct ForegroundPlan {
    install::ServerLaunch launch;
    double ram_gb = 0;  // the RAM limit to enforce as MemoryMax under systemd
    std::string error;  // set: nothing can start
};

// The command line and environment of the server from the settings of the installation
ForegroundPlan plan_foreground(const Env& env);

// The name of the systemd unit a process runs in ("" outside systemd), from the text of /proc/<pid>/cgroup
std::string own_unit_name(const std::string& cgroup_text);

// Applies the plan and replaces this process with the server; returns only on failure (1, message on `err`)
int run_foreground(const Env& env, std::ostream& err);

}  // namespace tona::client
