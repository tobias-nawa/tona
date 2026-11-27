// `tona uninstall [--yes] [--models]` and `tona --version`: what the user does to the installation as a whole.
#pragma once

#include <string>
#include <vector>

#include "client/commands.h"
#include "client/control.h"

namespace tona::client {

// Stops the server, turns autostart off, removes the unit or LaunchAgent, the command link, the settings and the
// install directory with its versions, after one question. The models directory only after a second one (`--models`
// answers it, `--yes` answers the first). Exit codes as in commands.h.
int run_uninstall(const Env& env, Control& control, const std::vector<std::string>& args, Io& io);

// The version of a program at `exe`: the folder name in <root>/versions/<version>/bin/, "dev" anywhere else
std::string version_of_exe(const std::string& exe);

// The line of `tona --version`: version, platform, backend
std::string version_line(const std::string& exe);

}  // namespace tona::client
