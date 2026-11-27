// `tona start | stop | restart | autostart`: the server in the background. A checkout does it through `just` (the same
// recipes a developer uses), a release installation through the systemd user unit tona-server that tona setup wrote
// (macOS: a background process of `tona start --foreground`). The output of the underlying command is shown as it
// comes.
#pragma once

#include <iosfwd>
#include <string>

#include "client/env.h"

namespace tona::client {

// True for the words this module runs
bool is_service_command(const std::string& word);

// Runs `action` (start, stop or restart); returns an exit code (0 ok, 1 failed)
int run_service_command(const Env& env, const std::string& action, std::ostream& out);

// `tona autostart on|off|status` (release installation): the start of the server at login (core/autostart.h). Exit
// code 0 ok, 1 failed, 2 wrong argument or a checkout.
int run_autostart_command(const Env& env, const std::string& arg, std::ostream& out, std::ostream& err);

// Error text when this is neither a checkout nor a release installation (the program was copied somewhere else), else
// empty
std::string layout_error(const Env& env);

}  // namespace tona::client
