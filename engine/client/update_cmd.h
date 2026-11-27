// `tona update [--yes] [--rollback]`: the update of key `u` without the screen. Check, show the version and the size,
// ask, install, restart the server, and go back by itself when the new version does not come up. `--rollback` goes back
// to the previous version on purpose. The same Updater as the centre uses, so both do the same.
#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "client/commands.h"
#include "update/updater.h"

namespace tona::client {

// `args` are the words after `update`. Exit codes as in commands.h (0 ok or nothing to do, 1 failed, 2 usage, 4 a
// question and no terminal to ask on).
int run_update(update::Updater& updater, const std::vector<std::string>& args, Io& io,
               std::chrono::milliseconds poll = std::chrono::milliseconds(200));

// `tona update` of this installation: the updater of the process (make_default_updater)
int run_update_command(const Env& env, const std::vector<std::string>& args, Io& io);

}  // namespace tona::client
