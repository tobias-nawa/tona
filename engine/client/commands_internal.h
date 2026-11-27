// The commands of commands_more.cpp, called by run_command
#pragma once

#include <string>
#include <vector>

#include "client/commands.h"

namespace tona::client {

int run_use(std::vector<std::string> args, const Env& env, Services& svc, Io& io);
int run_config(std::vector<std::string> args, const Env& env, Services& svc, Io& io);
int run_doctor(std::vector<std::string> args, const Env& env, Services& svc, Io& io);

// Model id from what the user typed: the id itself or the display name, any case ("" if no model matches)
std::string resolve_model(const Offer& offer, const std::string& typed);

}  // namespace tona::client
