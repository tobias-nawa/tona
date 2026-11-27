// The agents the user started from the Agents page (shown under "started recently"). A small JSON file in the state
// folder; a missing or damaged file is an empty history, never an error.
#pragma once

#include <string>
#include <vector>

#include "client/ui.h"

namespace tona::client {

constexpr size_t kMaxLaunches = 10;

std::vector<LaunchEntry> load_launches(const std::string& state_dir);
// Puts the entry first and writes the file (newest first, at most kMaxLaunches); returns an error text
std::string record_launch(const std::string& state_dir, const LaunchEntry& entry);

}  // namespace tona::client
