// Configuration of the installation: work/tona.conf in the repository, lines KEY=value. Read by just (dotenv),
// scripts/tona-service.sh and the client; the service gets it as EnvironmentFile. Environment variables win.
#pragma once

#include <map>
#include <string>

#include "core/plan.h"

namespace tona::install {

std::string conf_path(const std::string& repo);
// file contents for the settings; paths to model, drafter and profile come from the catalog
std::string render(const Settings& s, const std::string& repo);
// lays an existing file over the defaults (unknown keys are ignored)
Settings load(const std::string& path, Settings defaults);
bool save(const Settings& s, const std::string& repo);

// KEY=value lines of a settings file as a map (quotes removed); empty if the file is missing
using Values = std::map<std::string, std::string>;
Values parse_values(const std::string& text);
Values read_values(const std::string& path);
// Sets single keys in an existing file (other lines stay as they are, new keys are appended), atomically. The client
// uses it for keys that are not part of Settings (Settings is what the installer decides). Returns an error text.
std::string set_values(const std::string& path, const Values& changes);

}  // namespace tona::install
