// The settings the client shows and `tona config get|set` changes: one table, one set of get/set functions, so the
// Settings page and the command can never disagree. Values live in Settings (what the installer decides, written
// through the installer's config code) or, for keys the installer does not own, as extra lines of the same file.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "client/env.h"
#include "core/config.h"
#include "core/plan.h"

namespace tona::client {

enum class Kind { Choice, Number, Toggle };

struct SettingSpec {
    std::string key;    // as in `tona config set <key>`
    std::string group;  // text key of the group on the Settings page
    Kind kind = Kind::Choice;
    std::vector<std::string> choices;   // Choice: the allowed values
    double min = 0, max = 0, step = 0;  // Number
    std::string unit;                   // "GB"
    std::string help;                   // text key
    std::string conf_key;               // key in tona.conf for the extra (non-Settings) ones, else empty
};

const std::vector<SettingSpec>& setting_specs();
const SettingSpec* find_setting(const std::string& key);

// Settings and the client's own extra keys, as read from tona.conf
struct Config {
    install::Settings s;
    install::Values extras;  // TONA_USE, TONA_PRIORITY, TONA_CLOSE
    // Sets the context of "auto" after agents, context or the RAM limit changed (the backend module's plan); empty:
    // the context stays as it is
    std::function<void(install::Settings&)> resolve;
};

// A resolver for this machine: detects the hardware once, on the first call, and uses its backend module
std::function<void(install::Settings&)> local_resolver(const Env& env);

Config load_config(const Env& env);
// Writes the Settings (installer code) and the extras into the same file; a release also writes the unit
std::string save_config(const Env& env, const Config& c);

// Value as text ("balanced", "40", "on", context "auto" or "128K"); empty for an unknown key
std::string get_value(const Config& c, const std::string& key);
// Checks the value against the spec and sets it; returns an error text, empty on success
std::string set_value(Config& c, const std::string& key, const std::string& value);
// One step of a Number or Choice up (+1) or down (-1), cut at the limits; Toggle flips
std::string step_value(const Config& c, const SettingSpec& spec, int direction);

}  // namespace tona::client
