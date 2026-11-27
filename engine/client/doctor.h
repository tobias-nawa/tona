// tona doctor: what is wrong, with the fix for each line. `gather_facts` looks at the machine, `run_checks` is a pure
// function of the facts (unit-tested); the screen "server not running" shows the same lines.
#pragma once

#include <string>
#include <vector>

#include "client/env.h"
#include "core/plan.h"
#include "core/port.h"

namespace tona::client {

using install::PortState;  // core/port.h: the setup asks the same question

struct Facts {
    bool release = false;
    std::string server_binary;
    bool server_binary_exists = false;
    bool toolchain_needed = false, toolchain_exists = false;
    std::string toolchain_path;
    std::string model_id, drafter_id;
    bool model_present = false, drafter_present = false;
    std::string profile_path;
    bool profile_needed = false, profile_exists = false;
    int port = 8662;
    PortState port_state = PortState::Free;
    std::string unit_state;  // "active", "inactive", "failed", "" (no systemd)
    double ram_total_gb = 0, vram_total_gb = 0;
    double ram_limit_gb = 0, vram_limit_gb = 0;  // from the settings, 0 = unset
    bool key_needed = false, key_exists = false;
};

struct Check {
    std::string name;    // "binaries", "models", ...
    bool ok = true;      // false = warn
    std::string detail;  // what was found
    std::string fix;     // what to do (only when not ok)
};

Facts gather_facts(const Env& env, const install::Settings& s);
std::vector<Check> run_checks(const Facts& f);
using install::probe_port;

}  // namespace tona::client
