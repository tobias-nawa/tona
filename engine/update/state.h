// state.json: the highest accepted seq (rollback protection) and whether update checks are on
#pragma once

#include <cstdint>
#include <string>

#include "update/layout.h"

namespace tona::update {

struct State {
    uint64_t seq = 0;
    bool checks_enabled = false;  // default off: nothing leaves the machine until the user turns it on
};

struct StateResult {
    bool ok = false;
    std::string error;
    State state;
};

// A missing file is the default state; an unreadable or malformed one is an error (never silently seq 0, which
// would re-open the door to an old release)
StateResult load_state(const Layout& layout);

// Atomic: written to a temporary file, synced, renamed
std::string save_state(const Layout& layout, const State& state);

}  // namespace tona::update
