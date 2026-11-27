// TONA_PRIORITY: the user's answer to "who goes first when memory is short" (work, balanced, tona), written by the
// installer and by `tona config set priority`. Read by the Metal backend's reserve; the reference backend does not
// read it yet.
#pragma once

#include <string_view>

#include "backend/budget.h"
#include "backend/env.h"

namespace tona::backend {

enum class Priority { Work, Balanced, Tona };

// Unknown or unset counts as balanced, the default of the installer, so a typo never takes memory from the user
inline Priority priority_of(std::string_view v) {
    if (v == "work") return Priority::Work;
    if (v == "tona") return Priority::Tona;
    return Priority::Balanced;
}

inline Priority priority() {
    const char* v = env::text("TONA_PRIORITY");
    return v ? priority_of(v) : Priority::Balanced;
}

// The reserve's 1.5 GB of TIERED_CACHE_MIN_GB, which on the Mac is room for what passes wire: "tona" (server first)
// gives it to the experts in a start without TONA_RAM_GB, 19.6 against 16.7 tok/s decode on the Coder IQ1_M, paid
// with 28 to 525 MB of other programs' memory compressed per minute of requests
// (research/measurements/2026-11-17-mac-iq1m-reserve.md). With TONA_RAM_GB (the installer's start) it stays: there the
// reserve would fall to 1.5 GB and the Mac swapped (research/measurements/2026-11-25-mac-priority-check.md), and the
// installer's tona split already raises the limit. Work and balanced keep it: no unmeasured value in between.
inline double pass_room_gb(Priority p) { return p == Priority::Tona && ram_limit_gb() < 0 ? 0.0 : TIERED_CACHE_MIN_GB; }

}  // namespace tona::backend
