#include "core/room.h"

#include <algorithm>
#include <cmath>

namespace tona::install {

namespace {

constexpr double GIB = 1073741824.0;

// Share of RAM (after the system) kept for the user's programs per answer. 64 GB gives 27, 20 and 8 GB; the Tona
// share (33, 40, 52 GB) stays above the 31.6 GB from which the reference backend pins all experts, where more RAM
// gained nothing (research/measurements/2026-10-27-ram-limit.md).
constexpr double kProgramsShare[] = {0.45, 0.33, 0.13};

}  // namespace

const char* priority_id(Priority p) { return p == Priority::Work ? "work" : p == Priority::Tona ? "tona" : "balanced"; }

Priority priority_from_id(const std::string& id) {
    return id == "work" ? Priority::Work : id == "tona" ? Priority::Tona : Priority::Balanced;
}

const char* use_id(bool server_only) { return server_only ? "server" : "desktop"; }

double nominal_ram_gb(const Hardware& h) {
    return std::max(4.0, std::round(static_cast<double>(h.ram_total) / GIB / 4) * 4);
}

RoomPlan room_plan(const Hardware& h, bool server_only, Priority p, double vram_limit) {
    RoomPlan r;
    r.ram_total = nominal_ram_gb(h);
    r.system = kSystemGb;
    const double usable = r.ram_total - r.system;
    r.programs = server_only ? kServerProgramsGb : std::round(usable * kProgramsShare[static_cast<int>(p)]);
    r.tona = std::max(kMinTonaGb, usable - r.programs);
    r.programs = usable - r.tona;
    for (const Gpu& g : h.gpus) {
        if (g.integrated) continue;
        r.unified = g.vendor == "Apple";
        r.vram_total = std::round(static_cast<double>(g.vram_total) / GIB);
        break;
    }
    r.vram_tona = vram_limit;
    r.vram_screen = std::max(0.0, r.vram_total - vram_limit);
    return r;
}

std::vector<RoomPlan> room_proposals(const Hardware& h, bool server_only, double vram_limit) {
    std::vector<RoomPlan> out;
    for (Priority p : {Priority::Work, Priority::Balanced, Priority::Tona})
        out.push_back(room_plan(h, server_only, p, vram_limit));
    return out;
}

void apply_room(Settings& s, const RoomPlan& r, bool server_only, Priority p) {
    s.ram_gb = r.tona;
    s.use = use_id(server_only);
    s.priority = priority_id(p);
}

}  // namespace tona::install
