// How much memory Tona takes and how much stays for the user's own programs (page "Platz"). Three proposals come from
// the hardware, one per answer to "who has priority in a shortage". The engine side of the priority (temporary
// throttling of the other side) is not built yet: the answer is stored in the settings and the page says so.
#pragma once

#include <string>
#include <vector>

#include "core/hw.h"
#include "core/plan.h"

namespace tona::install {

enum class Priority { Work, Balanced, Tona };

const char* priority_id(Priority p);               // "work", "balanced", "tona"
Priority priority_from_id(const std::string& id);  // unknown: balanced
const char* use_id(bool server_only);              // "work", "server"

// For the bars of the page: all in GB
struct RoomPlan {
    double ram_total = 0;  // installed, rounded to the size on the box
    double system = 0;     // kept for the system
    double programs = 0;   // kept for the user's programs
    double tona = 0;       // what Tona may take (TONA_RAM_GB)
    double vram_total = 0, vram_tona = 0, vram_screen = 0;
    bool unified = false;  // Apple: no separate video memory
};

// Memory kept for the system in every case (design of the installer: "System 4 GB")
constexpr double kSystemGb = 4;
// Kept for own programs when the PC only serves Tona (the desktop session still needs some)
constexpr double kServerProgramsGb = 2;
// Lowest RAM limit the installer proposes: below it the engine has no room for its working memory
constexpr double kMinTonaGb = 8;

// Installed RAM as the box says: the system reports a bit less than 64 GB for a 64 GB kit
double nominal_ram_gb(const Hardware& h);

// The split for a use and a priority. vram_limit: the engine's VRAM limit of the settings.
RoomPlan room_plan(const Hardware& h, bool server_only, Priority p, double vram_limit);

// The three proposals of the page, in the order work, balanced, tona
std::vector<RoomPlan> room_proposals(const Hardware& h, bool server_only, double vram_limit);

// Puts the plan into the settings: RAM limit, use and priority
void apply_room(Settings& s, const RoomPlan& r, bool server_only, Priority p);

}  // namespace tona::install
