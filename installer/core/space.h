// What the chosen models take from the drive and how long loading takes, and what the user can do when the drive is
// too small. The numbers come from the pinned catalog (sizes of the files), nothing is guessed.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/flow.h"
#include "core/plan.h"

namespace tona::install {

// Room besides the models: checkpoints of conversations on the drive (design of the installer, page "Zu wenig
// Platz": "etwa 5 GB Puffer")
constexpr uint64_t kBufferBytes = 5'000'000'000ULL;
// The line speed the time estimate assumes; it is an assumption and the page says so (the design shows 100 MBit/s)
constexpr double kAssumedMbit = 100;

// Bytes of an entry of the catalog (all its files); 0 if unknown
uint64_t entry_bytes(const std::string& id);
// Bytes still to download for the models of `s` and the drafter of the first one: what is already there in
// `models_dir` and verified counts nothing
uint64_t bytes_to_load(const Settings& s, const std::string& models_dir);
// The same, with the bytes that are already in place and only get verified by the install step: a file with the
// pinned size counts as here (a mark of an earlier verification is not needed; hashing 58 GB is not done per frame)
struct LoadSplit {
    uint64_t download = 0, here = 0;
};
LoadSplit load_split(const Settings& s, const std::string& models_dir);
// One catalog entry, and one model with what it brings: its drafter (default_drafter). Every size on a page and in
// the plan comes from here, so the row, the sum and the plan agree.
LoadSplit entry_split(const std::string& id, const std::string& models_dir);
uint64_t model_bytes(const std::string& id);
// The model and its drafter together
LoadSplit model_split(const std::string& id, const std::string& models_dir);
// "To download" text: "6.5 GB", with "(62.5 GB already here, verified on install)" if something is there
std::string load_text(const LoadSplit& n);
double seconds_to_load(uint64_t bytes, double mbit_per_s = kAssumedMbit);

// Decimal GB with one decimal ("62.5 GB", "1.2 TB"): the only size format of pages, plan and messages
std::string size_text(double bytes);
// "≈ 1 h 40 min", "≈ 25 min", "< 1 min"
std::string duration_text(double seconds);

struct Place {
    std::string path;
    uint64_t free_bytes = 0;
};
// Writable mount points of other drives than the one of `current_dir`, with their free space (real file systems only,
// never tmpfs: it lives in RAM)
std::vector<Place> find_places(const std::string& current_dir);

struct SpaceContext {
    uint64_t free_bytes = 0;  // under the models folder
    std::string models_dir;
    std::vector<Place> places;  // release installation only; empty in a checkout
};

// A way out when the models do not fit. Settings are those the option would install.
struct SpaceOption {
    std::string label, note;
    Selection selection;  // the models this option keeps
    std::string place;    // empty: the same drive
};

bool fits(const Settings& s, const std::string& models_dir, uint64_t free_bytes);
// Options that fit, best first: a shorter list, a smaller alternative, another drive
std::vector<SpaceOption> space_options(const Settings& s, const Selection& sel, const std::string& cls,
                                       const SpaceContext& ctx);

}  // namespace tona::install
