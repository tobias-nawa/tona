// The plan of an installation as data and as text: what the commands without a screen print (--dry-run, --print,
// --json), made from the same choices as the pages. Nothing here changes the machine.
#pragma once

#include <string>
#include <vector>

#include "core/flow.h"
#include "core/hw.h"
#include "core/module.h"
#include "core/pkgs.h"
#include "core/plan.h"
#include "core/space.h"
#include "core/steps.h"

namespace tona::install {

struct PlanDoc {
    Hardware hw;
    const Module* module = nullptr;  // nullptr: no backend for this hardware
    HwMatch match;
    Selection selection;
    Settings settings;
    std::vector<Note> notes;
    std::vector<Package> packages;
    std::vector<Step> steps;
    std::string models_dir;   // where the models go, for what is already there
    uint64_t to_load = 0;     // bytes still to download
    uint64_t here = 0;        // bytes of the chosen models that are already there (verified on install)
    uint64_t free_bytes = 0;  // under the models folder
};

// Everything but the steps: the caller adds them (they need the target)
PlanDoc make_plan(const Hardware& h, const Module* m, const Selection& sel, const Settings& s, bool release,
                  const std::string& models_dir);

// 0 when the plan can run, else the ExitCode that says why not (exit_codes.h)
int plan_exit_code(const PlanDoc& p);

// The plan as text for a terminal or a log
std::string plan_text(const PlanDoc& p);
// The plan as one JSON document (schema in docs/use/installer.md)
std::string plan_json(const PlanDoc& p);

}  // namespace tona::install
