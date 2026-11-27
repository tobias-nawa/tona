// Where every run of the installer starts: the detected machine, the settings it recommends, an existing configuration
// laid over them, the choices named on the command line, and the selection that results. The pages and the commands
// without a screen both begin here, so they begin from the same state.
#pragma once

#include <string>

#include "core/cli_args.h"
#include "core/flow.h"
#include "core/module.h"
#include "core/steps.h"

namespace tona::install {

struct Start {
    Hardware h;
    const Module* module = nullptr;  // nullptr: no backend for this hardware
    HwMatch match;
    Settings recommended;  // what the module recommends, for the labels "Recommendation: ..."
    Settings settings;     // recommendation, existing configuration, room, and the selection applied
    Selection selection;
    bool had_conf = false;
};

// Folder the models of this target go to
std::string models_dir_of(const Target& t);

// `interactive`: a fresh machine starts with the default selection of the pages (coding and knowledge); without a
// screen it starts with the coding recommendation only, because nothing large is installed unasked. On a name that
// is wrong `err` says what and returns false.
bool make_start(const Options& o, const Target& t, bool interactive, Start& out, FlowError& err);

}  // namespace tona::install
