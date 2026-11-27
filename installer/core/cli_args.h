// The command line of tona setup. The options that name a choice (--task, --lang, --models) mean what the same
// choice on the pages means (flow.h).
#pragma once

#include <string>

#include "core/flow.h"

namespace tona::install {

struct Options {
    std::string repo, conf, model_from, install_dir, models_dir, cache_dir, priority, use, updates, port;
    NamedChoice named;  // --task, --lang, --models
    bool print = false, yes = false, dry = false, release = false, autostart = false, json = false, help = false,
         help_dev = false, dump_pages = false, no_modify_path = false, modify_path = false;

    // Something is named, so the user wants this choice and no questions about it
    bool names_something() const { return !named.tasks.empty() || !named.languages.empty() || !named.models.empty(); }
};

// Parses the arguments; false on an unknown option, a missing value or a value that is not allowed (`error` says
// which).
bool parse_args(int argc, const char* const* argv, Options& o, std::string& error);

extern const char* const kUsage;
// The options for developers: `tona setup --help-dev`
extern const char* const kUsageDev;

}  // namespace tona::install
