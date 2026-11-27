// Commands without a screen (see main.cpp for which options lead here)
#pragma once

#include "core/cli_args.h"
#include "core/steps.h"

namespace tona::install {

// Prints the plan (text, or JSON with --json); with dry = false also runs it. Returns an exit code (core/exit_codes.h).
int run_headless(const Options& o, const Target& t, bool dry);

}  // namespace tona::install
