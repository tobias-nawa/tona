// `tona setup`: the guided setup and its commands without a screen (see setup.cpp). The program `tona` calls it.
#pragma once

#include <string>
#include <vector>

namespace tona::install {

// `args` are the words after `tona setup`. Returns an exit code (core/exit_codes.h).
int run_setup(const std::vector<std::string>& args);

}  // namespace tona::install
