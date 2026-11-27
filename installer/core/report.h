// Text for the terminal and --print: hardware, settings, estimate, packages, steps.
#pragma once

#include <string>
#include <vector>

#include "core/hw.h"
#include "core/module.h"
#include "core/pkgs.h"
#include "core/plan.h"
#include "core/steps.h"

namespace tona::install {

std::string gb(double bytes);  // "15.9 GB"
std::string hardware_text(const Hardware& h, const Module* m);
std::string settings_text(const Settings& s);
std::string estimate_text(const Estimate& e);
std::string packages_text(const Hardware& h, const std::vector<Package>& p);
std::string steps_text(const std::vector<Step>& st);
// true unless a required package is missing or unusable
bool packages_ok(const std::vector<Package>& p);

}  // namespace tona::install
