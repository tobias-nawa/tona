// Backend module of the installer, one per platform backend (docs/backends/README.md). It detects its GPU, recommends
// settings, estimates the speed and names the extra packages it needs. A new backend brings its module and adds it to
// modules() (modules/registry.cpp).
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "core/context_plan.h"
#include "core/hw.h"
#include "core/plan.h"
#include "core/steps.h"

namespace tona::install {

struct Package;

// What a backend lets the user choose for the parallel agents and the context: the pages offer exactly this and the
// settings are cut to it (apply_limits). The values are data of the backend, so a new measurement raises them there.
struct Limits {
    int max_agents = kMaxAgents;
    std::vector<int> contexts;  // the sizes on offer; empty: the usual steps (context_steps())
    bool auto_context = true;   // "auto" exists: the backend has a plan for it (auto_context)
};

class Module {
public:
    virtual ~Module() = default;
    virtual std::string name() const = 0;   // as backend::Info::name, e.g. "hip-gfx1200"
    virtual std::string title() const = 0;  // for humans
    // the GPU this backend would use, else nullptr
    virtual const Gpu* pick(const Hardware& h) const = 0;
    virtual Settings recommend(const Hardware& h) const = 0;
    virtual Limits limits() const { return {}; }
    // Context per agent for `ctx_auto`: what fits besides the pinned experts; default: the setting as it is
    virtual int auto_context(const Settings& s) const { return s.ctx; }
    // Speed of the model `model` (catalog id) under these settings; no numbers for a model without a card
    virtual Estimate estimate(const Hardware& h, const Settings& s, const std::string& model) const = 0;
    virtual std::vector<Note> check(const Hardware& h, const Settings& s) const = 0;
    virtual std::vector<Package> packages(const Hardware& h) const = 0;
    // steps before the build (fetch and verify the toolchain)
    virtual std::vector<Step> toolchain(const std::string& repo) const = 0;
};

const Module& hip_gfx1200_module();  // modules/hip_gfx1200.cpp
const Module& metal_apple_module();  // modules/metal_apple.cpp
// The model the Mac recommends: the pre-tiled coder-dmt-164 while it and its MTP head can be installed (`installable`
// says whether an entry id has a pinned repository and revision), else the model that is published (coder-iq1_m)
std::string metal_apple_model(const std::function<bool(const std::string&)>& installable);
const std::vector<const Module*>& modules();
// first module whose GPU is present, else nullptr
const Module* find_module(const Hardware& h);

}  // namespace tona::install
