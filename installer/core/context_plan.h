// Parallel agents and context per agent: "auto" keeps the RAM for the experts to stay pinned and gives every agent the
// largest context that fits in what is left.
#pragma once

#include <string>
#include <vector>

#include "core/plan.h"

namespace tona::install {

class Module;

// Lowest context "auto" gives: a coding agent's system prompt and tool definitions take about 20,000 tokens
// (docs/contribute/development.md), so 32K leaves 12K to work in; a smaller context cannot hold a task
constexpr int kContextFloor = 32768;
constexpr int kMaxAgents = 4;

// What the user can pick besides "auto", smallest first; 256K is the maximum of every model Tona offers. "auto"
// gives the floor where not even the first step fits.
const std::vector<int>& context_steps();

struct ContextBudget {
    double ram_gb = 0;            // RAM limit of the engine
    double pin_gb = 0;            // RAM at which all experts stay pinned (the KV cache comes on top)
    double kv_bytes_per_pos = 0;  // KV cache of one agent per context position
    int agents = 1;
};

// The largest step whose KV cache for all agents fits in the RAM above `pin_gb`, never below kContextFloor
int plan_context(const ContextBudget& b);

// "256K" for 262144 (1K = 1024 tokens)
std::string context_text(int ctx);
// "256K" or "256k" to tokens; 0 if the text is not a size
int parse_context(const std::string& text);

// Cuts agents and context to what the backend offers (Module::limits): at most max_agents, no "auto" where there is no
// plan, a context from the backend's sizes (the largest not above the setting, else the smallest)
void apply_limits(const Module* m, Settings& s);

// Sets s.ctx from the backend's plan when s.ctx_auto; a fixed size stays. No module (no supported GPU): unchanged.
void resolve_context(const Module* m, Settings& s);

}  // namespace tona::install
