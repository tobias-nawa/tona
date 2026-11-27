// Speed of a model on a machine, as a range scaled from one measured anchor. Decode of the MoE models is bound by the
// bytes read per token: dense weights and the experts in GPU memory at its bandwidth, the experts in RAM at the RAM
// bandwidth, the experts left on the drive at its speed. The anchor is a machine where the rate was measured; this
// machine's rate is the anchor's times the ratio of the seconds per token. Each backend module (module.h) describes
// its machine and its anchor; this file holds the shared model.
#pragma once

#include <cstdint>
#include <string>

#include "core/plan.h"

namespace tona::install {

// What the model needs from a machine, in the units of its backend's memory plan
struct Machine {
    double gpu_gbs = 0;    // GB/s the GPU reads its own memory at (0: unknown)
    double units = 0;      // GPU compute units (CUs or cores), for prefill (0: unknown)
    double fast_gb = 0;    // GB of experts GPU memory holds beside the dense weights
    double ram_gb = 0;     // GB of RAM that can hold the experts GPU memory does not (0: none, unified memory)
    double ram_gbs = 0;    // GB/s the CPU reads RAM at (0: unknown)
    double drive_gbs = 0;  // GB/s the drive serves experts at (0: unknown)
    // Behavior of the engine the bandwidths do not show, measured on the anchor's backend (1: none)
    double decode_factor = 1, prefill_factor = 1;
};

// Bytes of a model, from its parameter counts and file size (the derivation sits next to each table)
struct Traffic {
    double dense_gb = 0;          // read for every token, from GPU memory
    double expert_gb = 0;         // experts read per token
    double experts_gb = 0;        // all experts together
    double kv_bytes_per_pos = 0;  // KV bytes one token reads per position of its context (0: the context costs nothing)
};

// What the user chose: parallel agents and the context each has. One pass of the engine serves every agent one token.
struct Load {
    int agents = 1;
    double context = 0;  // tokens per agent
};

struct Anchor {
    Machine machine;
    double decode = 0, prefill = 0;  // tok/s measured there
    uint32_t context_tokens = 0;     // context of the decode rate
    std::string label;               // "RX 9060 XT 16 GB, Ryzen 9 9950X, 64 GB DDR5-6000"
};

// Share of the expert reads served by the fast tier when it holds `share` of all experts (0 to 1). The routing is
// skewed: llama.cpp issue 29887 on this model gives 85 % hits at a third of the experts cached and 94 % at two
// thirds (docs/hardware/README.md, step 4); linear between these points.
double hit_rate(double share);

// Seconds of one pass on this machine: dense weights once, the union of the experts the agents route to (it grows with
// the agents, up to all experts), and every agent's KV cache. With one agent and no context this is the bytes of one
// token.
double seconds_per_pass(const Machine& m, const Traffic& t, const Load& load = {});
double seconds_per_token(const Machine& m, const Traffic& t);

// Fills decode, prefill, width, unknown and anchor of `e` for `t` on `m`. `same_class`: the machine is of the
// anchor's class (plus or minus 15 %), else plus or minus 30 %. Each unknown input widens the range by 15 points
// (60 at most); an unknown GPU bandwidth gives no estimate (e.setup says so).
void project(const Anchor& a, const Machine& m, const Traffic& t, bool same_class, Estimate& e);

// Fills per_agent and together of `e` for `load`: the anchor's rate (measured at its own context with one agent) times
// the ratio of the pass times. The range widens by 15 points for more than one agent and for a context more than 4
// times away from the anchor's: the further from the measured point, the wider the band.
void project_load(const Anchor& a, const Machine& m, const Traffic& t, const Load& load, Estimate& e);

// "58-78" for a range, one decimal below 10 tok/s
std::string range_text(const Range& r);

}  // namespace tona::install
