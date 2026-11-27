// Settings, recommendation and estimate of the installer. A backend module (module.h) recommends values for its
// hardware and estimates the speed; the front ends (TUI, --yes) only show them and let the user change them.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/hw.h"
#include "offer/offer.h"

namespace tona::install {

// everything the installer decides; written to work/tona.conf (config.h)
struct Settings {
    double ram_gb = 0;                  // RAM the engine may use, page cache included (TONA_RAM_GB, MemoryMax)
    double vram_gb = 0;                 // VRAM the engine may use (TONA_VRAM_GB)
    std::string model = "coder-iq1_m";  // entry of toolchain/models.toml
    std::string drafter = "mtp";        // mtp, dflash or none
    int ctx = 262144;                   // context per agent; with ctx_auto the module's plan (TONA_CTX)
    bool ctx_auto = true;               // "auto": experts pinned first, then the largest context (TONA_CTX_AUTO)
    int seqs = 2;                       // parallel agents, 1 to 4 (TONA_SEQS)
    int port = 8662;
    std::string effort = "medium";  // thinking effort tona starts Claude Code with
    bool oea = true;                // shared experts of parallel conversations (TONA_OEA_K0=0: off)
    std::string role_effort;        // effort per role (TONA_ROLE_EFFORT), empty: off
    bool clock_hold = false;        // keep the GPU clock up between requests (TONA_CLOCK_HOLD): the user's choice, off
    bool api_key = true;            // key for the API (~/.config/tona/api-key)
    double prompt_cache_gb = 16;    // checkpoints on the SSD (TONA_PROMPT_CACHE_GB)
    bool service = false;           // service that starts with the login
    bool shell = true;              // the command tona in ~/.local/bin
    bool modify_path = true;        // with it: put ~/.local/bin on PATH in the shell profiles (shell_path.h)
    std::string model_from;         // existing download to verify and link instead of downloading (not saved)
    // What the user chose on the pages (flow.h). `model` stays the one the server starts with; `models` are all that
    // are downloaded (empty: only `model`).
    std::vector<std::string> models, languages;
    std::string use = "desktop";        // "desktop": the user works on this PC, "server": only for Tona (room.h)
    std::string priority = "balanced";  // "work", "balanced", "tona": who has priority in a shortage (room.h)
    bool network = false;               // listen for other PCs (with the key), release installation only
    // Daily check for a new release (state.json of the install root, the same switch as key c in tona); on unless the
    // user says no: one line on the last page asks once. Not saved in tona.conf: state.json holds it.
    bool update_checks = true;
};

// Every model that gets downloaded, the one the server starts with first
std::vector<std::string> models_of(const Settings& s);

struct Range {
    double lo = 0, hi = 0;
};

// Speed of one model on this machine, for the plan: a range scaled from a measured anchor by this machine's memory
// bandwidths (speed.h). hi == 0: no estimate.
struct Estimate {
    std::string model;            // catalog id the numbers belong to
    Range decode, prefill;        // tok/s: decode (code, at `context_tokens`) and prefill
    uint32_t context_tokens = 0;  // context of the anchor's decode rate
    // Decode under the chosen settings: the rate of each agent and of all agents together, at `load_context` tokens per
    // agent. Scaled from the anchor's measured point by the cost of the KV cache and of the experts the agents add;
    // hi == 0 without an estimate.
    Range per_agent, together;
    int agents = 1;
    uint32_t load_context = 0;
    double width_load = 0;  // half width of per_agent and together
    double width = 0;       // half width of the range, 0.15 = plus or minus 15 %
    // what the estimate used: key (gpu-bandwidth, gpu-size, ram-bandwidth, drive-speed, gpu-experts, ram-experts) and
    // value; describe.cpp names them
    std::vector<std::pair<std::string, double>> inputs;
    std::vector<std::string> unknown;  // keys of the inputs the detection did not find; they widen the range
    std::string anchor;                // "measured on <machine>", the one line of the details
    double load_s = 0;                 // server start until ready (loading, pinning), 0: unknown
    bool pinned = true;                // all experts stay pinned in RAM under these settings
    std::string setup;                 // why there is no estimate, in words (empty with one)
};

// note on a choice: warn = it runs badly or not at all like this
struct Note {
    bool warn = false;
    std::string text;
};

// models and drafters from toolchain/models.toml (through tools/get_model.py --list)
struct ModelInfo {
    std::string id, title, kind, license, quality;  // kind: "model" or "drafter"
    std::string first_file;                         // name of the first file (the server gets this one)
    double size_gb = 0;
    bool tested = false;   // runs in the engine and is measured (the installer offers nothing else)
    bool present = false;  // already under work/models/<id>/ and verified
};
std::vector<ModelInfo> load_catalog(const std::string& repo);

}  // namespace tona::install
