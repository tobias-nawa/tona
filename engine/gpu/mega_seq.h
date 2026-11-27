// State of the sequences (slots) the Mega backend serves side by side.
#pragma once

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "backend/mtp_track.h"
#include "gpu/mega_types.h"

namespace tona::gpu {

// Buffer with one copy per slot (state, KV, pool); dev[0] is the one build() made, host[s] the host pointer if it
// lives in pinned RAM (KV with a long context)
struct SeqBuf {
    uint8_t* dev[MEGA_SMAX] = {};
    uint8_t* host[MEGA_SMAX] = {};
    size_t bytes = 0;
};

// Host state per slot (the active slot's state lives in the usual fields, use_slot swaps)
struct SeqHost {
    int64_t pos = 0;
    std::vector<int32_t> tokens, batch, snap_tokens;
    int pending_commit = 0, parity = 0, pass_nb = 1, snap_commit = 0, snap_parity = 0, fw_m = 0;
    bool awaiting_commit = false, snap_valid = false, trunk_ok = false;
    int64_t snap_pos = 0, fw_pos = 0, fw_row0 = 0;
    backend::MtpTrack mtp{.nb = 1};
    uint8_t* snap = nullptr;
    void *ops = nullptr, *mops = nullptr, *mkops = nullptr;  // programs with the addresses of the slot
};

// Pointer j of op `op` in the program of several sequences points at a per-token buffer: forward_multi sets it to
// the first row of slot `seq` (base + sb0 * row)
struct XPatch {
    int op, j, seq;
    uint8_t* base;
    int64_t row;
};

// The program of several sequences made from a program of one: steps with per-slot state run once per slot
struct SeqExpand {
    const std::vector<MegaOp>& ops;  // source program
    const std::vector<std::string>& labels;
    const std::set<int>& per_slot;  // kinds of the steps that run once per slot
    std::vector<MegaOp>& out;
    std::vector<std::string>& out_labels;
    std::vector<XPatch>& patches;
    float* ppart = nullptr;  // MTP: partial results of the pick, one set per slot
};

}  // namespace tona::gpu
