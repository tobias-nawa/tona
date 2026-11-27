// Host state of the dynamic rows of the MTP draft head (TONA_MTP_SCORER, device code in mega_op_mtp_dyn.h): which
// tokens the scorer must not offer (they already have a row in the truncated head), and the selected tokens for the
// sampling path.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

namespace tona::gpu {

struct MtpDyn {
    bool on = false;
    int rank = 0;
    int per_part = 0;               // exact rows per partition
    size_t rows = 0;                // dynamic rows in total (partitions x per_part)
    size_t first = 0;               // row of the first dynamic row in the logits and the token map
    std::vector<int32_t> cand_pos;  // token -> candidate row in tile order (-1: not a candidate)
    uint8_t* excl_dev = nullptr;    // per candidate row: 1 = has a row in the truncated head already
    uint8_t* excl_pin = nullptr;    // pinned mirror, so the copies of single bytes need no staging
    size_t excl_n = 0;
    std::vector<int32_t> pending;  // candidate rows changed since the last flush
    int32_t* tok_pin = nullptr;    // pinned: the selected tokens of the last pass (only with sampling)
    void *tiles = nullptr, *scale = nullptr, *ids = nullptr, *basis = nullptr;  // scorer on the device
    uint8_t* dynw = nullptr;  // exact rows of the selected tokens, in the format of the target's head

    // A token got (v = 1) or lost (v = 0) its row in the truncated head
    void note(int32_t tok, uint8_t v) {
        if (!on || tok < 0 || cand_pos[tok] < 0) return;
        excl_pin[cand_pos[tok]] = v;
        pending.push_back(cand_pos[tok]);
    }
    // To the device, in the stream of the pass: single bytes, or everything after a long prompt
    void flush();
    // Sampling: tokens of the dynamic rows of the pass to the host (queue before the sync, adopt after it)
    void queue_tokens(const int32_t* map_dev, bool sampling) const;
    void adopt_tokens(std::vector<int32_t>& vocab, bool sampling) const;
};

}  // namespace tona::gpu
