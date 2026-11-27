// The truncated LM head of the MTP head: the rows of a fixed token list (scripts/build-mtp-vocab.py, 41,489 ids)
// copied from the target's LM head, then TONA_MTP_DYN (8192) rows for context tokens outside the list, filled round
// robin. List, ring and map are shared with the reference (backend/mtp_vocab.h); this class copies the rows from the
// GGUF mapping into a Metal buffer. TONA_MTP_VOCAB=FILE (one id per line) and TONA_MTP_VOCAB_N limit the list.
#pragma once
#include <cstdint>
#include <vector>

#include "backend/metal/device.h"
#include "backend/metal/weights.h"
#include "backend/mtp_vocab.h"
#include "model/qwen4exp.h"

namespace tona::metal {

class MtpVocab {
public:
    MtpVocab(Device& dev, const Qwen4ExpModel& m);
    // Rows for the tokens of the context from scan on (the slot's: MtpTrack::scan, reset on reset and load_state) and
    // for next (< 0: none) that have none
    void sync(const std::vector<int32_t>& tokens, int32_t next, size_t& scan);
    const Mat& head() const { return head_; }
    const Buf& row_tokens() const { return rows_; }  // the token of each row on the device (-1: free), for the pick
    size_t rows() const { return map_.size(); }
    int32_t token(size_t r) const { return map_.token(r); }
    int64_t loaded() const { return loaded_; }  // rows loaded on demand so far

private:
    const TensorInfo& out_;
    Mat head_;
    Buf rows_;
    backend::MtpVocabMap map_;  // token of each row, -1: free
    int64_t loaded_ = 0;
};

}  // namespace tona::metal
