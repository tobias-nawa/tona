// The truncated LM head of the MTP head, the part every backend shares: the token list (built in or from a file),
// and the ring of rows that tokens of the context outside the list get on demand. Only the row copy differs per
// platform (a copy inside VRAM, a conversion to IQ4_NL, a memcpy from the mapping), so it is a callback of sync().
#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/env.h"

namespace tona::backend {

enum class MtpList { Wide, Narrow };  // 41,500 ids; 16,000 ids (the scorer ranks the rest, TONA_MTP_SCORER)

// Token ids of the truncated head: the built-in list of scripts/build-mtp-vocab.py or the file `file`
// (TONA_MTP_VOCAB) with one id per line, most frequent first; TONA_MTP_VOCAB_N takes the first N of them. Duplicates
// and ids outside the vocabulary are dropped. `who` names the backend in the error message.
inline std::vector<int32_t> mtp_vocab_list(uint32_t vocab, MtpList which, const char* file, const char* who) {
    static constexpr int32_t wide[] = {
#include "model/mtp_vocab.inc"
    };
    static constexpr int32_t narrow[] = {
#include "model/mtp_vocab_dyn.inc"
    };
    const size_t nmax = (size_t)env::integer("TONA_MTP_VOCAB_N", vocab);
    std::vector<uint8_t> seen(vocab);
    std::vector<int32_t> list;
    auto put = [&](long long id) {
        if (list.size() < nmax && id >= 0 && id < (long long)vocab && !seen[id])
            seen[id] = 1, list.push_back((int32_t)id);
    };
    if (file) {
        std::ifstream in(file);
        for (long long id; in >> id;) put(id);
    } else if (which == MtpList::Narrow) {
        for (int32_t id : narrow) put(id);
    } else {
        for (int32_t id : wide) put(id);
    }
    if (list.empty())
        throw std::runtime_error(std::string(who) + ": no token ids in " + (file ? file : "the built-in list"));
    return list;
}

// Row r -> token of the truncated head: the static rows of the list, then `ctx` rows for tokens of the context (a
// ring, oldest replaced first), then rows a backend appends for its own use (the scorer's rows); -1: free row.
class MtpVocabMap {
public:
    MtpVocabMap() = default;
    MtpVocabMap(std::vector<int32_t> list, size_t ctx_rows, uint32_t vocab)
        : tok_(std::move(list)), in_head_(vocab, 0), n_static_(tok_.size()), n_ctx_(ctx_rows) {
        for (int32_t t : tok_) in_head_[t] = 1;
        tok_.resize(n_static_ + n_ctx_, -1);
    }

    bool empty() const { return tok_.empty(); }
    size_t size() const { return tok_.size(); }
    size_t n_static() const { return n_static_; }
    size_t n_ctx() const { return n_ctx_; }
    int32_t operator[](size_t r) const { return tok_[r]; }
    int32_t token(size_t r) const { return r < tok_.size() ? tok_[r] : -1; }
    bool in_head(size_t t) const { return t < in_head_.size() && in_head_[t]; }
    const int32_t* data() const { return tok_.data(); }
    std::vector<int32_t>& tokens() { return tok_; }  // the backend writes the rows it appended
    auto begin() const { return tok_.begin(); }
    auto end() const { return tok_.end(); }
    // `extra` free rows behind the context rows
    void append_free(size_t extra) { tok_.insert(tok_.end(), extra, -1); }

    // The ring rows that `sync` changed: the first one and how many; at most two contiguous pieces on the device
    struct Taken {
        size_t first = 0, n = 0;
    };

    // Gives every token of `tokens` from `scan` on and `next` (< 0: none) that has no row the next ring row; `scan`
    // is the slot's (reset on load and reset). on_take(ring_index, old_token, new_token) runs after the row is
    // re-assigned, old_token -1 for a row that was free. At most n_ctx rows change per call: the rest of a longer
    // context is skipped (the oldest rows are not replaced within one call).
    template <class OnTake>
    Taken sync(const std::vector<int32_t>& tokens, int32_t next, size_t& scan, OnTake&& on_take) {
        if (n_ctx_ == 0) return {};
        if (scan > tokens.size()) scan = 0;
        Taken out{ring_, 0};
        auto take = [&](int32_t t) {
            if (t < 0 || in_head((size_t)t) || out.n == n_ctx_) return;
            const size_t row = n_static_ + ring_;
            const int32_t old = tok_[row];
            if (old >= 0) in_head_[old] = 0;
            tok_[row] = t, in_head_[t] = 1;
            on_take(ring_, old, t);
            ring_ = (ring_ + 1) % n_ctx_, out.n++;
        };
        for (; scan < tokens.size(); scan++) take(tokens[scan]);
        take(next);
        return out;
    }

private:
    std::vector<int32_t> tok_;
    std::vector<uint8_t> in_head_;  // per token: has a row
    size_t n_static_ = 0, n_ctx_ = 0, ring_ = 0;
};

}  // namespace tona::backend
