// Bookkeeping of the MTP head of one slot, the part every backend shares: which entries are computed, where the
// target's h of a position can be read, in which chunks the head catches up, and what happens to the integers on
// save, load and reset. The backend executes: it reads h from its own memory and runs the chunks.
//
// Entry p takes the embedding of token p and the target's h at p - 1 and proposes token p + 1.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace tona::backend {

// The rows of the last target pass that are still in the backend's buffer: positions [pos, pos + n)
struct PassWindow {
    bool ok = false;
    int64_t pos = 0, n = 0;
};

// Where the target's h at position q is
enum class HSource {
    None,     // q < 0: no h, the entry starts from zeros by design
    Pass,     // row `row` of the last pass
    Saved,    // the row kept for the start of the next fill
    Unknown,  // not kept: the entry is computed with h = 0 and counted as stale
};

struct HLoc {
    HSource src = HSource::Unknown;
    int64_t row = 0;  // with Pass: q - window.pos
};

// What catch_up works on: the slot's tokens, the position the entries run up to, the token after it (< 0: none)
struct CatchUpIn {
    const std::vector<int32_t>& tokens;
    int64_t end_pos;
    int32_t next;
};

// Where it puts the inputs of a chunk: at most bmax tokens and their h pointers
struct CatchUpBuf {
    int bmax;
    int32_t* toks;
    const float** hs;
};

struct MtpTrack {
    int64_t valid = 0;       // entries before this are computed
    int64_t pos = 0;         // first position of the last MTP pass ...
    int nb = 0;              // ... and its width; 0: no pass yet, a step has nothing to chain to
    int64_t hlast_pos = -1;  // position of the saved target h (-1: none)
    int32_t tent = -1;       // token an MTP fill computed ahead of the target; -1: none
    size_t scan = 0;         // context tokens offered to the dynamic rows so far (MtpVocabMap::sync)
    int64_t snap_valid = 0, snap_hlast_pos = -1;  // the same at the restore point

    HLoc h_of(int64_t q, const PassWindow& w) const {
        if (q < 0) return {HSource::None, 0};
        if (w.ok && q >= w.pos && q < w.pos + w.n) return {HSource::Pass, q - w.pos};
        if (q == hlast_pos) return {HSource::Saved, 0};
        return {HSource::Unknown, 0};
    }

    // Position of the next chained step
    int64_t next_pos() const { return pos + nb; }

    // A target pass starting at p0 with first token t0: the entry the head computed ahead is the target's token
    void on_target_pass(int64_t p0, int32_t t0) {
        if (tent >= 0 && valid == p0 && t0 == tent) valid = p0 + 1;
        tent = -1;
    }

    // Entries from min(valid, pos) up to pos (+ 1 with a next token), in chunks of at most bmax. For each chunk
    // buf.toks[b] is tokens[p + b] or next and buf.hs[b] = h_of_q(p + b - 1); run(p, nb, last) with last: this chunk
    // holds the entry at pos and the head must run.
    template <class HOf, class Run>
    void catch_up(const CatchUpIn& in, const CatchUpBuf& buf, HOf&& h_of_q, Run&& run) const {
        const int64_t end = in.end_pos + (in.next >= 0);
        for (int64_t p = std::min(valid, in.end_pos); p < end;) {
            const int nb = (int)std::min<int64_t>(buf.bmax, end - p);
            for (int b = 0; b < nb; b++) {
                buf.toks[b] = p + b < in.end_pos ? in.tokens[p + b] : in.next;
                buf.hs[b] = h_of_q(p + b - 1);
            }
            run(p, nb, p + nb == end && in.next >= 0);
            p += nb;
        }
    }

    // After a fill at pos. h_saved: the target's h at pos - 1 is in the saved row now (or q < 0).
    void after_fill(int64_t at_pos, int32_t next, bool h_saved) {
        if (h_saved) hlast_pos = at_pos - 1;
        valid = at_pos, tent = next;
    }

    // After a prefill chunk ended at pos with the saved row holding h at pos - 1
    void after_prefill(int64_t at_pos) { hlast_pos = at_pos - 1, valid = at_pos, tent = -1; }

    void on_save() { snap_valid = valid, snap_hlast_pos = hlast_pos; }
    // Restore point loaded; the position is the one of the snapshot
    void on_load(int64_t at_pos) {
        valid = std::min(snap_valid, at_pos), hlast_pos = snap_hlast_pos, tent = -1, scan = 0;
    }
    // Restore point stashed earlier: made after a prompt, MTP caught up to pos - 1 with h at pos - 1 in the state
    void on_unstash(int64_t at_pos) { snap_valid = at_pos, snap_hlast_pos = at_pos - 1; }
    void on_reset() { valid = 0, hlast_pos = -1, tent = -1, scan = 0; }
};

}  // namespace tona::backend
