// State of one sequence (slot) on the GPU and its copies. The recurrent part (DeltaNet state and conv window, PLE ring)
// exists three times: live, a backup taken before every pass of more than one token (commit(m < nb) restores it and
// replays the DeltaNet recurrence of the m tokens from their kept inputs), and the save point (save_state /
// load_state). KV cache and indexer keys exist once:
// rows behind the position are never read, so going back needs no copy of them.
#pragma once
#include <vector>

#include "backend/backend.h"
#include "backend/metal/device.h"
#include "backend/snap_layout.h"
#include "model/qwen4exp.h"

namespace tona::metal {

// The recurrent part; start is the oldest entry of the PLE ring
struct Recurrent {
    std::vector<Buf> dn_state, dn_win;  // per DeltaNet layer
    Buf ple_ring;                       // [(conv - 1) * ngram][streams * d]
    uint32_t start = 0;
};

// Inputs of the DeltaNet recurrence of the last decode pass, per DeltaNet layer ([BMAX][ch], [BMAX][v heads] twice):
// what commit(m < nb) replays, so the projections and the rest of the pass need not run again
struct DnKeep {
    std::vector<Buf> qkv, alpha, beta;
};

struct SeqState {
    Recurrent live, backup, saved;
    DnKeep keep;
    std::vector<Buf> kv_k, kv_v;         // per QSA layer [max_ctx][kv heads][head dim]
    std::vector<Buf> idx_raw, idx_pool;  // per QSA layer [max_ctx][idx dim], [max_ctx / block + 1][idx dim]
};

// Bytes of the layout, for the sizes of HostSnap and the checks of snap_fits
struct SeqLayout {
    uint64_t recurrent = 0;    // all buffers of one Recurrent (without start)
    uint64_t kv = 0, raw = 0;  // bytes of one position in one QSA layer: K (and V), raw indexer key
    uint64_t pool = 0;         // one pooled block in one QSA layer
    uint32_t n_qsa = 0, idx_block = 0;
    uint64_t kv_row() const { return n_qsa * (2 * kv + raw); }  // one position over all QSA layers
    uint64_t pool_rows(int64_t pos) const { return backend::snap_pool_rows(pos, idx_block); }
};

// extra_qsa: QSA buffers behind those of the layers (the MTP head's), never stashed
SeqState make_seq(Device& dev, const Hparams& h, int64_t max_ctx, int extra_qsa = 0);
SeqLayout seq_layout(const Hparams& h);
void clear(SeqState& s);

// GPU copy of the recurrent part (blit), encoded into cb
void copy(id<MTLCommandBuffer> cb, const Recurrent& from, Recurrent& to);
// commit(m < nb) of a pass of nb tokens: the DeltaNet state and conv windows from the backup, and the PLE ring slots
// the tokens m .. nb - 1 wrote; start as after m tokens
void restore_after(id<MTLCommandBuffer> cb, SeqState& s, int m, int nb, uint32_t hist);

// The save point and the first pos rows of KV, raw keys and pool into RAM, and back (the caller then loads it)
void stash(const SeqState& s, const SeqLayout& l, int64_t pos, backend::HostSnap& out);
void unstash(SeqState& s, const SeqLayout& l, const backend::HostSnap& in);
bool snap_fits(const SeqLayout& l, const backend::HostSnap& in, int64_t max_ctx);

}  // namespace tona::metal
