// Dynamic rows of the MTP draft head (TONA_MTP_SCORER = rank of the scorer, device code in mega_op_mtp_dyn.h): every
// token outside the static list is scored with a low-rank copy of the LM head, the best rows of each partition get
// exact logits. The scorer is made from the head at the first start and kept next to the model (backend/mtp_scorer.h,
// or the file TONA_MTP_SCORER_FILE).
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>

#include "backend/env.h"
#include "backend/mtp_scorer.h"
#include "gpu/arch/qwen4exp/program.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

namespace {

constexpr int kParts = 32;  // MTP_PARTS in mega_op_mtp_dyn.h

// Tiles of 32 candidate rows: 16 int8 values of one row per 16 bytes, the 32 rows of the tile next to each other
void tile_rows(const backend::MtpScorer& f, const std::vector<int32_t>& cand, size_t ncand_pad,
               std::vector<uint8_t>& tiles, std::vector<float>& scale, std::vector<int32_t>& ids) {
    const size_t r = (size_t)f.rank, r16 = r / 16;
    tiles.assign(ncand_pad * r, 0), scale.assign(ncand_pad, 0.0f), ids.assign(ncand_pad, -1);
    for (size_t i = 0; i < cand.size(); i++) {
        const size_t tile = i / 32, lane = i % 32;
        ids[i] = cand[i], scale[i] = f.scale[(size_t)cand[i]];
        for (size_t kk = 0; kk < r16; kk++)
            std::memcpy(&tiles[((tile * r16 + kk) * 32 + lane) * 16], &f.rows[(size_t)cand[i] * r + kk * 16], 16);
    }
}

}  // namespace

// Rank of the scorer (TONA_MTP_SCORER); 0: no dynamic rows, also when the rank is not usable or the engine runs several
// sequences (the program of several sequences runs the steps of the head once per pass, not per slot)
int Qwen4ExpProgram::mtp_dyn_rank() const {
    const int rank = (int)env::integer("TONA_MTP_SCORER", 128);
    return rank % 16 == 0 && rank <= 256 && g_.n_seqs_ <= 1 ? std::max(rank, 0) : 0;
}

// Scorer, candidate tiles and buffers of the dynamic rows; false (with the reason on stderr): the head stays as it is
bool Qwen4ExpProgram::mtp_dyn_setup() {
    const int rank = mtp_dyn_rank();
    if (rank <= 0) return false;
    const std::string path = m_.gguf().file_path(0);
    auto refuse = [&](const char* why) {
        std::fprintf(stderr, "MTP dynamic rows off: %s (%s)\n", why, path.c_str());
        return false;
    };
    const backend::MtpScorer f =
        backend::load_or_make_mtp_scorer(m_.gguf(), *m_.output(), rank, path, env::text("TONA_MTP_SCORER_FILE"));
    // Candidates: the ids below TONA_MTP_SCORER_IDS that the list does not have. The low ids are the early BPE merges,
    // the frequent tokens: on real head inputs the best 512 of those 59,553 candidates held the same share of the
    // full-head argmax as the best 512 of all 206,819 (99.72 % against 99.68 %,
    // research/measurements/2026-11-12-mtp-dynamic-head.md) at 29 % of the bytes of the scorer
    const size_t id_limit = std::min<size_t>(h_.vocab, (size_t)env::integer("TONA_MTP_SCORER_IDS", 98304));
    std::vector<int32_t> cand, pos(h_.vocab, -1);
    for (size_t t = 0; t < id_limit; t++)
        if (!g_.mtp_vocab_.in_head(t)) pos[t] = (int32_t)cand.size(), cand.push_back((int32_t)t);
    const size_t ncand_pad = (cand.size() + 31) / 32 * 32;
    if (ncand_pad / 32 > (size_t)kParts * MWAVES * 16) return refuse("too many candidates");
    MtpDyn& D = g_.mtp_dyn_;
    D.rank = rank, D.cand_pos = std::move(pos), D.excl_n = ncand_pad;
    D.per_part = (int)((env::integer("TONA_MTP_SCORER_ROWS", 1024) + kParts - 1) / kParts);
    D.rows = (size_t)D.per_part * kParts;
    std::vector<uint8_t> tiles;
    std::vector<float> scale;
    std::vector<int32_t> ids;
    tile_rows(f, cand, ncand_pad, tiles, scale, ids);
    D.tiles = pb_.dev_buf(tiles.data(), tiles.size());
    D.scale = pb_.dev_buf(scale.data(), scale.size() * 4);
    D.ids = pb_.dev_buf(ids.data(), ids.size() * 4);
    D.basis = pb_.dev_buf(f.basis_t.data(), f.basis_t.size() * 4);
    D.excl_dev = static_cast<uint8_t*>(pb_.dev_buf(nullptr, ncand_pad));
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&D.excl_pin), ncand_pad, hipHostMallocDefault));
    std::memset(D.excl_pin, 0, ncand_pad);
    g_.pin_allocs_.push_back(D.excl_pin);
    return true;
}

// Steps before the matvec of the head: x projected, then scores, selection and the copy of the rows into D.dynw. The
// static and context rows are in mtp_vocab_ already; the dynamic ones follow them.
void Qwen4ExpProgram::mtp_dyn_ops(int64_t rb) {
    MtpDyn& D = g_.mtp_dyn_;
    D.first = g_.mtp_vocab_.size();
    if (D.first + D.rows > h_.vocab)
        throw std::runtime_error("gpu::Mega: static, context and dynamic rows exceed the vocabulary");
    g_.mtp_vocab_.append_free(D.rows);
    mtp_upload_map();
    D.dynw = static_cast<uint8_t*>(pb_.dev_buf(nullptr, D.rows * (size_t)rb));
    float* pvec = g_.alloc(D.rank);
    int32_t* slots = g_.mtp_map_dev_ + D.first;
    pb_.op(OP_MTP_PROJ, {D.basis, b_.mixed, pvec}, {D.rank, dm_.d, (int)g_.stride_of(b_.mixed)});
    pb_.op(OP_MTP_DYN, {D.tiles, D.scale, D.ids, D.excl_dev, pvec, slots, const_cast<void*>(g_.head_w_), D.dynw},
           {(int)(D.excl_n / 32), D.rank, 0, D.per_part, (int)rb});
}

}  // namespace tona::gpu
