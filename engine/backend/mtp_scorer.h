// Low-rank scorer of the LM head for the dynamic rows of the MTP draft head (host side, no GPU): basis B = the top
// right singular vectors of the head W (subspace iteration on W^T W, as in vLLM 59740 and llama.cpp 30380), scorer
// row i = W_i B as int8 with a scale per row. Built once per model and kept in a file next to it.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "model/gguf.h"

namespace tona::backend {

struct MtpScorer {
    int rank = 0;
    std::vector<float> basis_t;  // B^T [rank][d]
    std::vector<float> scale;    // per token
    std::vector<int8_t> rows;    // [vocab][rank]; row i * scale[i] approximates W_i B
};

// Identity of the head tensor: FNV-1a 64 over its type, byte size and the raw bytes of the rows 0, the last one and
// every 64th (about 1.6 % of the head: cheap and deterministic). read_row(row, dst, n) gives n bytes of a row.
using RowReader = std::function<void(size_t row, uint8_t* dst, size_t n)>;
uint64_t head_fingerprint(uint32_t type, uint64_t bytes, size_t rows, const RowReader& read_row);

// File `path` if it holds a scorer of this rank for a head of d x vocab with this fingerprint (format version, sizes
// and the file length checked); rank 0: missing, truncated or of another head or format
MtpScorer read_mtp_scorer(const std::string& path, int rank, int d, size_t vocab, uint64_t fingerprint);
// Written to a temporary file and renamed (the temporary file is removed if that fails); false if the folder is not
// writable
bool write_mtp_scorer(const std::string& path, const MtpScorer& s, int d, size_t vocab, uint64_t fingerprint);
// From the output head of the model (any type the CPU dequantizes). Takes about 6 s for the Qwen3.8 head.
MtpScorer make_mtp_scorer(const Gguf& g, const TensorInfo& head, int d, size_t vocab, int rank);

// The scorer of the model: from `file` (TONA_MTP_SCORER_FILE) or else from <model>.mtp-scorer-r<rank> or, where the
// model folder is not writable, from $XDG_CACHE_HOME/tona (default ~/.cache/tona) under a name with the fingerprint. A
// file of another head is replaced after a log line; the scorer is made when no file fits and kept where it can be.
MtpScorer load_or_make_mtp_scorer(const Gguf& g, const TensorInfo& head, int rank, const std::string& model_path,
                                  const char* file);

}  // namespace tona::backend
