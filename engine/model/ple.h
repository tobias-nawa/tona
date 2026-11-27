// PLE n-gram table lookup, host side: the context of a token and the table rows its hashed n-grams select. The one
// definition for every backend; only what is done with a row (dequantize, gather, copy from the mapping) differs.
// Pure integer code: the rows are bit-identical wherever it is called from.
#pragma once

#include <cstdint>

#include "model/qwen4exp.h"

namespace tona::ple {

constexpr int MAX_NGRAM = 8;  // context buffers live on the stack

// Context of seq[i] with seq[0..i] as history: ctx[0] = seq[i], ctx[s] = seq[i - s]. A missing predecessor and
// everything before an EOS become EOS; the token's own EOS does not cut (llm_graph_input_qwen4exp_ple::set_input).
inline void context(const Hparams& h, const int32_t* seq, int64_t i, int64_t* ctx) {
    ctx[0] = seq[i];
    bool cut = false;
    for (int64_t s = 1; s < (int64_t)h.ple_ngram; s++) {
        const int64_t p = i - s;
        const int64_t t = cut || p < 0 ? -1 : seq[p];
        cut = cut || t < 0 || t == (int64_t)h.ple_eos;
        ctx[s] = cut ? (int64_t)h.ple_eos : t;
    }
}

// Table row of each head for a context: n-gram g (2..ngram) is hashed over ctx[0..g), the heads of that gram share
// the hash and differ in modulus and offset. out[ple_heads].
template <class Row>
inline void rows_of(const Hparams& h, const int64_t* ctx, Row* out) {
    const int64_t n = h.ple_ngram;
    const uint32_t per_gram = h.ple_heads / (uint32_t)(n - 1);
    for (int64_t g = 2; g <= n; g++) {
        uint64_t mixed = (uint64_t)ctx[0] * (uint64_t)h.ple_mult[0];
        for (int64_t j = 1; j < g; j++) mixed ^= (uint64_t)ctx[j] * (uint64_t)h.ple_mult[j];
        for (uint32_t k = 0; k < per_gram; k++) {
            const uint32_t head = (uint32_t)(g - 2) * per_gram + k;
            out[head] = (Row)(mixed % h.ple_head_vocab[head] + h.ple_head_offset[head]);
        }
    }
}

// Rows of seq[i] in one call
template <class Row>
inline void rows_at(const Hparams& h, const int32_t* seq, int64_t i, Row* out) {
    int64_t ctx[MAX_NGRAM];
    context(h, seq, i, ctx);
    rows_of(h, ctx, out);
}

}  // namespace tona::ple
