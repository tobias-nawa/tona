// Host code every backend shares: PLE rows (model/ple.h), snapshot sizes (backend/snap_layout.h), -log p
// (backend/host_logits.h), the IQ4_NL quantizer (model/quant_iq4.h), the RoPE pair (backend/rope_yarn.h), indexer cells
// and row sizes. The PLE test compares against a literal copy of the code the backends had before the move, so that a
// refactor cannot change a row.
#include <cmath>
#include <cstring>
#include <vector>

#include "backend/host_logits.h"
#include "backend/rope_yarn.h"
#include "backend/snap_layout.h"
#include "model/gguf.h"
#include "model/ple.h"
#include "model/quant.h"
#include "model/quant_iq4.h"
#include "tests/test_support.h"

namespace {

using namespace tona;

Hparams ple_params() {
    Hparams h;
    h.ple_ngram = 3, h.ple_heads = 4, h.ple_eos = 7;
    h.ple_mult = {3, 5, 11};
    h.ple_head_vocab = {13, 17, 19, 23};
    h.ple_head_offset = {0, 13, 30, 49};
    return h;
}

// The reference's old Forward::ple_context and ple_lookup (row part), kept as they were
std::vector<int64_t> old_rows(const Hparams& h, const std::vector<int32_t>& tokens) {
    const int64_t n = h.ple_ngram;
    std::vector<int64_t> ctx(n);
    ctx[0] = tokens.back();
    bool cut = false;
    for (int64_t s = 1; s < n; s++) {
        const int64_t p = (int64_t)tokens.size() - 1 - s;
        const int64_t t = cut || p < 0 ? -1 : tokens[p];
        cut = cut || t < 0 || t == h.ple_eos;
        ctx[s] = cut ? h.ple_eos : t;
    }
    std::vector<int64_t> rows(h.ple_heads);
    const uint32_t per_gram = h.ple_heads / (n - 1);
    for (int64_t g = 2; g <= n; g++) {
        uint64_t mixed = (uint64_t)ctx[0] * (uint64_t)h.ple_mult[0];
        for (int64_t j = 1; j < g; j++) mixed ^= (uint64_t)ctx[j] * (uint64_t)h.ple_mult[j];
        for (uint32_t k = 0; k < per_gram; k++) {
            const uint32_t head = (g - 2) * per_gram + k;
            rows[head] = (int64_t)(mixed % h.ple_head_vocab[head] + h.ple_head_offset[head]);
        }
    }
    return rows;
}

void test_ple_hand_values() {
    const Hparams h = ple_params();
    int64_t rows[4] = {};  // rows_at writes every head; zeroed so a missed head fails the check, not reads garbage
    // bigram heads 0, 1 hash t0*3 ^ t1*5; trigram heads 2, 3 also ^ t2*11
    const int32_t a[] = {4, 9, 2};
    ple::rows_at(h, a, 2, rows);
    const uint64_t bi = 2ull * 3 ^ 9ull * 5, tri = bi ^ 4ull * 11;
    EXPECT(rows[0] == (int64_t)(bi % 13) && rows[1] == (int64_t)(bi % 17 + 13));
    EXPECT(rows[2] == (int64_t)(tri % 19 + 30) && rows[3] == (int64_t)(tri % 23 + 49));
    // a missing predecessor becomes EOS
    const int32_t b[] = {2};
    ple::rows_at(h, b, 0, rows);
    EXPECT(rows[2] == (int64_t)((2 * 3 ^ 7 * 5 ^ 7 * 11) % 19 + 30));
    // an EOS cuts everything before it, the token's own EOS does not
    const int32_t c[] = {4, 7, 2};
    ple::rows_at(h, c, 2, rows);
    EXPECT(rows[3] == (int64_t)((2 * 3 ^ 7 * 5 ^ 7 * 11) % 23 + 49));
    int64_t ctx[ple::MAX_NGRAM] = {};
    const int32_t d[] = {4, 3, 7};
    ple::context(h, d, 2, ctx);
    EXPECT(ctx[0] == 7 && ctx[1] == 3 && ctx[2] == 4);
}

void test_ple_against_old_code() {
    const Hparams h = ple_params();
    uint32_t x = 12345;
    std::vector<int32_t> seq;
    for (int i = 0; i < 400; i++) {
        x = x * 1664525u + 1013904223u;
        seq.push_back((int32_t)((x >> 8) % 12));  // small vocabulary: EOS (7) comes often
        std::vector<int64_t> rows64(h.ple_heads);
        std::vector<int32_t> rows32(h.ple_heads);
        ple::rows_at(h, seq.data(), (int64_t)seq.size() - 1, rows64.data());
        ple::rows_at(h, seq.data(), (int64_t)seq.size() - 1, rows32.data());
        const auto want = old_rows(h, seq);
        bool same = rows64 == want;
        for (uint32_t k = 0; k < h.ple_heads; k++) same &= (int64_t)rows32[k] == want[k];
        EXPECT(same);
    }
}

void test_snap_layout() {
    using namespace tona::backend;
    EXPECT(snap_pool_rows(0, 4) == 1 && snap_pool_rows(3, 4) == 1 && snap_pool_rows(4, 4) == 2);
    EXPECT(snap_pool_bytes(100, 4, 128) == size_t(26) * 128 * sizeof(float));
    EXPECT(snap_total_bytes(10, 20, 30) == 60);
    HostSnap s;
    s.pos = 3, s.tokens = {1, 2, 3};
    EXPECT(snap_pos_ok(s, 3) && !snap_pos_ok(s, 2));
    s.tokens.pop_back();
    EXPECT(!snap_pos_ok(s, 8));
    s.pos = 0, s.tokens.clear();
    EXPECT(!snap_pos_ok(s, 8));
    s.kv.resize(5), s.pool.resize(7);
    EXPECT(snap_kv_pool_ok(s, 5, 7) && !snap_kv_pool_ok(s, 5, 8) && !snap_kv_pool_ok(s, 6, 7));
}

void test_neg_log_p() {
    const float uniform[4] = {1, 1, 1, 1};
    EXPECT(std::fabs(backend::neg_log_p(uniform, 4, 2) - std::log(4.0)) < 1e-12);
    const float lg[3] = {0, 2, 1};
    const double z = std::exp(0.0 - 2) + 1 + std::exp(1.0 - 2);
    EXPECT(std::fabs(backend::neg_log_p(lg, 3, 1) - std::log(z)) < 1e-12);
    EXPECT(backend::neg_log_p(lg, 3, 0) > backend::neg_log_p(lg, 3, 2));
}

void test_iq4() {
    std::vector<float> zero(32, 0.0f), x(64);
    uint8_t out[36] = {};
    iq4::quantize_iq4_nl(zero.data(), out, 32);
    float back[32];
    dequantize_row(20, out, back, 32);
    for (float v : back) EXPECT(v == 0.0f);
    // values on the grid kIq4[j % 16] * 0.5 round-trip exactly
    for (int j = 0; j < 64; j++) x[j] = iq4::kIq4[(j * 5) % 16] * 0.5f;
    std::vector<uint8_t> q(36);
    iq4::quantize_iq4_nl(x.data(), q.data(), 64);
    std::vector<float> y(64);
    dequantize_row(20, q.data(), y.data(), 32), dequantize_row(20, q.data() + 18, y.data() + 32, 32);
    double err = 0;
    for (int j = 0; j < 64; j++) err = std::max(err, (double)std::fabs(x[j] - y[j]));
    EXPECT(err < 1.0);  // the grid step is at least 12 * 0.5 apart; a wrong nibble order would be off by far more
    EXPECT(iq4::nearest_iq4(-200.0f) == 0 && iq4::nearest_iq4(200.0f) == 15 && iq4::nearest_iq4(1.2f) == 8);
}

void test_rope_pair() {
    const rope::YarnTable t = rope::yarn_table(1.0, 1.0, 10000.0);
    auto cs = rope::pair_cos_sin(t, 8, 0, 10000.0f, 2);
    EXPECT(cs.c == 1.0 && cs.s == 0.0);
    cs = rope::pair_cos_sin(t, 8, 3, 10000.0f, 0);
    EXPECT(std::fabs(cs.c - std::cos(3.0)) < 1e-15 && std::fabs(cs.s - std::sin(3.0)) < 1e-15);
    cs = rope::pair_cos_sin(t, 8, 3, 10000.0f, 1);
    EXPECT(std::fabs(cs.c - std::cos(3.0 * std::pow(10000.0, -0.25))) < 1e-15);
}

void test_cells_and_sizes() {
    QsaCells c = qsa_cells(2, 4, 64);  // three tokens, no block complete
    EXPECT(c.n_pool == 0 && c.top_blocks == 0 && c.tail0 == 0 && c.n_tail == 3 && c.count() == 3);
    c = qsa_cells(3, 4, 64);  // block 0 complete, it is the only one
    EXPECT(c.n_pool == 1 && c.top_blocks == 1 && c.n_tail == 0 && c.count() == 4);
    c = qsa_cells(100, 4, 64);  // 25 blocks, 16 taken, one token in progress
    EXPECT(c.n_pool == 25 && c.top_blocks == 16 && c.tail0 == 100 && c.n_tail == 1 && c.count() == 65);

    EXPECT(type_row_bytes(0, 32) == 128);
    EXPECT(type_row_bytes(8, 64) == 68 && type_row_bytes(20, 64) == 36 && type_row_bytes(12, 512) == 288);
    TensorInfo t;
    t.dims = {64, 8, 4}, t.bytes = 4000;
    EXPECT(bytes_per_expert(&t) == 1000);
}

}  // namespace

int main() {
    return tona::test::run("test_shared_host", [] {
        test_ple_hand_values();
        test_ple_against_old_code();
        test_snap_layout();
        test_neg_log_p();
        test_iq4();
        test_rope_pair();
        test_cells_and_sizes();
    });
}
