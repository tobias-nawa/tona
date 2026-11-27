// Host parts of the Metal pass without a GPU: RoPE table, n-gram rows and indexer cells (backend/metal/host_math.h)
#include <cmath>
#include <cstdio>
#include <exception>

#include "backend/metal/host_math.h"
#include "backend/metal/stop_wait.h"

static int g_fail = 0;
#define EXPECT(c)                                                                       \
    do {                                                                                \
        if (!(c)) {                                                                     \
            fprintf(stderr, "%s:%d: expectation failed: %s\n", __FILE__, __LINE__, #c); \
            g_fail++;                                                                   \
        }                                                                               \
    } while (0)

using namespace tona::metal;
using tona::qsa_cells;
using tona::QsaCells;

namespace {

void rope() {
    const auto c0 = rope_table(8, 0, 10000.0f);
    for (int i = 0; i < 4; i++) EXPECT(c0[i] == 1.0f && c0[4 + i] == 0.0f);
    // pair 0 turns by pos radians, pair 1 by pos * base^(-2/8)
    const auto c3 = rope_table(8, 3, 10000.0f);
    EXPECT(std::fabs(c3[0] - std::cos(3.0f)) < 1e-6f && std::fabs(c3[4] - std::sin(3.0f)) < 1e-6f);
    EXPECT(std::fabs(c3[1] - (float)std::cos(3.0 * std::pow(10000.0, -0.25))) < 1e-6f);
}

void cells() {
    QsaCells c = qsa_cells(2, 4, 64);  // three tokens, no block complete
    EXPECT(c.n_pool == 0 && c.top_blocks == 0 && c.tail0 == 0 && c.n_tail == 3 && c.count() == 3);
    c = qsa_cells(3, 4, 64);  // block 0 complete, it is the only one
    EXPECT(c.n_pool == 1 && c.top_blocks == 1 && c.n_tail == 0 && c.count() == 4);
    c = qsa_cells(100, 4, 64);  // 25 blocks, 16 taken, one token in progress
    EXPECT(c.n_pool == 25 && c.top_blocks == 16 && c.tail0 == 100 && c.n_tail == 1 && c.count() == 65);
}

void ngrams() {
    tona::Hparams h;
    h.ple_ngram = 3, h.ple_heads = 4, h.ple_eos = 7;
    h.ple_mult = {3, 5, 11};
    h.ple_head_vocab = {13, 17, 19, 23};
    h.ple_head_offset = {0, 13, 30, 49};
    // bigram heads 0, 1 hash t0*3 ^ t1*5; trigram heads 2, 3 also ^ t2*11
    auto rows = ple_rows(h, {4, 9, 2});
    const uint64_t bi = 2ull * 3 ^ 9ull * 5, tri = bi ^ 4ull * 11;
    EXPECT(rows[0] == (int)(bi % 13) && rows[1] == (int)(bi % 17 + 13));
    EXPECT(rows[2] == (int)(tri % 19 + 30) && rows[3] == (int)(tri % 23 + 49));
    // missing predecessors become EOS; an EOS cuts everything before it
    rows = ple_rows(h, {2});
    EXPECT(rows[2] == (int)((2 * 3 ^ 7 * 5 ^ 7 * 11) % 19 + 30));
    rows = ple_rows(h, {4, 7, 2});
    EXPECT(rows[3] == (int)((2 * 3 ^ 7 * 5 ^ 7 * 11) % 23 + 49));
}

// Tensors at fake addresses: windows hold whole tensors, start on a page and stay within the limit
void buffer_windows() {
    alignas(4096) static uint8_t mem[64 * 1024];
    std::vector<tona::TensorInfo> t(4);
    const uint64_t at[4] = {100, 5000, 9000, 30000}, len[4] = {4000, 3000, 12000, 2000};
    std::vector<const tona::TensorInfo*> ts;
    for (int i = 0; i < 4; i++) t[i].data = mem + at[i], t[i].bytes = len[i], ts.push_back(&t[i]);
    const auto w = windows(ts, 16384, 4096);
    // tensors 0 and 1 end at 8000 (2 pages); tensor 2 would reach 21000 > 16384 and opens a window at page 8192;
    // tensor 3 ends 23808 bytes after that page and opens the third
    EXPECT(w.size() == 3);
    EXPECT(w[0].base == mem && w[0].bytes == 8192);
    EXPECT(w[1].base == mem + 8192 && w[1].bytes == 16384);
    EXPECT(w[2].base == mem + 28672 && w[2].bytes == 4096);
    t[3].bytes = 20000;
    bool threw = false;
    try {
        windows(ts, 16384, 4096);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT(threw);
}

// A gap limit of one page: tensors 0 to 2 start within a page of the window's end (24576 after tensor 2), tensor 3 at
// 30000 does not and opens a window of its own
void buffer_windows_gap() {
    alignas(4096) static uint8_t mem[64 * 1024];
    std::vector<tona::TensorInfo> t(4);
    const uint64_t at[4] = {100, 5000, 9000, 30000}, len[4] = {4000, 3000, 12000, 2000};
    std::vector<const tona::TensorInfo*> ts;
    for (int i = 0; i < 4; i++) t[i].data = mem + at[i], t[i].bytes = len[i], ts.push_back(&t[i]);
    const auto g = windows(ts, 65536, 4096, 4096);
    EXPECT(g.size() == 2);
    EXPECT(g[0].base == mem && g[0].bytes == 24576);
    EXPECT(g[1].base == mem + 28672 && g[1].bytes == 4096);
}

// The smallest of three limits wins; a limit already exceeded gives 0, not a negative budget
// The stop wait gives up on the GPU's error or a buffer done without the stop at once; otherwise only after 10 s of
// wall time in which the waiting thread ran 10 s: a held or paging process (little CPU time) waits on
void stop_waits() {
    EXPECT(stop_verdict({0.01, 0.01, true, false}) == StopVerdict::gpu_failed);
    EXPECT(stop_verdict({0.01, 0.01, false, true}) == StopVerdict::missed);
    EXPECT(stop_verdict({4.5, 4.5, false, false}) == StopVerdict::wait);  // the old deadline failed here
    EXPECT(stop_verdict({30, 2, false, false}) == StopVerdict::wait);     // held: wall time without CPU time
    EXPECT(stop_verdict({10, 10, false, false}) == StopVerdict::hung);
    EXPECT(stop_verdict({60, 9.9, false, false}) == StopVerdict::wait);
    using tona::metal::SleepRule;
    using tona::metal::StopPace;
    StopPace p;
    EXPECT(pace_sleep(p, SleepRule{0.5, -1, 0}) == 0);                         // nothing seen yet
    for (int i = 0; i < 50; i++) pace_update(p, i % 2 ? 0.009 : 0.011, true);  // mean 10 ms, deviation 1 ms
    EXPECT(std::fabs(p.mean - 0.010) < 0.0005);
    EXPECT(std::fabs(pace_sleep(p, SleepRule{0.5, -1, 0}) - 0.005) < 0.0003);  // half the mean
    const double two_sigma = pace_sleep(p, SleepRule{0.9, 2, 0});
    EXPECT(two_sigma > 0.0075 && two_sigma < 0.0090);     // the mean less about 2 ms, under 0.9 of it
    EXPECT(pace_sleep(p, SleepRule{0.9, 2, 0.02}) == 0);  // shorter than the floor: no sleep
    pace_update(p, 0, false);
    EXPECT(std::fabs(p.mean - 0.008) < 0.0005);  // flag there on waking: the mean drops
}

void arena_budget() {
    MemNow m;
    m.ram = 34, m.taken = 8, m.mapped = 5, m.keep = 7;  // RAM: 14 left
    m.working_set = 27, m.gpu = 10, m.slack = 1;        // working set: 16
    m.max_buffer = 20;
    EXPECT(arena_auto_budget(m) == 14);
    m.taken = 3;  // RAM 19: the working set decides
    EXPECT(arena_auto_budget(m) == 16);
    m.gpu = 0;  // RAM 19, working set 26: one buffer at most
    m.max_buffer = 18;
    EXPECT(arena_auto_budget(m) == 18);
    m.taken = 40;
    EXPECT(arena_auto_budget(m) == 0);
    EXPECT(arena_ceiling(m) == 26);  // RAM and the buffer limit do not bound it, the working set does
    m.gpu = 30;
    EXPECT(arena_ceiling(m) == 0);
}

}  // namespace

int main() {
    try {
        rope();
        cells();
        ngrams();
        buffer_windows();
        buffer_windows_gap();
        arena_budget();
        stop_waits();
    } catch (const std::exception& e) {
        printf("metal host: %s\n", e.what());
        return 1;
    }
    if (g_fail) return 1;
    printf("metal host: ok\n");
    return 0;
}
