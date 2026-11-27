// gpu/kv_place.h: how much of the KV cache sits in VRAM, by position, for a budget
#include "gpu/kv_place.h"
#include "tests/test_support.h"

int main() {
    using namespace tona::gpu;
    constexpr size_t row = 1152;  // Ornith: 2 KV heads of 256, K and V int8 plus the scales
    constexpr int64_t ctx = 262144;
    const size_t buf = (size_t)ctx * row;  // 0.30 GB per layer

    // everything fits: no split, nothing in RAM
    KvSplit p = KvSplit::plan(10, 1, ctx, row, 11 * buf);
    EXPECT(p.hot == ctx && !p.mtp_host && p.host_frac == 0.0);

    // 2.4 GB at ctx 262144 with ten layers and the MTP layer: the MTP layer whole, the rest by position
    p = KvSplit::plan(10, 1, ctx, row, 2400000000ull);
    EXPECT(!p.mtp_host);
    EXPECT(p.hot % 256 == 0 && p.hot > 160000 && p.hot < 190000);  // about (2.4 - 0.30) GB / 10 / 1152 B
    EXPECT((size_t)p.hot * row * 10 + buf <= 2400000000ull);       // never over the budget
    EXPECT(p.host_frac > 0.1 && p.host_frac < 0.4);
    EXPECT(p.cells_in_vram(3, 10, ctx) == p.hot && p.cells_in_vram(10, 10, ctx) == ctx);

    // a budget below the MTP layer: it goes to RAM, the main layers get the whole budget
    p = KvSplit::plan(10, 1, ctx, row, buf / 2);
    EXPECT(p.mtp_host && p.hot > 0 && p.cells_in_vram(10, 10, ctx) == 0);

    // too few cells to be worth it: all in RAM
    p = KvSplit::plan(10, 0, ctx, row, (size_t)10 * 1000 * row);
    EXPECT(p.hot == 0 && p.host_frac == 1.0);
    EXPECT(KvSplit::plan(10, 0, ctx, row, 0).hot == 0);

    // no attention layers: nothing to place
    EXPECT(KvSplit::plan(0, 0, ctx, row, 1).host_frac == 0.0);

    // not planned (the Coder): the model follows Mega::kv_host_
    EXPECT(!KvSplit{}.planned());

    // the default budget: 128,000 cells of ten layers plus the MTP layer, capped at 13.5 % of the free VRAM after the
    // reserve
    const int64_t G = 1000000000, reserve = 1073741824;
    const size_t want = ((size_t)10 * 128000 + (size_t)ctx) * row;
    EXPECT(KvSplit::default_budget(10, 1, ctx, row, 15 * G, reserve) == want);  // 1.78 GB, below 13.5 % of 13.9 GB
    EXPECT(KvSplit::default_budget(10, 1, ctx, row, 8 * G, reserve) < want);    // a small card: the cap
    EXPECT(KvSplit::default_budget(10, 1, ctx, row, 8 * G, reserve) == (size_t)(0.135 * (8 * G - reserve)));
    EXPECT(KvSplit::default_budget(10, 1, ctx, row, 500000000, reserve) == 0);
    EXPECT(KvSplit::default_budget(10, 1, 70000, row, 15 * G, reserve) == (size_t)11 * 70000 * row);  // all fits
    const KvSplit q = KvSplit::plan(10, 1, ctx, row, want);
    EXPECT(q.hot >= 127744 && q.hot <= 128000 && !q.mtp_host);
    return tona::test::finish("kv_place");
}
