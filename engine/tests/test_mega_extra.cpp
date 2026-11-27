// Extra expert slots in the prefill workspace (gpu/mega_extra.h): layout on the slot grid inside the workspace,
// rank-wise interleaving, giving the slots up for a prefill and getting them back, and the checks that reject a layout
// that does not fit.
#include <algorithm>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <vector>

#include "gpu/mega_extra.h"
#include "tests/test_support.h"

using namespace tona::gpu;

namespace {

// 6 layers with different slot sizes (as the quantization types differ), 20 base slots each, the workspace behind them
ExtraLayout layout(int ranks = 2, size_t arena_mb = 40) {
    ExtraLayout in;
    const size_t sizes[] = {2000000, 2100000, 1900000, 2200000, 2000000, 2050000};
    size_t off = 0;
    for (size_t sb : sizes) {
        in.slot_bytes.push_back(sb);
        in.slot_off.push_back(off);
        off += (sb * 20 + 4096 + 255) / 256 * 256;
    }
    in.arena_start = off, in.arena_bytes = arena_mb << 20;
    in.base_slots = 20, in.max_slots = 256, in.ranks = ranks, in.read_pad = 64;
    return in;
}

std::vector<LayerCache> caches(const ExtraLayout& in) {
    std::vector<LayerCache> cs(in.slot_bytes.size());
    for (LayerCache& C : cs) {
        C.n_slots = in.base_slots;
        for (int s = in.base_slots - 1; s >= 0; s--) C.free_slots.push_back(s);
    }
    return cs;
}

int total(const std::vector<std::vector<ExtraBlock>>& b) {
    int n = 0;
    for (const auto& l : b)
        for (const ExtraBlock& x : l) n += x.count;
    return n;
}

int per_layer(const std::vector<ExtraBlock>& l) {
    int n = 0;
    for (const ExtraBlock& b : l) n += b.count;
    return n;
}

std::vector<int> per_counts(const std::vector<std::vector<ExtraBlock>>& b) {
    std::vector<int> n(b.size());
    std::transform(b.begin(), b.end(), n.begin(), [](const auto& l) { return per_layer(l); });
    return n;
}

void test_layout_fills_the_workspace() {
    const ExtraLayout in = layout();
    const auto blocks = plan_extras(in);
    check_extras(in, blocks);  // must not throw
    EXPECT(total(blocks) > 0);
    // (nearly) the same number of extra slots in every layer, and little of the workspace left over
    for (const auto& l : blocks) EXPECT(std::abs(per_layer(l) - per_layer(blocks[0])) <= 1);
    size_t bytes = 0, sum_sb = 0;
    for (size_t l = 0; l < blocks.size(); l++)
        for (const ExtraBlock& b : blocks[l]) bytes += (size_t)b.count * in.slot_bytes[l];
    for (size_t sb : in.slot_bytes) sum_sb += sb;
    EXPECT(bytes <= in.arena_bytes);
    EXPECT(in.arena_bytes - bytes < (size_t)in.ranks * sum_sb);
}

void test_no_room_no_extras() {
    ExtraLayout in = layout(2, 1);
    EXPECT(total(plan_extras(in)) == 0);
    in = layout();
    in.max_slots = in.base_slots;  // every expert has a slot already
    EXPECT(total(plan_extras(in)) == 0);
    in = layout(2, 100);
    in.max_slots = in.base_slots + 3;
    for (const auto& l : plan_extras(in)) EXPECT(per_layer(l) == 3);
}

void test_checks_reject_a_bad_layout() {
    const ExtraLayout in = layout();
    auto blocks = plan_extras(in);
    auto off_grid = blocks;
    off_grid[2][0].off += 1;
    EXPECT(tona::test::throws([&] { check_extras(in, off_grid); }));
    auto outside = blocks;
    outside[0][0].count += 1000;
    EXPECT(tona::test::throws([&] { check_extras(in, outside); }));
    auto below = blocks;
    below[1][0].first = 3;
    EXPECT(tona::test::throws([&] { check_extras(in, below); }));
    auto overlap = blocks;
    overlap[3] = overlap[4];
    overlap[3][0].first = overlap[4][0].first;
    EXPECT(tona::test::throws([&] { check_extras(in, overlap); }));
}

void test_top_block_keeps_the_read_padding() {
    const ExtraLayout in = layout();
    auto blocks = plan_extras(in);
    size_t top = 0;
    for (size_t l = 0; l < blocks.size(); l++)
        for (const ExtraBlock& b : blocks[l]) top = std::max(top, b.off + (size_t)b.count * in.slot_bytes[l]);
    EXPECT(top + in.read_pad <= in.arena_bytes);
    // a block that ends at the very end of the workspace would be read past the allocation
    for (size_t l = 0; l < blocks.size(); l++)
        for (ExtraBlock& b : blocks[l])
            if (b.off + (size_t)b.count * in.slot_bytes[l] == top) b.off += in.arena_bytes - top;
    EXPECT(tona::test::throws([&] { check_extras(in, blocks); }));
}

void test_slot_index_reaches_the_block() {
    const ExtraLayout in = layout();
    const auto blocks = plan_extras(in);
    for (size_t l = 0; l < blocks.size(); l++)
        for (const ExtraBlock& b : blocks[l])
            EXPECT(in.slot_off[l] + (size_t)b.first * in.slot_bytes[l] == in.arena_start + b.off);
}

void test_evict_takes_only_what_the_prefill_covers() {
    const ExtraLayout in = layout();
    const auto blocks = plan_extras(in);
    auto cs = caches(in);
    install_extras(cs, blocks);
    const std::vector<int> cnt = per_counts(blocks);
    for (size_t l = 0; l < cs.size(); l++)
        EXPECT(cs[l].n_slots == in.base_slots + cnt[l] && (int)cs[l].free_slots.size() == cs[l].n_slots);
    // fill every slot of layer 0 with a distinct expert
    LayerCache& C0 = cs[0];
    for (int e = 0; e < C0.n_slots; e++) C0.add_front(e, C0.take_slot());
    EXPECT(C0.free_slots.empty() && (int)C0.where.size() == C0.n_slots);
    EXPECT(evict_extras(cs, 0) == 0);  // a prefill that needs nothing
    const int taken = evict_extras(cs, in.arena_bytes / 2);
    EXPECT(taken > 0 && taken < total(blocks));
    for (size_t l = 0; l < cs.size(); l++) {
        // no expert in a slot that is out of use, and the base slots are untouched
        for (const auto& [e, w] : cs[l].where)
            for (const ExtraBlock& b : cs[l].extras)
                EXPECT(b.live || w.first < b.first || w.first >= b.first + b.count);
        EXPECT(live_extras(cs[l]) > 0);  // interleaved by rank: every layer keeps some
    }
    EXPECT((int)C0.where.size() == C0.n_slots - (cnt[0] - live_extras(C0)));
    EXPECT(C0.where.size() >= (size_t)in.base_slots);
    EXPECT(evict_extras(cs, in.arena_bytes) > 0);
    for (const LayerCache& C : cs) EXPECT(live_extras(C) == 0 && (int)C.where.size() <= in.base_slots);
    EXPECT(restore_extras(cs) == total(blocks));
    for (size_t l = 0; l < cs.size(); l++) EXPECT(live_extras(cs[l]) == cnt[l]);
    // the base slots are used first again: they stay at the back of the free list
    EXPECT(cs[1].take_slot() < in.base_slots);
}

void test_evict_is_idempotent_and_restore_only_gives_back_dead_blocks() {
    const ExtraLayout in = layout();
    const auto blocks = plan_extras(in);
    auto cs = caches(in);
    install_extras(cs, blocks);
    const int a = evict_extras(cs, in.arena_bytes);
    EXPECT(a == total(blocks));
    EXPECT(evict_extras(cs, in.arena_bytes) == 0);
    EXPECT(restore_extras(cs) == a);
    EXPECT(restore_extras(cs) == 0);
    for (const LayerCache& C : cs) {
        std::set<int> distinct(C.free_slots.begin(), C.free_slots.end());
        EXPECT(distinct.size() == C.free_slots.size() && (int)distinct.size() == C.n_slots);
    }
}

// Elastic blocks (mega_elastic.h): added at the back of the free list, killed for good, never revived by a prefill's
// restore
void test_elastic_blocks_die_for_good() {
    auto cs = caches(layout());
    LayerCache& C = cs[0];
    add_elastic_block(C, 1000, 4, false);
    EXPECT(C.n_slots == 24 && C.free_slots.back() == 1000 && C.free_slots.size() == 24);
    // An expert sits in one of the new slots, another in a base slot
    const int s = C.take_slot();
    EXPECT(s == 1000);
    C.add_front(7, s);
    std::erase(C.free_slots, 5);
    C.add_front(8, 5);  // a base slot
    EXPECT(kill_block(C, 1000) == 4);
    EXPECT(C.n_slots == 20 && !C.where.contains(7) && C.where.contains(8) == 1);
    for (int f : C.free_slots) EXPECT(f < 20);  // no free slot in the dead block
    EXPECT(kill_block(C, 1000) == 0);           // a second time: nothing
    EXPECT(restore_extras(cs) == 0);            // a prefill's restore does not bring it back
    EXPECT(evict_extras(cs, SIZE_MAX / 2) == 0);
    reset_slots(C, 20);
    for (int f : C.free_slots) EXPECT(f < 20);
    EXPECT(live_extras(C) == 0);
    // At start the elastic slots go to the front of the list: the preload (which takes from the back) fills the core
    // first
    LayerCache D;
    for (int s = 19; s >= 0; s--) D.free_slots.push_back(s);
    add_elastic_block(D, 500, 4, true);
    EXPECT(D.free_slots.back() == 0 && D.free_slots.front() == 503 && D.free_slots.size() == 24);
}

}  // namespace

int main() {
    return tona::test::guard("mega_extra", [&] {
        test_layout_fills_the_workspace();
        test_no_room_no_extras();
        test_checks_reject_a_bad_layout();
        test_top_block_keeps_the_read_padding();
        test_slot_index_reaches_the_block();
        test_evict_takes_only_what_the_prefill_covers();
        test_evict_is_idempotent_and_restore_only_gives_back_dead_blocks();
        test_elastic_blocks_die_for_good();
        return tona::test::finish("mega_extra");
    });
}
