#include "gpu/mega_extra.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>

namespace tona::gpu {

namespace {

bool place(const ExtraLayout& in, const std::vector<int>& x, std::vector<std::vector<ExtraBlock>>& blocks) {
    const size_t L = in.slot_bytes.size();
    blocks.assign(L, {});
    if (in.arena_bytes < in.read_pad) return false;
    size_t cursor = in.arena_bytes - in.read_pad;  // the top block's over-read stays inside the allocation
    for (int r = 0; r < in.ranks; r++) {
        for (size_t l = 0; l < L; l++) {
            const int n = x[l] / in.ranks + (r < x[l] % in.ranks);
            if (n == 0) continue;
            const size_t sb = in.slot_bytes[l], len = (size_t)n * sb;
            if (cursor < len) return false;
            const size_t hi = cursor - len;
            // The index must be a whole number of slots from the layer's first slot
            const size_t phase = (in.arena_start + hi - in.slot_off[l]) % sb;
            if (phase > hi) return false;
            const size_t a = hi - phase;
            blocks[l].push_back({(int)((in.arena_start + a - in.slot_off[l]) / sb), n, a, true});
            cursor = a;
        }
    }
    return true;
}

void drop_block(LayerCache& C, const ExtraBlock& b) {
    std::vector<int> gone;
    for (const auto& [e, where] : C.where)
        if (where.first >= b.first && where.first < b.first + b.count) gone.push_back(e);
    for (int e : gone) C.erase(C.where.find(e));
    std::erase_if(C.free_slots, [&](int s) { return s >= b.first && s < b.first + b.count; });
}

}  // namespace

std::vector<std::vector<ExtraBlock>> plan_extras(const ExtraLayout& in) {
    std::vector<std::vector<ExtraBlock>> blocks(in.slot_bytes.size());
    size_t per_set = 0;
    for (size_t sb : in.slot_bytes) per_set += sb;
    if (per_set == 0 || in.ranks < 1) return blocks;
    const size_t L = in.slot_bytes.size();
    const int cap = std::max(in.max_slots - in.base_slots, 0);
    std::vector<int> x(L, (int)std::min<size_t>(in.arena_bytes / per_set, (size_t)cap));
    while (x[0] > 0 && !place(in, x, blocks)) x.assign(L, x[0] - 1);
    if (x[0] <= 0) return std::vector<std::vector<ExtraBlock>>(L);
    // The alignment of the blocks eats part of the workspace, which the same count for every layer cannot use fully:
    // one more slot for each layer in turn as long as it fits
    for (size_t l = 0; l < L; l++) {
        if (x[l] >= cap) continue;
        x[l]++;
        if (!place(in, x, blocks)) x[l]--;
    }
    if (!place(in, x, blocks)) throw std::runtime_error("gpu::Mega: extra slots, layout lost");
    check_extras(in, blocks);
    return blocks;
}

void check_extras(const ExtraLayout& in, const std::vector<std::vector<ExtraBlock>>& blocks) {
    std::vector<std::pair<size_t, size_t>> spans;  // in the workspace, over all layers
    for (size_t l = 0; l < blocks.size(); l++)
        for (const ExtraBlock& b : blocks[l]) {
            const size_t sb = in.slot_bytes[l], len = (size_t)b.count * sb;
            auto bad = [&](const char* why) {
                throw std::runtime_error(
                    std::format("gpu::Mega: extra slots, layer {} block at {}: {}", l, b.off, why));
            };
            if (b.count <= 0 || b.off + len + in.read_pad > in.arena_bytes)
                bad("outside the workspace (with the read padding)");
            if (b.first < in.base_slots) bad("index inside the base slots");
            if ((size_t)b.first * sb + in.slot_off[l] != in.arena_start + b.off) bad("not on the slot grid");
            if ((size_t)b.first + (size_t)b.count > (size_t)INT32_MAX) bad("index beyond 32 bits");
            spans.emplace_back(b.off, b.off + len);
        }
    std::sort(spans.begin(), spans.end());
    for (size_t i = 1; i < spans.size(); i++)
        if (spans[i].first < spans[i - 1].second)
            throw std::runtime_error(std::format("gpu::Mega: extra slots overlap at {}", spans[i].first));
}

void install_extras(std::vector<LayerCache>& caches, const std::vector<std::vector<ExtraBlock>>& blocks) {
    for (size_t l = 0; l < caches.size() && l < blocks.size(); l++) {
        LayerCache& C = caches[l];
        C.extras = blocks[l];
        std::vector<int> add;
        for (const ExtraBlock& b : C.extras)
            for (int s = b.first + b.count - 1; s >= b.first; s--) add.push_back(s);
        C.free_slots.insert(C.free_slots.begin(), add.begin(), add.end());
        C.n_slots = (int)C.free_slots.size() + (int)C.where.size();
    }
}

int evict_extras(std::vector<LayerCache>& caches, size_t used) {
    int taken = 0;
    for (LayerCache& C : caches)
        for (ExtraBlock& b : C.extras)
            if (b.live && b.off < used) {
                drop_block(C, b);
                b.live = false;
                taken += b.count;
            }
    return taken;
}

int kill_block(LayerCache& C, int first) {
    for (ExtraBlock& b : C.extras) {
        if (b.first != first || b.dead) continue;
        if (b.live) drop_block(C, b);  // a block that a prefill holds has no experts and no free slots already
        b.live = false, b.dead = true;
        C.n_slots -= b.count;
        return b.count;
    }
    return 0;
}

void add_elastic_block(LayerCache& C, int first, int count, bool front) {
    C.extras.push_back({first, count, SIZE_MAX, true, false});
    std::vector<int> add;
    for (int s = first + count - 1; s >= first; s--) add.push_back(s);
    C.free_slots.insert(front ? C.free_slots.begin() : C.free_slots.end(), add.begin(), add.end());
    C.n_slots += count;
}

int restore_extras(std::vector<LayerCache>& caches) {
    int given = 0;
    for (LayerCache& C : caches)
        for (ExtraBlock& b : C.extras) {
            if (b.live || b.dead) continue;
            b.live = true;
            std::vector<int> add;
            for (int s = b.first + b.count - 1; s >= b.first; s--) add.push_back(s);
            C.free_slots.insert(C.free_slots.begin(), add.begin(), add.end());
            given += b.count;
        }
    return given;
}

void reset_slots(LayerCache& C, int base) {
    C.clear_order(), C.free_slots.clear(), C.installed.clear(), C.pending.clear();
    for (const ExtraBlock& b : C.extras)
        if (b.live && !b.dead)
            for (int s = b.first; s < b.first + b.count; s++) C.free_slots.push_back(s);
    for (int s = base - 1; s >= 0; s--) C.free_slots.push_back(s);
}

int live_extras(const LayerCache& C) {
    int n = 0;
    for (const ExtraBlock& b : C.extras) n += b.live ? b.count : 0;
    return n;
}

}  // namespace tona::gpu
