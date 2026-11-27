// Which experts a layer's cache holds and in which order they are replaced: plain LRU or ClassLRU. Host code only, the
// same for every backend that keeps experts in fast memory; a slot is an index, what lives behind it (VRAM, a buffer)
// is the backend's.
#pragma once

#include <cstdint>
#include <iterator>
#include <list>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backend/class_lru.h"

namespace tona::backend {

struct ExpertOrder {
    using Where = std::unordered_map<int, std::pair<int, std::list<int>::iterator>>;

    std::list<int> lru;
    Where where;  // expert -> (slot, place in the order)
    std::vector<int> free_slots;

    // TONA_EXPERT_POLICY=classlru: the order of replacement is kept here instead of in lru (which stays empty)
    std::optional<ClassLru> order;

    // The order of replacement. lru: front = most recently used, back = the next victim. Hits, new experts at the
    // front, preloaded ones at the back; with `order` the same calls go to the class lists.
    void touch(Where::iterator it) {
        if (order)
            order->promote(it->first, it->second.second);
        else
            lru.splice(lru.begin(), lru, it->second.second);
    }
    void add_front(int e, int slot) {
        if (order) return void(where[e] = {slot, order->insert(e)});
        lru.push_front(e);
        where[e] = {slot, lru.begin()};
    }
    void add_back(int e, int slot) {
        if (order) return void(where[e] = {slot, order->insert_fresh(e)});
        lru.push_back(e);
        where[e] = {slot, std::prev(lru.end())};
    }
    void erase(Where::iterator it) {
        if (order)
            order->erase(it->first, it->second.second);
        else
            lru.erase(it->second.second);
        where.erase(it);
    }
    void clear_order() {
        where.clear(), lru.clear();
        if (order) order->clear();
    }
    // Once per pass: every expert of the union counts as used (before the hits are moved), then the clock advances by
    // the tokens of the pass. Without ClassLRU nothing to do.
    void credit(const std::vector<int>& uni) {
        if (order)
            for (int e : uni) order->use(e);
    }
    void tick(int tokens) {
        if (order) order->tick(tokens);
    }

    // Slot for a new expert: a free one, else that of the next victim that is not in keep; -1 if all occupied ones are
    // in keep. The evicted expert leaves the order and where.
    int take_slot(const std::unordered_set<int>* keep = nullptr) {
        if (!free_slots.empty()) {
            const int s = free_slots.back();
            free_slots.pop_back();
            return s;
        }
        if (order) {
            const int e = order->victim(keep);
            if (e < 0) return -1;
            const auto it = where.find(e);
            const int s = it->second.first;
            erase(it);
            return s;
        }
        if (lru.empty()) return -1;
        auto v = std::prev(lru.end());
        if (keep)
            while (keep->contains(*v)) {
                if (v == lru.begin()) return -1;
                --v;
            }
        const auto it = where.find(*v);
        const int s = it->second.first;
        where.erase(it);
        lru.erase(v);
        return s;
    }
};

// The union of a pass enters the cache (GPU path): hits move to the front, a new expert takes a free slot or the one of
// the next victim that is not in the union (LRU: the union sits at the front of the list and cannot be evicted, so
// `keep` is only built for ClassLRU). slot[u] is the slot of uni[u]; the misses, with their slots, are for the backend
// to fill. Stops with out_of_slots when every slot holds an expert of the union.
struct SlotPlan {
    std::vector<std::pair<int, int>> miss;  // (expert, slot)
    int64_t hits = 0;
    bool out_of_slots = false;
};

inline void assign_slots(ExpertOrder& C, const std::vector<int>& uni, int tokens, int32_t* slot, SlotPlan& out) {
    std::unordered_set<int> keep;
    if (C.order) keep.insert(uni.begin(), uni.end());
    const std::unordered_set<int>* keep_ptr = C.order ? &keep : nullptr;
    C.credit(uni);
    for (size_t u = 0; u < uni.size(); u++) {
        const int e = uni[u];
        auto it = C.where.find(e);
        if (it != C.where.end()) {
            out.hits++;
            slot[u] = it->second.first;
            C.touch(it);
            continue;
        }
        const int s = C.take_slot(keep_ptr);
        if (s < 0) {
            out.out_of_slots = true;
            return;
        }
        C.add_front(e, s);
        out.miss.emplace_back(e, s);
        slot[u] = s;
    }
    C.tick(tokens);
}

}  // namespace tona::backend
