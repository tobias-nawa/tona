// The never-remap rule of the elastic slot pool (docs/architecture/expert-tiers.md, section 5a): an address range that
// has been unmapped is never mapped again, because the GPU may still hold a translation for it and then writes into
// pages that were given back to the driver (ROCm/TheRock #8744, reproduced on this machine: 29 of 400 chunks wrong).
// The ledger knows every range the pool ever mapped; a map request that touches a dead range is refused, in every build
// type. Pure, so a unit test can drive it without a GPU.
#pragma once

#include <cstdint>
#include <iterator>
#include <map>

namespace tona::backend {

class VaLedger {
public:
    enum class State : uint8_t { live, dead };

    // False (and nothing recorded) if [begin, begin + bytes) overlaps any range that is live or dead
    bool map(uint64_t begin, uint64_t bytes) {
        if (bytes == 0 || overlaps(begin, bytes)) return refuse();
        ranges_[begin] = {begin + bytes, State::live};
        return true;
    }

    // The range [begin, ...) was mapped exactly as given: it dies and stays in the ledger for good
    bool unmap(uint64_t begin) {
        auto it = ranges_.find(begin);
        if (it == ranges_.end() || it->second.state == State::dead) return refuse();
        it->second.state = State::dead;
        return true;
    }

    // Every byte of [begin, begin + bytes) is inside one live range: the check of every path that writes into a slot
    bool is_live(uint64_t begin, uint64_t bytes) const {
        auto it = ranges_.upper_bound(begin);
        if (it == ranges_.begin()) return false;
        --it;
        return it->second.state == State::live && begin + bytes <= it->second.end;
    }

    uint64_t refused() const { return refused_; }
    size_t live_count() const {
        size_t n = 0;
        for (const auto& [b, r] : ranges_) n += r.state == State::live;
        return n;
    }

private:
    struct Range {
        uint64_t end;
        State state;
    };
    bool overlaps(uint64_t begin, uint64_t bytes) const {
        auto it = ranges_.lower_bound(begin);
        if (it != ranges_.end() && it->first < begin + bytes) return true;
        if (it == ranges_.begin()) return false;
        return std::prev(it)->second.end > begin;
    }
    bool refuse() {
        refused_++;
        return false;
    }

    std::map<uint64_t, Range> ranges_;
    uint64_t refused_ = 0;
};

}  // namespace tona::backend
