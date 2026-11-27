// Read-ahead of a pass's n-gram rows (TONA_PLE_PREFETCH, on by default on macOS, off elsewhere). The table is a
// file mapping larger than the RAM; copying its rows one after the other faults each uncached page in turn, one SSD
// read at queue depth 1 (5.4 ms per decode pass in slow server starts on an M1 Max,
// research/measurements/2026-11-14-mac-ple-copy.md). Asking for all pages first lets the SSD serve them together:
// 0.37 against 4.7 ms per 35 cold pages of the same file (micro probe). The copy afterwards stays the real read, so
// the bytes copied never change; a failed hint costs only the time it would have saved.
#pragma once

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "backend/env.h"

namespace tona::backend::ple_prefetch {

// 0 off; 1 the pass's own rows just before the copy; 2 also the next target pass's rows while the MTP head drafts
// them (decode +1.2 % after 8000 tokens of prompt, research/measurements/2026-11-14-mac-ple-prefetch.md)
inline int level() {
#ifdef __APPLE__
    // Mac server decode +4.2 % (code) and +5.9 % (text) with cold rows at level 1, +1.2 % more at 2; bit-identical,
    // no memory of its own (research/measurements/2026-11-14-mac-ple-prefetch.md)
    constexpr int64_t kDefault = 2;
#else
    constexpr int64_t kDefault = 0;  // Linux: off until measured on the reference PC
#endif
    static const int on = (int)env::integer("TONA_PLE_PREFETCH", kDefault);
    return on;
}

inline bool enabled() { return level() >= 1; }
inline bool ahead() { return level() >= 2; }

struct Range {
    uintptr_t begin, end;  // page-aligned addresses, [begin, end)
};

// The pages the rows touch, each once, adjacent pages merged into one range; a row can straddle two pages
inline std::vector<Range> ranges(uintptr_t base, uint64_t row_bytes, const int32_t* rows, size_t n, uint64_t page) {
    std::vector<uintptr_t> pages;
    pages.reserve(n * 2);
    for (size_t i = 0; i < n; i++) {
        const uintptr_t first = base + (uint64_t)rows[i] * row_bytes;
        for (uintptr_t p = first / page; p <= (first + row_bytes - 1) / page; p++) pages.push_back(p);
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    std::vector<Range> out;
    for (const uintptr_t p : pages) {
        if (!out.empty() && out.back().end == p * page) {
            out.back().end += page;
        } else {
            out.push_back({p * page, (p + 1) * page});
        }
    }
    return out;
}

// Asks the kernel to read the rows' pages ahead (macOS and Linux read file pages asynchronously on MADV_WILLNEED);
// returns how many requests failed
inline int prefetch(const uint8_t* base, uint64_t row_bytes, const int32_t* rows, size_t n) {
    static const auto page = (uint64_t)sysconf(_SC_PAGESIZE);  // 16 KB on Apple silicon, 4 KB on x86-64 Linux
    int failed = 0;
    for (const Range& r : ranges((uintptr_t)base, row_bytes, rows, n, page)) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr): the address is the mapping's own, rounded down to its page
        failed += madvise(reinterpret_cast<void*>(r.begin), r.end - r.begin, MADV_WILLNEED) != 0;
    }
    return failed;
}

}  // namespace tona::backend::ple_prefetch
