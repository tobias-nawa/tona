// RAM read bandwidth for the speed estimate (hw.h).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <new>
#include <thread>
#include <vector>

#include "core/hw.h"

namespace tona::install {

namespace {

constexpr size_t kWords = (512ull << 20) / sizeof(uint64_t);

uint64_t sum_slice(const uint64_t* p, size_t n) {
    uint64_t a = 0, b = 0, c = 0, d = 0;  // four chains: one load chain would be latency bound
    for (size_t i = 0; i + 4 <= n; i += 4) a += p[i], b += p[i + 1], c += p[i + 2], d += p[i + 3];
    return a + b + c + d;
}

}  // namespace

double measure_ram_gbs(int threads) {
    // 8 threads read dual-channel DDR5 at the same rate as 32 on the reference PC (docs/backends/reference-platform.md)
    const int n = std::clamp(threads, 2, 8);
    std::vector<uint64_t> buf;
    try {
        buf.assign(kWords, 1);  // writing touches every page
    } catch (const std::bad_alloc&) {
        return 0;
    }
    std::vector<uint64_t> sink(static_cast<size_t>(n));
    double best = 0;
    for (int pass = 0; pass < 3; pass++) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; i++)
            pool.emplace_back([&, i] {
                const size_t part = kWords / static_cast<size_t>(n);
                sink[static_cast<size_t>(i)] = sum_slice(buf.data() + part * static_cast<size_t>(i), part);
            });
        for (std::thread& t : pool) t.join();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        best = std::max(best, static_cast<double>(kWords * sizeof(uint64_t)) / s / 1e9);
    }
    return std::any_of(sink.begin(), sink.end(), [](uint64_t v) { return v == 0; }) ? 0 : best;
}

}  // namespace tona::install
