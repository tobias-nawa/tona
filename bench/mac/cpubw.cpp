// CPU memory bandwidth on Apple silicon with NEON: read, write, copy over 1..N threads. Each thread works on its own
// slice of one buffer. Output: one JSON line per point (median, min, max of the repetitions).
// Plan and evaluation: research/measurements/2026-11-11-mac-bandwidth.md
// Build: clang++ -std=c++20 -O3 -mcpu=apple-m1 bench/mac/cpubw.cpp -o work/build/mac/cpubw
// Usage: cpubw [MiB per buffer, default 1024] [repetitions, default 5] [thread counts, default 1,2,4,8,10]
#include <arm_neon.h>
#include <pthread.h>
#include <sys/qos.h>

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

uint64_t g_sink = 0;

void op_read(const uint8_t* p, size_t n) {
    uint32x4_t a0 = vdupq_n_u32(0), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i < n; i += 64) {
        a0 = vaddq_u32(a0, vld1q_u32(reinterpret_cast<const uint32_t*>(p + i)));
        a1 = vaddq_u32(a1, vld1q_u32(reinterpret_cast<const uint32_t*>(p + i + 16)));
        a2 = vaddq_u32(a2, vld1q_u32(reinterpret_cast<const uint32_t*>(p + i + 32)));
        a3 = vaddq_u32(a3, vld1q_u32(reinterpret_cast<const uint32_t*>(p + i + 48)));
    }
    const uint32x4_t s = vaddq_u32(vaddq_u32(a0, a1), vaddq_u32(a2, a3));
    __atomic_fetch_add(&g_sink, vaddvq_u32(s), __ATOMIC_RELAXED);
}

// read with 8 independent sums and 64-byte loads (ld1 of four q registers): checks whether op_read is limited by its
// add chains instead of the memory
void op_read8(const uint8_t* p, size_t n) {
    uint32x4_t s[8];
    for (auto& v : s) v = vdupq_n_u32(0);
    for (size_t i = 0; i < n; i += 128) {
        const uint32x4x4_t x = vld1q_u32_x4(reinterpret_cast<const uint32_t*>(p + i));
        const uint32x4x4_t y = vld1q_u32_x4(reinterpret_cast<const uint32_t*>(p + i + 64));
        s[0] = vaddq_u32(s[0], x.val[0]), s[1] = vaddq_u32(s[1], x.val[1]);
        s[2] = vaddq_u32(s[2], x.val[2]), s[3] = vaddq_u32(s[3], x.val[3]);
        s[4] = vaddq_u32(s[4], y.val[0]), s[5] = vaddq_u32(s[5], y.val[1]);
        s[6] = vaddq_u32(s[6], y.val[2]), s[7] = vaddq_u32(s[7], y.val[3]);
    }
    uint32x4_t t = s[0];
    for (int k = 1; k < 8; k++) t = vaddq_u32(t, s[k]);
    __atomic_fetch_add(&g_sink, vaddvq_u32(t), __ATOMIC_RELAXED);
}

void op_write(uint8_t* p, size_t n) {
    const uint32x4_t z = vdupq_n_u32(0x01010101u);
    for (size_t i = 0; i < n; i += 64) {
        vst1q_u32(reinterpret_cast<uint32_t*>(p + i), z);
        vst1q_u32(reinterpret_cast<uint32_t*>(p + i + 16), z);
        vst1q_u32(reinterpret_cast<uint32_t*>(p + i + 32), z);
        vst1q_u32(reinterpret_cast<uint32_t*>(p + i + 48), z);
    }
}

void op_copy(const uint8_t* src, uint8_t* dst, size_t n) { std::memcpy(dst, src, n); }

struct Point {
    const char* test;
    int threads;
    double bytes_moved;
    std::vector<double> gbs;
};

// One timed pass: all threads start at a barrier, the time runs from the first start to the last finish
double run_pass(int nt, size_t bytes, uint8_t* a, uint8_t* b, int op) {
    const size_t slice = bytes / (size_t)nt / 128 * 128;
    std::barrier sync(nt + 1);
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; t++) {
        ts.emplace_back([&, t] {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
            uint8_t* pa = a + (size_t)t * slice;
            uint8_t* pb = b + (size_t)t * slice;
            sync.arrive_and_wait();
            if (op == 0) op_read(pa, slice);
            if (op == 1) op_write(pa, slice);
            if (op == 2) op_copy(pa, pb, slice);
            if (op == 3) op_read8(pa, slice);
            sync.arrive_and_wait();
        });
    }
    sync.arrive_and_wait();
    const auto t0 = std::chrono::steady_clock::now();
    sync.arrive_and_wait();
    const auto t1 = std::chrono::steady_clock::now();
    for (auto& t : ts) t.join();
    return std::chrono::duration<double>(t1 - t0).count();
}

std::vector<int> parse_threads(const char* s) {
    std::vector<int> out;
    std::string v(s);
    size_t at = 0;
    while (at < v.size()) {
        const size_t comma = v.find(',', at);
        out.push_back(std::atoi(v.substr(at, comma - at).c_str()));
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    return out;
}

void print_point(const Point& p) {
    std::vector<double> g = p.gbs;
    std::sort(g.begin(), g.end());
    std::printf(
        "{\"kind\": \"cpu\", \"test\": \"%s\", \"threads\": %d, \"median_gbs\": %.2f, \"min_gbs\": %.2f, "
        "\"max_gbs\": %.2f, \"runs\": %zu}\n",
        p.test, p.threads, g[g.size() / 2], g.front(), g.back(), g.size());
}

}  // namespace

int main(int argc, char** argv) {
    const size_t mib = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1024;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
    const std::vector<int> counts = argc > 3 ? parse_threads(argv[3]) : std::vector<int>{1, 2, 4, 8, 10};
    const size_t bytes = mib << 20;
    auto* a = static_cast<uint8_t*>(std::aligned_alloc(16384, bytes));
    auto* b = static_cast<uint8_t*>(std::aligned_alloc(16384, bytes));
    if (!a || !b) return 1;
    std::memset(a, 1, bytes);  // touch every page before timing
    std::memset(b, 2, bytes);
    const char* names[4] = {"read", "write", "copy", "read8"};
    const int first = std::getenv("CPUBW_ONLY_READ8") ? 3 : 0;
    for (const int nt : counts) {
        for (int op = first; op < 4; op++) {
            Point p{names[op], nt, (op == 2 ? 2.0 : 1.0) * (double)(bytes / (size_t)nt / 128 * 128) * nt, {}};
            run_pass(nt, bytes, a, b, op);  // warm-up
            for (int r = 0; r < reps; r++) p.gbs.push_back(p.bytes_moved / run_pass(nt, bytes, a, b, op) / 1e9);
            print_point(p);
        }
    }
    std::fprintf(stderr, "sink %llu\n", (unsigned long long)g_sink);
    std::free(a);
    std::free(b);
}
