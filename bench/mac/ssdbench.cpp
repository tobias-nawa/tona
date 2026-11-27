// SSD reads on macOS for expert-sized blocks: pread from N threads (queue depth = threads, one request each),
// random offsets aligned to the block size or sequential, with or without the page cache (F_NOCACHE).
// Output: one JSON line per point. Plan and evaluation: research/measurements/2026-11-11-mac-ssd.md
// Build: clang++ -std=c++20 -O3 -mcpu=apple-m1 bench/mac/ssdbench.cpp -o work/build/mac/ssdbench
// Usage: ssdbench write FILE GIB
//        ssdbench read FILE rand|seq BLOCK_BYTES QD TOTAL_BYTES NOCACHE(0|1) [SEED]
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

uint64_t xorshift(uint64_t& s) {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return s;
}

int write_file(const char* path, uint64_t gib) {
    const int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return std::perror("open"), 1;
    ::fcntl(fd, F_NOCACHE, 1);  // the data must not stay in the page cache: the first buffered read is cold
    const size_t chunk = 64ull << 20;
    std::vector<uint64_t> buf(chunk / 8);
    uint64_t s = 0x9e3779b97f4a7c15ull;
    const auto t0 = Clock::now();
    for (uint64_t done = 0; done < gib << 30; done += chunk) {
        for (auto& v : buf) v = xorshift(s);
        if (::write(fd, buf.data(), chunk) != (ssize_t)chunk) return std::perror("write"), 1;
    }
    ::fsync(fd);
    ::close(fd);
    const double t = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("{\"kind\": \"write\", \"bytes\": %llu, \"s\": %.3f, \"gbs\": %.3f}\n", (unsigned long long)(gib << 30),
                t, (double)(gib << 30) / t / 1e9);
    return 0;
}

struct ReadArgs {
    const char* path;
    bool random;
    size_t block;
    int qd;
    uint64_t total;
    bool nocache;
    uint64_t seed;
};

int read_test(const ReadArgs& a) {
    const int fd = ::open(a.path, O_RDONLY);
    if (fd < 0) return std::perror("open"), 1;
    if (a.nocache) ::fcntl(fd, F_NOCACHE, 1);
    if (a.random) ::fcntl(fd, F_RDAHEAD, 0);
    const uint64_t size = (uint64_t)::lseek(fd, 0, SEEK_END);
    const uint64_t nblocks = size / a.block, nreq = std::max<uint64_t>(1, a.total / a.block);
    std::vector<uint64_t> off(nreq);
    uint64_t s = a.seed;
    for (uint64_t i = 0; i < nreq; i++) off[i] = (a.random ? xorshift(s) % nblocks : i % nblocks) * a.block;
    std::atomic<uint64_t> next{0};
    std::atomic<bool> failed{false};
    std::vector<std::vector<double>> lat(a.qd);
    std::vector<std::thread> ts;
    const auto t0 = Clock::now();
    for (int t = 0; t < a.qd; t++) {
        ts.emplace_back([&, t] {
            void* buf = std::aligned_alloc(16384, (a.block + 16383) / 16384 * 16384);
            for (uint64_t i; (i = next.fetch_add(1)) < nreq;) {
                const auto r0 = Clock::now();
                if (::pread(fd, buf, a.block, (off_t)off[i]) != (ssize_t)a.block) failed = true;
                lat[t].push_back(std::chrono::duration<double, std::micro>(Clock::now() - r0).count());
            }
            std::free(buf);
        });
    }
    for (auto& t : ts) t.join();
    const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    ::close(fd);
    if (failed) return std::fprintf(stderr, "short read\n"), 1;
    std::vector<double> all;
    for (auto& v : lat) all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());
    const double bytes = (double)nreq * (double)a.block;
    std::printf(
        "{\"kind\": \"read\", \"pattern\": \"%s\", \"block\": %zu, \"qd\": %d, \"nocache\": %d, \"requests\": "
        "%llu, \"s\": %.4f, \"gbs\": %.3f, \"iops\": %.0f, \"lat_med_us\": %.1f, \"lat_p99_us\": %.1f}\n",
        a.random ? "rand" : "seq", a.block, a.qd, a.nocache ? 1 : 0, (unsigned long long)nreq, sec, bytes / sec / 1e9,
        (double)nreq / sec, all[all.size() / 2], all[all.size() * 99 / 100]);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "write") return write_file(argv[2], std::strtoull(argv[3], nullptr, 10));
    if (argc >= 8 && std::string(argv[1]) == "read") {
        const ReadArgs a{argv[2],
                         std::string(argv[3]) == "rand",
                         (size_t)std::strtoull(argv[4], nullptr, 10),
                         std::atoi(argv[5]),
                         std::strtoull(argv[6], nullptr, 10),
                         std::atoi(argv[7]) != 0,
                         argc > 8 ? std::strtoull(argv[8], nullptr, 10) : 12345};
        return read_test(a);
    }
    std::fprintf(stderr, "usage: see the header of bench/mac/ssdbench.cpp\n");
    return 2;
}
