#include "core/disk.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "core/parse.h"
#include "core/sys.h"

namespace fs = std::filesystem;

namespace tona::install {

namespace {

using Clock = std::chrono::steady_clock;

std::string mount_source(const std::string& path) {
    std::error_code ec;
    return parse::mount_source(read_file("/proc/self/mountinfo"), fs::weakly_canonical(path, ec).string());
}

// from a block device (dm-0, nvme0n1p3, sda2) to the disk: device mapper through slaves/, partition through the parent
std::string to_disk(std::string dev, int depth = 0) {
    if (depth > 8) return dev;
    const fs::path sys = fs::path("/sys/class/block") / dev;
    std::error_code ec;
    for (const auto& s : fs::directory_iterator(sys / "slaves", ec)) return to_disk(s.path().filename(), depth + 1);
    if (fs::exists(sys / "partition")) return fs::canonical(sys, ec).parent_path().filename().string();
    return dev;
}

}  // namespace

// largest .gguf under models_dir, empty if there is none of at least 1 GiB (fresh install)
std::string largest_model_file(const std::string& models_dir) {
    uintmax_t best = 0;
    std::string file;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(models_dir, ec); it != fs::recursive_directory_iterator();
         it.increment(ec)) {
        if (it->path().extension() != ".gguf") continue;
        const fs::path p = fs::canonical(it->path(), ec);
        const uintmax_t n = fs::file_size(p, ec);
        if (!ec && n > best) best = n, file = p.string();
    }
    return best >= (1ULL << 30) ? file : std::string{};
}

namespace {

// random 4 KiB reads from 32 threads, one after the other each (like a ring with depth 32)
void measure_random(int fd, int64_t blocks, double seconds, DiskSpeed& r) {
    std::atomic<int64_t> n{0}, ns{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    threads.reserve(32);
    for (int t = 0; t < 32; t++)
        threads.emplace_back([&, t] {
            void* buf = std::aligned_alloc(4096, 4096);
            std::mt19937_64 rng(t + 1);
            while (!stop) {
                const auto t0 = Clock::now();
                if (pread(fd, buf, 4096, (off_t)(rng() % blocks) * 4096) != 4096) break;
                ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
                n++;
            }
            std::free(buf);
        });
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop = true;
    for (auto& t : threads) t.join();
    r.rand_iops = (double)n / seconds;
    r.rand_lat_us = n ? (double)ns / 1e3 / (double)n : 0;
}

// sequential 1 MiB reads from a random offset in the first half of the file
void measure_sequential(int fd, int64_t size, double seconds, DiskSpeed& r) {
    const size_t big = 1 << 20;
    void* buf = std::aligned_alloc(4096, big);
    off_t off = (off_t)(std::mt19937_64(7)() % std::max<int64_t>(1, size / 2)) / 4096 * 4096;
    int64_t bytes = 0;
    const auto t0 = Clock::now();
    double dt = 0;
    while (dt < seconds && off + (off_t)big <= size) {
        if (pread(fd, buf, big, off) != (ssize_t)big) break;
        off += big, bytes += big;
        dt = std::chrono::duration<double>(Clock::now() - t0).count();
    }
    std::free(buf);
    r.seq_mb_s = dt > 0 ? (double)bytes / dt / 1e6 : 0;
}

// 2 MiB blocks at random offsets from 16 threads, one read each at a time: the engine's expert reads. The figure of
// the Mac anchor (13 GB/s, research/measurements/2026-11-11-mac-ssd.md) was measured this way.
[[maybe_unused]] void measure_experts(int fd, int64_t size, double seconds, DiskSpeed& r) {
    constexpr size_t kBlock = 2 << 20;
    const int64_t blocks = size / static_cast<int64_t>(kBlock) - 1;
    if (blocks < 16) return;
    std::atomic<int64_t> bytes{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    threads.reserve(16);
    const auto t0 = Clock::now();
    for (int t = 0; t < 16; t++)
        threads.emplace_back([&, t] {
            void* buf = std::aligned_alloc(4096, kBlock);
            std::mt19937_64 rng(t + 101);
            while (!stop) {
                if (pread(fd, buf, kBlock, static_cast<off_t>(rng() % blocks) * static_cast<off_t>(kBlock)) !=
                    static_cast<ssize_t>(kBlock))
                    break;
                bytes += static_cast<int64_t>(kBlock);
            }
            std::free(buf);
        });
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop = true;
    for (auto& t : threads) t.join();
    const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
    r.expert_mb_s = dt > 0 ? static_cast<double>(bytes) / dt / 1e6 : 0;
}

}  // namespace

Disk detect_disk(const std::string& path) {
    Disk d;
    std::string src = mount_source(path);
    if (!src.starts_with("/dev/")) return d;
    std::error_code ec;
    src = fs::canonical(src, ec).filename().string();  // /dev/mapper/x → dm-0
    d.device = to_disk(src);
    const fs::path q = fs::path("/sys/block") / d.device;
    d.model = trim(read_file(q / "device/model"));
    if (d.device.starts_with("nvme"))
        d.kind = "nvme";
    else if (trim(read_file(q / "queue/rotational")) == "1")
        d.kind = "hdd";
    else if (fs::exists(q))
        d.kind = "ssd";
    return d;
}

DiskSpeed measure_disk(const std::string& repo, std::string file, double seconds) {
    DiskSpeed r;
    if (file.empty()) file = largest_model_file(repo + "/work/models");
    if (file.empty()) return r;  // nothing large yet: only the kind of drive is known
#ifdef __APPLE__
    const int fd = open(file.c_str(), O_RDONLY);  // macOS has no O_DIRECT; F_NOCACHE bypasses the page cache
    if (fd >= 0) fcntl(fd, F_NOCACHE, 1);
#else
    const int fd = open(file.c_str(), O_RDONLY | O_DIRECT);
#endif
    if (fd < 0) return r;
    struct stat st{};
    fstat(fd, &st);
    const int64_t blocks = st.st_size / 4096 - 1;
    if (blocks >= 1024) {
#ifdef __APPLE__
        // the same 3 s as on Linux, shared by the three reads
        measure_random(fd, blocks, seconds / 2, r);
        measure_sequential(fd, st.st_size, seconds / 4, r);
        measure_experts(fd, st.st_size, seconds / 4, r);
#else
        measure_random(fd, blocks, seconds * 2 / 3, r);
        measure_sequential(fd, st.st_size, seconds / 3, r);
#endif
        r.measured = r.rand_iops > 0 && r.seq_mb_s > 0;
        r.file = file;
    }
    close(fd);
    return r;
}

}  // namespace tona::install
