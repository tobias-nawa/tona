// Memory limits of the engine, the same for all backends. TONA_RAM_GB: how much RAM the engine may hold (pinned
// experts, KV in RAM, sequence buffers, stashed sequences; not the page cache of the model file, which the kernel
// releases when needed). TONA_VRAM_GB: how much VRAM it may use (instead of "everything except TONA_VRAM_RESERVE_MB").
// Without a value the previous behavior applies. Finer switches (TONA_PIN_GB, TONA_PROMPT_RAM_GB) take precedence.
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "backend/env.h"
#if defined(__APPLE__)
#include "backend/mem_macos.h"
#elif defined(_WIN32)
#include "backend/mem_win32.h"
#endif

namespace tona::backend {

// Value of an environment variable in GB, < 0: not set
inline double env_gb(const char* name) { return env::number(name, -1.0); }

inline double ram_limit_gb() {
    static const double v = env_gb("TONA_RAM_GB");
    return v;
}

inline double vram_limit_gb() {
    static const double v = env_gb("TONA_VRAM_GB");
    return v;
}

// With TONA_RAM_GB the experts are pinned all or not at all: all of them if they fit the limit together with
// CACHE_MIN_GB of page cache (PLE rows, reloading), otherwise none. Pinning partially was the worst choice at tight
// limits: pins and page cache evict each other, page faults wait for the cgroup to free memory, passes abort
// (20 GB: 15.6 GB pinned gave 2 aborts, nothing pinned gave 43.7 tok/s decode;
// research/measurements/2026-10-27-ram-limit.md). It ran well with >= 4.3 GB next to pins and fixed RAM and badly
// below 3.5 GB: pinning all 25.2 GB starts at a limit of ~31.6 GB.
// Headroom for the rest of the process and the minimum for stashed sequences:
constexpr double RAM_SLACK_GB = 0.5, PROMPT_MIN_GB = 1.0, CACHE_MIN_GB = 3.0;
// Page cache kept next to partial pins (the tiers of backend/expert_tiers.h, the Metal arena): 1.5 GB, decided by the
// floor sweep 3.0 / 1.5 / 0.5 GB (0.5 gave 2 to 7 % more decode at the cost of the user's room;
// research/measurements/2026-11-15-expert-tiers.md)
constexpr double TIERED_CACHE_MIN_GB = 1.5;

// RAM the process already holds (anonymous and shared, e.g. pinned buffers), in bytes
inline long long ram_held_bytes();

// RAM for stashed sequences: TONA_PROMPT_RAM_GB, otherwise 12 GB; with TONA_RAM_GB what is left after the backend has
// loaded, at most a quarter of the limit (the rest stays page cache), 1 to 12 GB. Call only after the backend has
// been created
inline double prompt_ram_gb() {
    if (const double v = env_gb("TONA_PROMPT_RAM_GB"); v >= 0) return v;
    if (ram_limit_gb() < 0) return 12.0;
    const double left = ram_limit_gb() - RAM_SLACK_GB - ram_held_bytes() / 1e9;
    return std::clamp(std::min(left, ram_limit_gb() / 4), PROMPT_MIN_GB, 12.0);
}

// Part of it to keep free already when pinning the experts: TONA_PROMPT_RAM_GB or the minimum
inline double prompt_reserve_gb() {
    const double v = env_gb("TONA_PROMPT_RAM_GB");
    return v >= 0 ? v : PROMPT_MIN_GB;
}

inline long long ram_held_bytes() {
#if defined(__APPLE__)
    return macos::held_bytes();
#elif defined(_WIN32)
    return win32::held_bytes();
#else
    long long anon = 0, shmem = 0;
    if (FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f)) {
            if (std::strncmp(line, "RssAnon:", 8) == 0) anon = std::strtoll(line + 8, nullptr, 10);
            if (std::strncmp(line, "RssShmem:", 9) == 0) shmem = std::strtoll(line + 9, nullptr, 10);
        }
        std::fclose(f);
    }
    return (anon + shmem) * 1024;
#endif
}

// Value of "key value" in a /proc or cgroup stat text, -1 if the key is missing
inline long long stat_value(const std::string& text, const char* key) {
    const std::string k = std::string(key) + " ";
    for (size_t at = 0; (at = text.find(k, at)) != std::string::npos; at += k.size())
        if (at == 0 || text[at - 1] == '\n') return std::strtoll(text.c_str() + at + k.size(), nullptr, 10);
    return -1;
}

inline std::string read_text(const std::string& path) {
    std::string out;
    if (FILE* f = std::fopen(path.c_str(), "r")) {
        char buf[4096];
        size_t n = 0;
        do {  // a short read is the end or an error: no read after it
            n = std::fread(buf, 1, sizeof buf, f);
            out.append(buf, n);
        } while (n == sizeof buf);
        std::fclose(f);
    }
    return out;
}

// Room left in one cgroup (v2) below its memory.max, counting only what the kernel cannot take back (anonymous,
// shared and kernel memory; page cache is reclaimable). < 0: no limit set
inline double cgroup_headroom_bytes(const std::string& dir) {
    const std::string max = read_text(dir + "/memory.max");
    if (max.empty() || max[0] == 'm') return -1;  // "max": unlimited
    const std::string st = read_text(dir + "/memory.stat");
    long long used = 0;
    for (const char* k : {"anon", "shmem", "kernel"}) used += std::max(0LL, stat_value(st, k));
    return (double)std::strtoll(max.c_str(), nullptr, 10) - (double)used;
}

// RAM that can still be taken without the OOM killer: the smallest room of /proc/meminfo (MemAvailable) and of every
// cgroup from ours up to the root (systemd-run -p MemoryMax=... sets the limit on a scope)
inline double free_ram_bytes() {
#if defined(__APPLE__)
    return macos::available_bytes();  // no cgroups on macOS
#elif defined(_WIN32)
    return win32::available_bytes();
#else
    double room = std::max(0LL, stat_value(read_text("/proc/meminfo"), "MemAvailable:")) * 1024.0;
    std::string path = read_text("/proc/self/cgroup");
    const size_t at = path.find("0::");
    if (at == std::string::npos) return room;
    path = path.substr(at + 3, path.find('\n', at) - at - 3);
    while (!path.empty()) {
        const double h = cgroup_headroom_bytes("/sys/fs/cgroup" + path);
        if (h >= 0) room = std::min(room, h);
        path.resize(path.rfind('/'));
    }
    return room;
#endif
}

// RAM that must stay free after a stash is copied (host MemAvailable and the cgroup headroom alike). Measured, not
// derived: with 5.0 GB left after the copy the engine thread hung for 48 min in hipMemcpy of the stash (--ctx 540672,
// 2 sequence slots, MemoryMax 54G: memory.current 52.9 of 58 GB, anon 9.5 + shmem 43.2, MemAvailable 5 GB, 0.5 GB
// free, no kernel message; research/measurements/2026-10-31-long-context-topk.md). The cause is not established;
// 8 GB is twice the 4 GB first guess, which the hang case (5.0 GB left) would still have passed.
constexpr double STASH_RESERVE_GB = 8.0;

// TONA_TEST_STASH_RESERVE_GB lowers the reserve to reproduce the stash-copy hang; not for production
inline double stash_reserve_gb() {
    static const double v =
        env_gb("TONA_TEST_STASH_RESERVE_GB") >= 0 ? env_gb("TONA_TEST_STASH_RESERVE_GB") : STASH_RESERVE_GB;
    return v;
}

// Does a stash of `bytes` fit? Only if it fits the budget alone (`cap_gb`, TONA_PROMPT_RAM_GB or derived) and the free
// RAM leaves STASH_RESERVE_GB after the copy, counting `evictable` bytes of older stashes that would be dropped
// first. Pinned memory (shmem in the cgroup) is never free: free_ram_bytes counts it as used. Without the check the
// copy was made first and trimmed afterwards: a 1M conversation (16.8 GB) on top of 45 GB of pinned memory was
// OOM-killed although TONA_PROMPT_RAM_GB=0.
inline bool stash_fits(double bytes, double cap_gb, double free_bytes, double evictable) {
    return bytes <= cap_gb * 1e9 && bytes + stash_reserve_gb() * 1e9 <= free_bytes + evictable;
}

}  // namespace tona::backend
