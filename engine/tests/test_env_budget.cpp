// backend/env.h (typed TONA_* variables), backend/idx_share.h (TONA_IDX_SHARE) and backend/budget.h (RAM and VRAM
// limits). The limits are read once per process, so the program takes the scenario as its argument: `limited`
// (TONA_RAM_GB=20, TONA_VRAM_GB=9.5) or `unlimited`.
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "backend/budget.h"
#include "backend/env.h"
#include "backend/idx_share.h"
#include "backend/priority.h"
#include "tests/test_support.h"

namespace {

constexpr const char* VAR = "TONA_TEST_VALUE";

void set(const char* v) {
    if (v)
        setenv(VAR, v, 1);
    else
        unsetenv(VAR);
}

void test_text_and_flag() {
    set(nullptr);
    EXPECT(tona::env::text(VAR) == nullptr && !tona::env::flag(VAR));
    set("");
    EXPECT(tona::env::text(VAR) == nullptr && !tona::env::flag(VAR));  // empty counts as unset
    set("abc");
    EXPECT(std::string(tona::env::text(VAR)) == "abc" && tona::env::flag(VAR));
    set("0");
    EXPECT(!tona::env::flag(VAR));
    set("1");
    EXPECT(tona::env::flag(VAR));
    set("false");
    EXPECT(tona::env::flag(VAR));  // only "0" switches off
}

void test_integer() {
    set(nullptr);
    EXPECT(tona::env::integer(VAR, 7) == 7);
    set("42");
    EXPECT(tona::env::integer(VAR, 7) == 42);
    set("-3");
    EXPECT(tona::env::integer(VAR, 7) == -3);
    set("9000000000");
    EXPECT(tona::env::integer(VAR, 7) == 9000000000LL);
    for (const char* bad : {"4x", "x4", "1.5", " 5", "0x10", "99999999999999999999"}) {
        set(bad);
        EXPECT(tona::env::integer(VAR, 7) == 7);  // never a silent 0
    }
}

void test_number() {
    set(nullptr);
    EXPECT(tona::env::number(VAR, 2.5) == 2.5);
    set("1.25");
    EXPECT(tona::env::number(VAR, 0) == 1.25);
    set("1e3");
    EXPECT(tona::env::number(VAR, 0) == 1000.0);
    set("-0.5");
    EXPECT(tona::env::number(VAR, 0) == -0.5);
    for (const char* bad : {"abc", "1.5x", "", "--1", " 1", "+1", "0x10"}) {  // what from_chars refuses, everywhere
        set(bad);
        EXPECT(tona::env::number(VAR, 9) == 9);
    }
    // read_double takes a prefix as from_chars does: "8.0 GT/s" ends after "8.0", "0x10" after the "0"
    double v = 0;
    const char* speed = "8.0 GT/s";
    EXPECT(tona::env::read_double(speed, speed + 8, v) == speed + 3 && v == 8.0);
    const char* hex = "0x10";
    EXPECT(tona::env::read_double(hex, hex + 4, v) == hex + 1 && v == 0.0);
    EXPECT(tona::env::read_double(speed, speed, v) == nullptr);
}

void test_idx_share() {
    const std::vector<bool> every_second{true, false, true, false, true, false, true, false, true, false, true, false};
    unsetenv("TONA_IDX_SHARE");
    EXPECT(tona::env::idx_full_layers(12) == every_second);
    setenv("TONA_IDX_SHARE", "all", 1);
    EXPECT(tona::env::idx_full_layers(12) == std::vector<bool>(12, true));
    setenv("TONA_IDX_SHARE", "3,9", 1);  // layer 0 is always full
    const std::vector<bool> listed{true, false, false, true, false, false, false, false, false, true, false, false};
    EXPECT(tona::env::idx_full_layers(12) == listed);
    unsetenv("TONA_IDX_SHARE");
}

void test_ram_held() {
    const long long before = tona::backend::ram_held_bytes();
    EXPECT(before > 0);
    std::vector<char> big(64 << 20, 1);               // touched, so the pages count
    asm volatile("" : : "r"(big.data()) : "memory");  // the compiler must not drop the allocation
    const long long after = tona::backend::ram_held_bytes();
    asm volatile("" : : "r"(big.data()) : "memory");
    if (after < before + (48 << 20))  // the one failure seen was not reproducible; print what the kernel reported
        std::fprintf(stderr, "ram_held: before %lld after %lld\n", before, after);
    EXPECT(after >= before + (48 << 20));
}

void test_limited() {
    EXPECT(tona::backend::ram_limit_gb() == 20.0 && tona::backend::vram_limit_gb() == 9.5);
    unsetenv("TONA_PROMPT_RAM_GB");
    // A quarter of the limit when plenty is left
    EXPECT(tona::backend::prompt_ram_gb() == 5.0);
    EXPECT(tona::backend::prompt_reserve_gb() == tona::backend::PROMPT_MIN_GB);
    setenv("TONA_PROMPT_RAM_GB", "3.5", 1);
    EXPECT(tona::backend::prompt_ram_gb() == 3.5 &&
           tona::backend::prompt_reserve_gb() == 3.5);  // the finer switch wins
    setenv("TONA_PROMPT_RAM_GB", "0", 1);
    EXPECT(tona::backend::prompt_ram_gb() == 0.0);
}

void test_unlimited() {
    EXPECT(tona::backend::ram_limit_gb() < 0 && tona::backend::vram_limit_gb() < 0);
    unsetenv("TONA_PROMPT_RAM_GB");
    EXPECT(tona::backend::prompt_ram_gb() == 12.0 &&
           tona::backend::prompt_reserve_gb() == tona::backend::PROMPT_MIN_GB);
    EXPECT(tona::backend::env_gb("TONA_SURELY_UNSET") < 0);
}

// TONA_PRIORITY: only "tona" gives the reserve's pass room away, and only without TONA_RAM_GB; unset and unknown
// text count as balanced
void test_priority(bool limited) {
    using tona::backend::Priority;
    unsetenv("TONA_PRIORITY");
    EXPECT(tona::backend::priority() == Priority::Balanced);
    setenv("TONA_PRIORITY", "tona", 1);
    EXPECT(tona::backend::priority() == Priority::Tona &&
           tona::backend::pass_room_gb(Priority::Tona) == (limited ? tona::backend::TIERED_CACHE_MIN_GB : 0.0));
    setenv("TONA_PRIORITY", "work", 1);
    EXPECT(tona::backend::priority() == Priority::Work &&
           tona::backend::pass_room_gb(Priority::Work) == tona::backend::TIERED_CACHE_MIN_GB);
    setenv("TONA_PRIORITY", "Tona", 1);  // the installer writes lower case; anything else keeps the memory
    EXPECT(tona::backend::priority() == Priority::Balanced &&
           tona::backend::pass_room_gb(Priority::Balanced) == tona::backend::TIERED_CACHE_MIN_GB);
    unsetenv("TONA_PRIORITY");
}

// The stash rule: it must fit the budget alone and the free RAM (plus what eviction frees), with the slack left over
void test_stash_fits() {
    constexpr double GB = 1e9;
    EXPECT(tona::backend::stash_fits(1 * GB, 12, 40 * GB, 0));
    EXPECT(!tona::backend::stash_fits(1, 0, 40 * GB, 0));            // TONA_PROMPT_RAM_GB=0: nothing is stashed
    EXPECT(!tona::backend::stash_fits(14.5 * GB, 12, 40 * GB, 0));   // larger than the budget, plenty of RAM
    EXPECT(!tona::backend::stash_fits(7 * GB, 12, 10 * GB, 0));      // within the budget, but the reserve is missing
    EXPECT(tona::backend::stash_fits(7 * GB, 12, 10 * GB, 6 * GB));  // eviction of older ones makes room
    const double edge = 20 * GB - tona::backend::STASH_RESERVE_GB * GB;
    EXPECT(tona::backend::stash_fits(edge, 12, 20 * GB, 0) && !tona::backend::stash_fits(edge + 1, 12, 20 * GB, 0));
}

// The server that hung for 48 min in the stash (512K, 2 slots, MemoryMax 54G): the numbers of the cgroup and the
// host before and after the copy of 7.6 GB. Neither state may allow it.
void test_hang_case() {
    namespace fs = std::filesystem;
    // One folder per process: ctest -j runs env_budget and env_budget_unlimited (the same binary) at the same time
    const fs::path dir = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
                         ("tona-test-hang-" + std::to_string(::getpid()));
    fs::create_directories(dir);
    std::ofstream(dir / "memory.max") << "57982058496\n";
    std::ofstream(dir / "memory.stat") << "anon 2000000000\nfile 43171000320\nshmem 43170516992\nkernel 214003712\n";
    const double before = tona::backend::cgroup_headroom_bytes(dir.string());  // 12.6 GB; host MemAvailable then ~12.7
    EXPECT(before > 12.5e9 && before < 12.7e9);
    EXPECT(!tona::backend::stash_fits(7.6e9, 12, std::min(before, 12.7e9), 0));
    std::ofstream(dir / "memory.stat") << "anon 9502650368\nfile 43171000320\nshmem 43170516992\nkernel 214003712\n";
    const double after = tona::backend::cgroup_headroom_bytes(dir.string());  // 5.1 GB
    EXPECT(after > 5.0e9 && after < 5.2e9);
    EXPECT(!tona::backend::stash_fits(7.6e9, 12, std::min(after, 5.0e9), 0));
    fs::remove_all(dir);
}

void test_stat_and_cgroup() {
    EXPECT(tona::backend::stat_value("a 1\nanon 7\nxanon 9\n", "anon") == 7);  // whole key, at the start of a line
    EXPECT(tona::backend::stat_value("a 1\n", "anon") == -1);
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") / "tona-test-cgroup";
    fs::create_directories(dir);
    std::ofstream(dir / "memory.max") << "1000000\n";
    std::ofstream(dir / "memory.stat") << "anon 300000\nfile 900000\nshmem 100000\nkernel 50000\n";
    EXPECT(tona::backend::cgroup_headroom_bytes(dir.string()) == 1000000 - 450000);  // page cache does not count
    std::ofstream(dir / "memory.max") << "max\n";
    EXPECT(tona::backend::cgroup_headroom_bytes(dir.string()) < 0);
    fs::remove_all(dir);
    EXPECT(tona::backend::cgroup_headroom_bytes("/nonexistent") < 0);
    EXPECT(tona::backend::free_ram_bytes() > 0);
}

}  // namespace

int main(int argc, char** argv) {
    const bool limited = argc > 1 && std::strcmp(argv[1], "limited") == 0;
    return tona::test::run("test_env_budget", [limited] {
        unsetenv("TONA_RAM_GB"), unsetenv("TONA_VRAM_GB");
        if (limited) setenv("TONA_RAM_GB", "20", 1), setenv("TONA_VRAM_GB", "9.5", 1);
        test_text_and_flag();
        test_integer();
        test_number();
        test_idx_share();
        test_ram_held();
        test_stash_fits();
        test_priority(limited);
        test_hang_case();
        test_stat_and_cgroup();
        limited ? test_limited() : test_unlimited();
    });
}
