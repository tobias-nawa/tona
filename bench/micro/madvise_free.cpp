// Can data in MADV_FREE memory be trusted after a copy? Needed for the stash design of
// docs/architecture/expert-tiers.md section 5b: the kernel may replace a lazily freed page by a zero page at any time
// without notice. The rule tested: COPY the page first, then look at the magic word of the SOURCE page; a freed page
// stays zero until written, so a magic word still present after the copy proves that the copied bytes were valid.
// Checking first and copying after is racy. CPU only, no GPU. Build: g++ -O2 -std=c++20 -pthread madvise_free.cpp -o
// madvise_free Run (pressure comes from the cgroup limit, swap off so that nothing else can be reclaimed):
//   systemd-run --user --scope -q -p MemoryMax=300M -p MemorySwapMax=0 ./madvise_free
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

constexpr size_t PAGE = 4096, PAGES = 32768;  // 128 MiB
constexpr uint64_t MAGIC = 0x7A55AA3C5A7E1D01ull;

static uint8_t* region;
static uint8_t byte_of(size_t page, size_t i) { return (uint8_t)((page * 131 + i * 7 + 1) | 1); }  // never zero

static void fill(size_t p) {
    uint8_t* a = region + p * PAGE;
    for (size_t i = 8; i < PAGE; i++) a[i] = byte_of(p, i);
    std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t*>(a)).store(MAGIC ^ p, std::memory_order_release);
}
static bool magic_ok(size_t p) {
    return std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t*>(region + p * PAGE)).load(std::memory_order_acquire) ==
           (MAGIC ^ p);
}
static bool bytes_ok(const uint8_t* c, size_t p) {
    for (size_t i = 8; i < PAGE; i++)
        if (c[i] != byte_of(p, i)) return false;
    return true;
}
// Copy first, then check the source's magic word
static bool copy_then_check(size_t p, uint8_t* out) {
    std::memcpy(out, region + p * PAGE, PAGE);
    return magic_ok(p);
}
static bool check_then_copy(size_t p, uint8_t* out) {
    if (!magic_ok(p)) return false;
    std::memcpy(out, region + p * PAGE, PAGE);
    return true;
}
static size_t resident(size_t from, size_t n) {
    std::vector<unsigned char> v(n);
    if (mincore(region + from * PAGE, n * PAGE, v.data()) != 0) return 0;
    size_t r = 0;
    for (unsigned char c : v) r += c & 1;
    return r;
}

struct Counts {
    size_t good = 0, lost = 0, wrong_order_bad = 0, invariant_broken = 0, reads = 0;
};

// Another mapping is touched while a reader copies the freed pages in both orders
static Counts under_pressure(std::vector<uint8_t>& out) {
    Counts c;
    std::atomic<bool> stop{false};
    size_t big = 200ull << 20;  // region 128 MiB + this exceeds the 300 MiB limit: the freed pages must give way
    std::thread pressure([&] {
        uint8_t* b = (uint8_t*)mmap(nullptr, big, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        for (size_t o = 0; o < big; o += PAGE) b[o] = 1, usleep(o % (64 * PAGE) == 0 ? 200 : 0);
        stop = true;
    });
    while (!stop) {
        for (size_t p = 0; p < PAGES; p++) {
            const bool a = copy_then_check(p, out.data());
            if (a)
                c.good++, c.invariant_broken += !bytes_ok(out.data(), p);
            else
                c.lost++;
            c.reads++;
            if (p % 64 == 0 && check_then_copy(p, out.data()) && !bytes_ok(out.data(), p)) c.wrong_order_bad++;
        }
    }
    pressure.join();
    return c;
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    region = (uint8_t*)mmap(nullptr, PAGES * PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (size_t p = 0; p < PAGES; p++) fill(p);
    std::printf("filled %zu pages, resident per mincore %zu\n", PAGES, resident(0, PAGES));
    if (madvise(region, PAGES * PAGE, MADV_FREE) != 0) return std::perror("MADV_FREE"), 1;
    std::vector<uint8_t> out(PAGE);
    size_t ok = 0;
    for (size_t p = 0; p < PAGES; p++) ok += copy_then_check(p, out.data()) && bytes_ok(out.data(), p);
    std::printf("E1 no pressure after MADV_FREE: %zu of %zu pages valid, resident per mincore %zu\n", ok, PAGES,
                resident(0, PAGES));
    fill(5);  // a write cancels the free of that page
    const Counts c = under_pressure(out);
    size_t zero_read = 0;  // what a freed page looks like when read (no fault, zeros)
    for (size_t p = 0; p < PAGES; p++) zero_read += (!magic_ok(p) && *(volatile uint64_t*)(region + p * PAGE) == 0);
    std::printf(
        "E2 under pressure: %zu reads, %zu valid by copy-then-check, %zu lost; valid but wrong bytes (must be 0): "
        "%zu\n",
        c.reads, c.good, c.lost, c.invariant_broken);
    std::printf("E3 check-then-copy: copies that passed the check but have wrong bytes: %zu (a race, may be 0)\n",
                c.wrong_order_bad);
    std::printf(
        "E4 freed pages read as zero without a fault: %zu; page 5 (written after the advice) still valid: %d; resident "
        "per mincore now %zu\n",
        zero_read, (int)magic_ok(5), resident(0, PAGES));
    return c.invariant_broken == 0 ? 0 : 2;
}
