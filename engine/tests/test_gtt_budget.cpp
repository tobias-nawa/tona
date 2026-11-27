// gpu/gtt_budget.h: the cap of the pinned experts to the GTT and the check which kind of host memory is in use
#include "gpu/gtt_budget.h"
#include "tests/test_support.h"

int main() {
    using namespace tona::gpu;
    constexpr int64_t G = 1LL << 30, total = 32 * G;
    EXPECT(cap_pin_budget(29 * G, total, 2 * G) == 29 * G);     // enough room: unchanged
    EXPECT(cap_pin_budget(29 * G, total, 6 * G) == 25 * G);     // total - used - margin (1 GiB)
    EXPECT(cap_pin_budget(29 * G, total, total) == 0);          // GTT full
    EXPECT(cap_pin_budget(29 * G, total, total + 5 * G) == 0);  // used above total: never negative
    EXPECT(cap_pin_budget(0, total, 0) == 0);                   // no pinning stays none
    EXPECT(cap_pin_budget(10 * G, total, 6 * G, 0) == 10 * G);  // margin 0: the budget is below the room
    EXPECT(cap_pin_budget(40 * G, total, 6 * G, 0) == 26 * G);

    constexpr int64_t P = 64LL << 20;
    EXPECT(classify_host_memory(5 * G, 5 * G + P, P) == HostMemory::Gtt);
    EXPECT(classify_host_memory(5 * G, 5 * G + P + 4096, P) == HostMemory::Gtt);  // rounding up
    EXPECT(classify_host_memory(5 * G, 5 * G, P) == HostMemory::Userptr);
    EXPECT(classify_host_memory(5 * G, 5 * G - P, P) == HostMemory::Userptr);  // another process freed memory
    EXPECT(classify_host_memory(5 * G, 5 * G + P / 2, P) == HostMemory::Unknown);
    return tona::test::finish("gtt_budget");
}
