// backend/cpu_cost.h: the planner's estimate of a CPU expert follows a slowdown of the CPU, stays at the base without
// one, comes back when the load ends and does not flap in between.
#include <vector>

#include "backend/cpu_cost.h"
#include "tests/test_support.h"

namespace {

using tona::backend::CpuCost;

// One layer as in ExpertSelect::split_misses: `hits` experts in VRAM, `misses` pinned misses, round robin between the
// GPU (15 us each) and the CPU, `unpinned` more misses that must run on the CPU. Returns the number of pinned experts
// on the CPU and updates the estimate like Mega does: a sample only if all CPU experts are pinned, a decay if no pinned
// expert is on the CPU but some on the GPU.
int layer(CpuCost& cost, int hits, int misses, double true_us_per_expert, int unpinned = 0) {
    const double cpu_us = cost.us();
    double t_gpu = hits * 12.0, t_cpu = unpinned * cpu_us;  // the unpinned ones are counted first, as in Mega
    int on_cpu = 0, on_gpu = 0;
    for (int i = 0; i < misses; i++) {
        if (tona::backend::to_gpu(t_gpu, t_cpu, 15, cpu_us))
            on_gpu++, t_gpu += 15;
        else
            on_cpu++, t_cpu += cpu_us;
    }
    if (on_cpu > 0 && unpinned == 0)
        cost.observe(on_cpu, on_cpu * true_us_per_expert);
    else if (on_cpu == 0 && on_gpu > 0)
        cost.skipped();
    return on_cpu;
}

void test_dead_zone() {
    CpuCost c(60, true);
    EXPECT(c.us() == 60);
    for (int i = 0; i < 1000; i++) c.observe(5, 5 * 80.0);  // the normal state: 76-80 us per expert
    EXPECT(c.us() == 60);
    for (int i = 0; i < 1000; i++) c.observe(5, 5 * 115.0);  // still inside the dead zone (below 2 x base)
    EXPECT(c.us() == 60);
}

void test_follows_slowdown() {
    CpuCost c(60, true);
    for (int i = 0; i < 200; i++) c.observe(4, 4 * 800.0);
    EXPECT(c.us() > 300 && c.us() <= 60 * tona::backend::CpuCost::kMax);
    for (int i = 0; i < 5000; i++) c.skipped();
    EXPECT(c.us() == 60);  // without samples the estimate falls back
}

void test_fixed_plan() {
    CpuCost c(60, false);  // TONA_DETERMINISTIC, TONA_PLAN, TONA_CPU_ADAPT=0
    for (int i = 0; i < 100; i++) c.observe(4, 4 * 800.0), c.skipped();
    EXPECT(c.us() == 60 && c.ratio() == 1);
}

// 500 layers normal, 1000 under load (10x), 500 normal again
void test_closed_loop() {
    CpuCost c(60, true);
    std::vector<int> n_cpu;
    for (int i = 0; i < 2000; i++) {
        const bool load = i >= 500 && i < 1500;
        n_cpu.push_back(layer(c, 10, 3, load ? 800.0 : 80.0));
    }
    for (int i = 0; i < 500; i++) EXPECT(n_cpu[i] == 2);  // no change of the plan without load
    int probes = 0, slow_experts = 0;
    for (int i = 600; i < 1500; i++) probes += n_cpu[i] > 0, slow_experts += n_cpu[i];
    EXPECT(probes < 60);  // under load the CPU is only probed: about one layer in seventeen
    EXPECT(slow_experts < 60);
    int back = 1500;
    while (back < 2000 && n_cpu[back] != 2) back++;
    EXPECT(back < 1500 + 700);  // back to the normal plan within a few passes of 40 layers
    int flaps = 0;
    for (int i = back + 1; i < 2000; i++) flaps += n_cpu[i] != n_cpu[i - 1];
    EXPECT(flaps == 0);
}

// A load burst raises the estimate; from then on every layer has an unpinned miss that must run on the CPU. Without
// samples the estimate must still come back, so that the pinned misses are shared again (TONA_PLAN=gpu costs 7 %).
void test_unpinned_every_layer() {
    CpuCost c(60, true);
    std::vector<int> n_cpu(1500);
    for (int i = 0; i < 40; i++) layer(c, 10, 3, 800.0);  // a burst of one pass, all pinned: the estimate rises
    EXPECT(c.us() > 120);
    for (int i = 0; i < 1500; i++) n_cpu[i] = layer(c, 10, 3, 800.0, 1);  // load goes on, unpinned miss each
    CpuCost fixed(60, false);
    const int normal = layer(fixed, 10, 3, 80.0, 1);  // the plan without adaptation: one pinned expert on the CPU
    EXPECT(normal == 1);
    int back = 0;
    while (back < 1500 && n_cpu[back] != normal) back++;
    EXPECT(back < 500);  // the normal plan returns within about 10 passes of 40 layers
    int flaps = 0;
    for (int i = back + 1; i < 1500; i++) flaps += n_cpu[i] != n_cpu[i - 1];
    EXPECT(flaps == 0);
}

}  // namespace

int main() {
    return tona::test::guard("cpu_cost", [&] {
        test_dead_zone();
        test_follows_slowdown();
        test_fixed_plan();
        test_closed_loop();
        test_unpinned_every_layer();
        return tona::test::finish("cpu_cost");
    });
}
