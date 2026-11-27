// backend/runtime_env.h: when does the process restart itself, and with which variables?
#include "backend/runtime_env.h"
#include "tests/test_support.h"

static void test_restart_plan() {
    using tona::runtime_env::plan;
    auto p = plan(nullptr, nullptr, nullptr);  // nothing chosen: restart with both
    EXPECT(p.omp_policy && p.userptr && p.any());
    p = plan("", "", "");  // empty counts as unset
    EXPECT(p.omp_policy && p.userptr);
    p = plan("passive", nullptr, nullptr);  // the user's policy wins, also the one after our own restart
    EXPECT(!p.omp_policy && p.userptr);
    p = plan("active", nullptr, nullptr);
    EXPECT(!p.omp_policy && p.userptr);
    p = plan(nullptr, "1000", nullptr);  // an explicit spin count wins too
    EXPECT(!p.omp_policy && p.userptr);
    p = plan(nullptr, nullptr, "1");  // the user wants userptr: left as it is
    EXPECT(p.omp_policy && !p.userptr);
    p = plan(nullptr, nullptr, "0");
    EXPECT(p.omp_policy && !p.userptr);
    p = plan("passive", "1000", "0");  // after our own restart: nothing left to do, no loop
    EXPECT(!p.any());
    p = plan("passive", nullptr, "1");
    EXPECT(!p.any());
}

static void test_thread_count() {
    using tona::runtime_env::plan;
    // OMP_NUM_THREADS: the physical cores unless the user (or a launcher, or our own restart) set it
    using tona::runtime_env::physical_cores;
    EXPECT(physical_cores(32, 1) == 16);  // Ryzen 9 9950X with SMT
    EXPECT(physical_cores(12, 1) == 6);   // Ryzen 5 9600X with SMT
    EXPECT(physical_cores(16, 0) == 16);  // SMT off: every logical CPU is a core
    EXPECT(physical_cores(1, 1) == 1 && physical_cores(0, 1) == 1 && physical_cores(0, 0) == 1);
    auto p = plan("passive", "", "0", nullptr, 16);
    EXPECT(p.omp_threads == 16 && p.any() && !p.omp_policy && !p.userptr);
    p = plan("passive", "", "0", "", 16);  // empty counts as unset
    EXPECT(p.omp_threads == 16);
    p = plan("passive", "", "0", "6", 16);  // a launcher's or the user's value stays
    EXPECT(p.omp_threads == 0 && !p.any());
    p = plan("passive", "", "0", nullptr, 0);  // cores unknown (macOS): leave it
    EXPECT(p.omp_threads == 0 && !p.any());
    p = plan(nullptr, nullptr, nullptr, "16", 16);  // after our own restart: nothing left, no loop
    EXPECT(p.omp_threads == 0 && p.omp_policy && p.userptr);
}

int main() {
    test_restart_plan();
    test_thread_count();
    return tona::test::finish("runtime_env");
}
