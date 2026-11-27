// The elastic slot pool follows the free VRAM (TONA_VRAM_GOVERNOR=1, docs/architecture/expert-tiers.md section 5a).
// Runs on the engine thread at the start of a pass, when no kernel is in flight: hipMemUnmap waits until the device is
// idle, so it must never be called while k_mega runs or waits for the CPU.
#include <cstdio>

#include "backend/env.h"
#include "gpu/mega.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

bool Mega::vram_tick() {
    ElasticPool& P = elastic_;
    if (!P.on() || !P.governing()) return false;
    const double now = P.now();
    if (!P.due(now)) return true;
    // A pass, a prefill loan or an uncommitted pass in flight: not now (the next tick is 250 ms later). hipStreamQuery
    // does not block. The stream is the one k_mega runs on.
    if (!pf_loans_.empty() || awaiting_commit_ || hipStreamQuery(nullptr) != hipSuccess) return true;
    const int64_t free_bytes = P.fake_on() ? P.fake_free(now) : vram_free_sysfs();
    backend::VramGovernor& gov = P.governor();
    const int n = gov.sample(free_bytes, now, false);
    static bool logged = false;
    if (gov.rebased() && !logged) {
        logged = true;
        std::fprintf(stderr, "VRAM governor: the first sample (%.0f MB free) was below the shrink mark, rebased\n",
                     free_bytes / 1e6);
    }
    if (n == 0) return true;
    prefetch_drain();  // no copy into a slot may be in flight
    P.step(cache_, n, now, free_bytes);
    return true;
}

}  // namespace tona::gpu
