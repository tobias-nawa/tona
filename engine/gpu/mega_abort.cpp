// What the CPU reports about an abort of the kernel beyond the step: the cause the aborting block recorded (note_abort)
// and, with TONA_MEGA_ABORT_TRACE, which blocks had reached the barrier of that step (research/measurements/
// 2026-10-30-barrier-aborts.md).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

#include "backend/env.h"
#include "gpu/mega.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

// The wait that counted the abort: how long it waited, against which limit, how far the pass deadline was
std::string Mega::cause_info() const {
    unsigned long long d[7];
    if (hipMemcpy(d, static_cast<uint8_t*>(d_bar_) + BAR_DIAG, sizeof d, hipMemcpyDeviceToHost) != hipSuccess ||
        d[0] == 0)
        return "";
    const double ms = 1.0 / wall_khz_;
    const unsigned kind = (unsigned)(d[0] & 0xff), block = (unsigned)(d[0] >> 8);
    const char* what = (kind & 8) ? "wait for the CPU" : (kind & 4) ? "dependency wait" : "barrier";
    const char* hit = (kind & 3) == 3 ? "limit and deadline" : (kind & 2) ? "pass deadline" : "limit";
    std::string s = std::format(
        ", cause: {} at its {} (block {} waited {:.1f} ms of {:.1f} ms, pass deadline {:+.1f} ms)", what, hit, block,
        (double)(long long)(d[2] - d[1]) * ms, (double)d[4] * ms, (double)(long long)(d[3] - d[2]) * ms);
    if (kind & 8) return s;
    if (kind & 4) return s + std::format(", first missing block {} of mask {:#x}", d[6], d[5]);
    return s + std::format(", {} arrivals counted, generation {}", d[5], d[6]);
}

// TONA_MEGA_GAP_HIST: how often the waits polled and how long the gaps between two polls were (a gap that stands out
// is a wave that did not run)
std::string Mega::gap_hist_info() const {
    unsigned long long h[8];
    if (hipMemcpy(h, static_cast<uint8_t*>(d_bar_) + BAR_BYTES, sizeof h, hipMemcpyDeviceToHost) != hipSuccess ||
        h[0] == 0)
        return "";
    return std::format(
        "gpu::Mega: poll gaps: {} polls; gap 16-64 us {}, -256 us {}, -1 ms {}, -2 ms {}, -10 ms {}, -100 ms {}, "
        "longer {}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
}

// TONA_MEGA_STALL_MS=n (with TONA_MEGA_PROF): one line per step of a pass that took longer than n ms. Counts the stalls
// of the whole grid that stay below the limit of the barrier, to see how often they happen and how long they last.
void Mega::log_slow_steps(const std::vector<unsigned long long>& t) const {
    static const double limit_ms = (double)env::integer("TONA_MEGA_STALL_MS", 0);
    static int passes = 0;
    if (limit_ms <= 0) return;
    if (++passes % 100 == 0)
        if (const std::string h = gap_hist_info(); !h.empty()) std::fprintf(stderr, "%s\n", h.c_str());
    const double ms = 1.0 / wall_khz_;
    unsigned long long prev = t[0];
    for (size_t o = 0; o < ops_.size(); o++) {
        const double dt = (double)(long long)(t[2 * o + 1] - prev) * ms;
        const unsigned long long from = prev;
        prev = t[2 * o + 1];
        if (dt <= limit_ms) continue;
        const auto wall =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        std::fprintf(stderr,
                     "gpu::Mega: step %zu %s took %.1f ms at position %lld (%.1f ms into the pass of %.1f ms, end of "
                     "pass at %lld ms)%s\n",
                     o, labels_[o].c_str(), dt, (long long)pos_, (double)(long long)(prev - t[0]) * ms,
                     (double)(long long)(t[2 * ops_.size() - 1] - t[0]) * ms, (long long)wall,
                     arrival_spread(o, from).c_str());
    }
}

// With the arrival stamps: how the blocks reached the barrier of step o, from the release of the barrier before it
std::string Mega::arrival_spread(size_t o, unsigned long long from) const {
    if (!d_arr_ || pass_nb_ < 1) return "";
    const int nblk = kblocks_[kix(pass_nb_)];
    std::vector<unsigned long long> a(nblk);
    if (hipMemcpy(a.data(), d_arr_ + o * nblk, nblk * sizeof a[0], hipMemcpyDeviceToHost) != hipSuccess) return "";
    std::vector<int> order(nblk);
    for (int b = 0; b < nblk; b++) order[b] = b;
    std::sort(order.begin(), order.end(), [&](int x, int y) { return a[x] < a[y]; });
    const double ms = 1.0 / wall_khz_;
    auto rel = [&](int b) { return (double)(long long)(a[b] - from) * ms; };
    return std::format(
        "; blocks arrive {:.2f} to {:.2f} ms after the previous barrier, median {:.2f}, last block {}, "
        "second last {} at {:.2f}",
        rel(order[0]), rel(order[nblk - 1]), rel(order[nblk / 2]), order[nblk - 1], order[nblk - 2],
        rel(order[nblk - 2]));
}

// TONA_MEGA_ABORT_TRACE: the arrival of every block at the barrier behind step k, relative to the start of the pass.
// A stamp older than the start belongs to an earlier pass: the block had not arrived.
std::string Mega::arrival_info(unsigned k) const {
    if (!d_arr_ || !d_prof_ || k >= ops_.size() || last_launch_.ops != d_ops_) return "";
    const int nblk = kblocks_[kix(last_launch_.nb)];
    std::vector<unsigned long long> a(nblk);
    unsigned long long t0 = 0;
    if (hipMemcpy(&t0, d_prof_, sizeof t0, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(a.data(), d_arr_ + (size_t)k * nblk, nblk * sizeof a[0], hipMemcpyDeviceToHost) != hipSuccess)
        return "";
    const double ms = 1.0 / wall_khz_;
    std::string missing;
    long long lo = -1, hi = -1;
    int arrived = 0;
    for (int b = 0; b < nblk; b++) {
        if (a[b] < t0) {
            missing += std::format(" {}", b);
            continue;
        }
        const long long rel = (long long)(a[b] - t0);
        lo = lo < 0 ? rel : std::min(lo, rel), hi = std::max(hi, rel);
        arrived++;
    }
    std::string s = std::format(", arrivals at step {}: {} of {} blocks", k, arrived, nblk);
    if (arrived) s += std::format(" between {:.2f} and {:.2f} ms after the start", (double)lo * ms, (double)hi * ms);
    return s + (missing.empty() ? "" : ", missing blocks" + missing);
}

}  // namespace tona::gpu
