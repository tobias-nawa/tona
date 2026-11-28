// Probe (TONA_PLE_STATS=1): how long the host copy of a pass's n-gram rows takes and how many pages it reads back from
// disk. The table is a file mapping far larger than the RAM (28.8 GB in the Coder); a token reads 16 rows of 90 bytes
// on 16 random pages, so a row whose page was dropped stalls the pass on the SSD before the GPU can start.
// Off: nothing is timed. On: one stderr line per 1000 decode passes with the interval's means, and the time from one
// decode copy to the next (the pass period, measured, gaps over 1 s between requests left out).
// The fault count is not the wait: after a read-ahead (TONA_PLE_PREFETCH) the copy still counts each page as a major
// fault, but the page is already on its way (33 faults and 0.37 ms per 35 cold pages against 33 and 4.7 ms without).
#pragma once

#ifndef _WIN32
#include <sys/resource.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>

#include "backend/env.h"

namespace tona::backend {

class PleStats {
public:
    static bool enabled() {
        static const bool on = env::flag("TONA_PLE_STATS");
        return on;
    }

    void refused(int n) { decode_.failed += n; }  // read-ahead requests for a later pass the kernel refused

    // Times one copy from construction to destruction; faults are the whole process's (other threads included)
    class Copy {
    public:
        Copy(PleStats& s, bool prefill, int tokens) : s_(s), prefill_(prefill), tokens_(tokens) {
            if (enabled()) t0_ = std::chrono::steady_clock::now(), f0_ = faults();
        }
        ~Copy() {
            if (!enabled()) return;
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0_).count();
            s_.add(prefill_, tokens_, us, faults() - f0_, failed_);
            if (!prefill_) s_.period(t0_);
        }
        void failed(int n) { failed_ += n; }  // read-ahead requests the kernel refused
        Copy(const Copy&) = delete;
        Copy& operator=(const Copy&) = delete;

    private:
        PleStats& s_;
        bool prefill_;
        int tokens_;
        std::chrono::steady_clock::time_point t0_{};
        int64_t f0_ = 0;
        int failed_ = 0;
    };

private:
    struct Sum {
        int64_t passes = 0, tokens = 0, faults = 0, failed = 0;
        double us = 0;
    };
    using Clock = std::chrono::steady_clock;
    static constexpr int64_t kEvery = 1000;  // decode passes per line: about 30 s of decode

    static int64_t faults() {
#ifdef _WIN32
        return 0;  // Win32 has no per-process major-fault counter matching getrusage(RUSAGE_SELF)
#else
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        return ru.ru_majflt;
#endif
    }

    void period(Clock::time_point t0) {
        const double us = std::chrono::duration<double, std::micro>(t0 - last_).count();
        if (last_ != Clock::time_point{} && us < 1e6) periods_++, period_us_ += us;
        last_ = t0;
    }

    void add(bool prefill, int tokens, double us, int64_t faults, int failed) {
        Sum& s = prefill ? prefill_ : decode_;
        s.passes++, s.tokens += tokens, s.us += us, s.faults += faults, s.failed += failed;
        if (decode_.passes < kEvery) return;
        const auto mean = [](const Sum& x, double v) { return x.passes ? v / (double)x.passes : 0.0; };
        std::fprintf(stderr,
                     "PLE copy: decode %lld passes, %.1f us and %.2f major faults per pass (%.2f tokens), %.0f us "
                     "from pass to pass; prefill %lld passes, %.0f us and %.1f faults per pass (%.0f tokens); %lld "
                     "read-ahead requests refused\n",
                     (long long)decode_.passes, mean(decode_, decode_.us), mean(decode_, (double)decode_.faults),
                     mean(decode_, (double)decode_.tokens), periods_ ? period_us_ / (double)periods_ : 0.0,
                     (long long)prefill_.passes, mean(prefill_, prefill_.us), mean(prefill_, (double)prefill_.faults),
                     mean(prefill_, (double)prefill_.tokens), (long long)decode_.failed + (long long)prefill_.failed);
        decode_ = {}, prefill_ = {}, periods_ = 0, period_us_ = 0;
    }

    Sum decode_, prefill_;
    Clock::time_point last_{};  // start of the previous decode copy
    int64_t periods_ = 0;
    double period_us_ = 0;
};

}  // namespace tona::backend
