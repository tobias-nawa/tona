// TONA_HOST_TRACE=<file>: timestamps of the host side of a decode cycle (draft steps, launches, the wait for the GPU at
// each router, selection, logits, verification), one line "<ns> <mark> <arg>" each. For the breakdown of the cycle that
// the GPU profile (TONA_MEGA_PROF) does not show: what the CPU does between two GPU steps. tools/host_trace_report.py
// turns the file into per-segment means. Off (unset): one branch per mark. Marks are buffered and written when the
// buffer is full and at the end of a cycle, so the file does not stall the pass.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "backend/env.h"

namespace tona::backend {

class HostTrace {
public:
    // arg: layer, token count or step number of the mark; -1 if it has none
    static void mark(const char* what, int arg = -1) {
        HostTrace& t = instance();
        if (t.file_) t.add(what, arg, now_ns());
    }
    // A mark with a time measured earlier (inside a parallel region, where the buffer is not thread safe)
    static void mark_at(const char* what, int64_t ns, int arg = -1) {
        HostTrace& t = instance();
        if (t.file_) t.add(what, arg, ns);
    }
    static bool on() { return instance().file_ != nullptr; }
    static int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    HostTrace(const HostTrace&) = delete;
    HostTrace& operator=(const HostTrace&) = delete;

private:
    struct Rec {
        int64_t ns;
        const char* what;
        int arg;
    };

    HostTrace() {
        if (const char* path = env::text("TONA_HOST_TRACE")) file_ = std::fopen(path, "w");
        buf_.reserve(kFlushAt + 16);
    }
    ~HostTrace() {
        flush();
        if (file_) std::fclose(file_);
    }
    static HostTrace& instance() {
        static HostTrace t;
        return t;
    }

    void add(const char* what, int arg, int64_t ns) {
        buf_.push_back({ns, what, arg});
        // "cycle_end" is the last mark of a decode cycle: writing there keeps the write out of the passes
        if (buf_.size() >= kFlushAt || (what[0] == 'c' && what[1] == 'y')) flush();
    }

    void flush() {
        for (const Rec& r : buf_) std::fprintf(file_, "%lld %s %d\n", (long long)r.ns, r.what, r.arg);
        buf_.clear();
        std::fflush(file_);
    }

    static constexpr size_t kFlushAt = 8192;
    std::FILE* file_ = nullptr;
    std::vector<Rec> buf_;
};

// Times inside a parallel loop over rows: when the first worker row starts, when the first row ends. The marks go out
// in time order after the loop: <enter> (before the loop), <worker> (start of the earliest row from the second on,
// i.e. the earliest thread that is not the one that entered; the first row if there is only one), <first> (end of the
// earliest row), <exit> (after the loop). Off: one branch per call.
class RegionTrace {
public:
    RegionTrace(const char* enter, const char* worker, const char* first, const char* exit, int rows)
        : on_(HostTrace::on()), names_{enter, worker, first, exit} {
        if (on_) s_.assign((size_t)rows, 0), e_.assign((size_t)rows, 0);
    }
    void enter() const {
        if (on_) HostTrace::mark(names_[0]);
    }
    void row_begin(int r) {
        if (on_) s_[(size_t)r] = HostTrace::now_ns();
    }
    void row_end(int r) {
        if (on_) e_[(size_t)r] = HostTrace::now_ns();
    }
    void leave() const {
        if (!on_) return;
        int64_t w = INT64_MAX, f = INT64_MAX;
        for (size_t r = 0; r < s_.size(); r++) {
            if (r >= 1 || s_.size() == 1) w = std::min(w, s_[r]);
            f = std::min(f, e_[r]);
        }
        if (w <= f)
            HostTrace::mark_at(names_[1], w), HostTrace::mark_at(names_[2], f);
        else
            HostTrace::mark_at(names_[2], f), HostTrace::mark_at(names_[1], w);
        HostTrace::mark(names_[3]);
    }

private:
    bool on_;
    const char* names_[4];
    std::vector<int64_t> s_, e_;
};

}  // namespace tona::backend
