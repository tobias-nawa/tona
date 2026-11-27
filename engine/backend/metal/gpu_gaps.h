// Probe (TONA_GPU_GAPS=1): where the GPU waits between the command buffers of decode. Every decode command buffer is
// recorded with the host time it was prepared from, committed and waited on, and its GPU start and end (Metal's
// GPUStartTime/GPUEndTime share the host clock); each gap between two command buffers is split into
//   wake     GPU end of the earlier one until the CPU returns from its wait,
//   backend  host work of the backend after that wait and before the next command buffer's inputs (picks, copies),
//   outside  host work between two backend calls: shared code (sampling, acceptance, MTP drafting decisions),
//   inputs   writing the next command buffer's inputs and encoding it; of a target pass the probe also reports the
//            copy of the token history, the n-gram rows (row ids, read-ahead hint, copy) and the attention positions,
//            of an MTP pass the head's inputs (embedding rows, h); the rest is encoding, of target passes and of the
//            other command buffers (MTP head, commit) apart,
//   submit   commit until the GPU starts it,
//   split    a pass committed in two halves (TONA_METAL_EARLY_COMMIT): the GPU idle between them; of that the part
//            before the host committed the second half ("committed late"; the rest is the GPU starting it late),
//            the share of passes with more than 0.1 ms of it and the longest.
// The gap before the first command buffer after a cut is not idle of decode; of it the probe reports the longest
// submit (commit until the GPU starts it), e.g. the first decode pass after a long prefill.
// Off: nothing is recorded. On: one stderr line per 1000 target passes (or per N with TONA_GPU_GAPS=N > 1) with the
// means per target pass.
#pragma once

#include <mach/mach_time.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

#include "backend/env.h"

namespace tona::metal {

class GpuGaps {
public:
    // The backend calls whose command buffers are recorded
    enum Step : uint8_t { kTarget, kFill, kStep, kCommit, kSteps };

    // Parts of a target pass's inputs, seconds
    struct Parts {
        double history = 0, rows = 0, positions = 0, mtp = 0;
    };
    struct Cb {
        double prepared = 0, committed = 0, gpu_start = 0, gpu_end = 0, waited = 0;  // seconds, host clock
        Parts parts;
        double inner = 0;  // GPU idle inside [gpu_start, gpu_end) (split)
        double late = 0;   // of inner: until the host committed the second half
    };

    static int64_t every_from_env() {
        static const int64_t n = env::integer("TONA_GPU_GAPS", 0);
        return n == 1 ? 1000 : n;
    }
    static bool enabled() { return every_from_env() > 0; }

    static double now() {
        static const double scale = [] {
            mach_timebase_info_data_t tb{};
            mach_timebase_info(&tb);
            return (double)tb.numer / tb.denom * 1e-9;
        }();
        return (double)mach_absolute_time() * scale;
    }

    explicit GpuGaps(int64_t every = every_from_env()) : every_(every) {}

    // One backend call from construction to destruction
    class Scope {
    public:
        Scope(GpuGaps& g, Step s) : g_(g) {
            if (enabled()) g_.enter(s, now());
        }
        ~Scope() {
            if (enabled()) g_.leave(now());
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        GpuGaps& g_;
    };

    void enter(Step s, double t) { call_ = {s, t, ++calls_}, inside_ = true; }
    void leave(double t) { last_exit_ = t, exited_ = call_.id, inside_ = false; }
    // Nothing recorded happens between the last command buffer and the next one (prefill, a state copy): no gap
    void cut() { has_prev_ = false; }

    void add(const Cb& cb) {
        if (!inside_) return cut();  // a command buffer outside the recorded calls (prefill's MTP entries)
        if (has_prev_)
            gap(cb);
        else
            sum_.cold_submit = std::max(sum_.cold_submit, cb.gpu_start - cb.committed);
        if (call_.step == kTarget) target_passes_++, sum_.target_busy += cb.gpu_end - cb.gpu_start;
        sum_.busy += cb.gpu_end - cb.gpu_start - cb.inner, sum_.split += cb.inner, sum_.split_late += cb.late;
        sum_.split_big += cb.inner > kBigSplit, sum_.split_max = std::max(sum_.split_max, cb.inner);
        prev_ = cb, prev_call_ = call_, has_prev_ = true;
        if (target_passes_ >= every_) report();
    }

    // Means per target pass in ms, for the tests and the report
    struct Means {
        double busy = 0, idle = 0, wake = 0, backend = 0, outside = 0, inputs = 0, submit = 0, split = 0;
        double split_late = 0, split_big = 0, split_max = 0;  // of split: ms committed late; percent of target passes
                                                              // over 0.1 ms; the longest, ms
        double cold_submit = 0;  // longest submit of a first command buffer after a cut (prefill, a pause), ms
        double history = 0, rows = 0, positions = 0, mtp = 0;  // of inputs
        double encode_target = 0, encode_other = 0;            // the rest of inputs
        double by_step[kSteps][kSteps] = {};  // idle from a command buffer of call [a] to one of call [b]
    };
    Means means() const {
        Means m;
        const double k = target_passes_ ? 1e3 / (double)target_passes_ : 0;
        m.busy = sum_.busy * k, m.wake = sum_.wake * k, m.backend = sum_.backend * k, m.outside = sum_.outside * k;
        m.inputs = sum_.inputs * k, m.submit = sum_.submit * k, m.split = sum_.split * k;
        m.split_late = sum_.split_late * k, m.split_big = (double)sum_.split_big * k * 0.1,
        m.split_max = sum_.split_max * 1e3;
        m.cold_submit = sum_.cold_submit * 1e3;
        m.history = sum_.parts.history * k, m.rows = sum_.parts.rows * k, m.positions = sum_.parts.positions * k;
        m.mtp = sum_.parts.mtp * k, m.encode_target = sum_.encode_target * k, m.encode_other = sum_.encode_other * k;
        m.idle = m.wake + m.backend + m.outside + m.inputs + m.submit + m.split;
        for (int a = 0; a < kSteps; a++)
            for (int b = 0; b < kSteps; b++) m.by_step[a][b] = sum_.by_step[a][b] * k;
        return m;
    }

private:
    struct Call {
        Step step = kTarget;
        double entered = 0;
        uint64_t id = 0;
    };
    struct Sum {
        double busy = 0, target_busy = 0, wake = 0, backend = 0, outside = 0, inputs = 0, submit = 0, split = 0;
        double split_late = 0, split_max = 0, cold_submit = 0;
        int64_t split_big = 0;
        Parts parts;
        double encode_target = 0, encode_other = 0;
        double by_step[kSteps][kSteps] = {};
    };
    static constexpr double kMaxGap = 1.0;     // s: a longer gap is the time between requests, not decode
    static constexpr double kBigSplit = 1e-4;  // s

    void gap(const Cb& b) {
        const Cb& a = prev_;
        if (b.gpu_start - a.gpu_end > kMaxGap) return;
        const double wake = a.waited - a.gpu_end;
        double backend = 0, outside = 0;
        if (prev_call_.id == call_.id) {
            backend = b.prepared - a.waited;
        } else {
            const double exit = exited_ == prev_call_.id ? last_exit_ : a.waited;
            backend = (exit - a.waited) + (b.prepared - call_.entered);
            outside = call_.entered - exit;
        }
        const double inputs = b.committed - b.prepared, submit = b.gpu_start - b.committed;
        sum_.wake += wake, sum_.backend += backend, sum_.outside += outside, sum_.inputs += inputs;
        sum_.submit += submit;
        sum_.parts.history += b.parts.history, sum_.parts.rows += b.parts.rows;
        sum_.parts.positions += b.parts.positions, sum_.parts.mtp += b.parts.mtp;
        const double encode = inputs - b.parts.history - b.parts.rows - b.parts.positions - b.parts.mtp;
        (call_.step == kTarget ? sum_.encode_target : sum_.encode_other) += encode;
        sum_.by_step[prev_call_.step][call_.step] += wake + backend + outside + inputs + submit;
    }

    void report() {
        static const char* const names[kSteps] = {"target", "fill", "step", "commit"};
        const Means m = means();
        std::string steps;
        for (int a = 0; a < kSteps; a++)
            for (int b = 0; b < kSteps; b++)
                if (m.by_step[a][b] >= 0.05) {
                    char one[48];
                    std::snprintf(one, sizeof one, " %s->%s %.2f", names[a], names[b], m.by_step[a][b]);
                    steps += one;
                }
        std::fprintf(
            stderr,
            "GPU gaps per target pass: busy %.2f ms, idle %.2f ms (%.1f %%): wake %.2f, backend %.2f, "
            "outside %.2f, inputs %.2f (history %.2f, n-gram rows %.2f, positions %.2f, MTP inputs %.2f, "
            "encoding target %.2f, other %.2f), submit %.2f, split %.2f (committed late %.2f, %.1f %% of passes over "
            "0.1 ms, longest %.1f); first command buffer after a pause: longest submit %.1f; idle by step (ms):%s\n",
            m.busy, m.idle, m.busy + m.idle > 0 ? 100 * m.idle / (m.busy + m.idle) : 0.0, m.wake, m.backend, m.outside,
            m.inputs, m.history, m.rows, m.positions, m.mtp, m.encode_target, m.encode_other, m.submit, m.split,
            m.split_late, m.split_big, m.split_max, m.cold_submit, steps.c_str());
        sum_ = {}, target_passes_ = 0;
    }

    int64_t every_;
    Call call_, prev_call_;
    Cb prev_;
    bool has_prev_ = false, inside_ = false;
    uint64_t calls_ = 0, exited_ = 0;
    double last_exit_ = 0;
    int64_t target_passes_ = 0;
    Sum sum_;
};

}  // namespace tona::metal
